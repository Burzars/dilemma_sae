#!/usr/bin/env python3
"""Доступ к полному тексту книги по Gutenberg id.

Кластеризация (cluster_books.py) даёт clusters.json со списками id. Здесь — как по
этим id достать сам текст. Раскладка датасета Kaggle заранее точно не известна
(«Books AND Metadata»: тексты могут лежать либо ОТДЕЛЬНЫМИ файлами вида
`{id}.txt`/`pg{id}.txt`, либо КОЛОНКОЙ text/context в CSV/parquet), поэтому источник
определяется автоматически, а на крайний случай есть загрузка прямо с gutenberg.org
по id — она работает всегда, независимо от раскладки.

    # что вообще лежит в датасете (запустить на сервере — точный ответ про раскладку):
    python book_text.py --inspect --data-dir /path/to/dataset

    # текст одной книги:
    python book_text.py --id 1342 --data-dir /path/to/dataset

    # выгрузить все тексты кластера из clusters.json в папку:
    python book_text.py --cluster "Detective Fiction" --out outputs/texts
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S",
)
log = logging.getLogger("book_text")

_INT = re.compile(r"\d+")
_TEXT_COL_NAMES = {"context", "text", "content", "body", "fulltext", "booktext",
                   "rawtext", "plaintext", "book"}
# шаблоны URL Project Gutenberg (по убыванию частоты успеха)
_PG_URLS = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
    "https://www.gutenberg.org/ebooks/{id}.txt.utf-8",
]
_UA = "Mozilla/5.0 (dilemma_sae book_text.py; research use)"


# ---------------------------------------------------------------------------
# Определение источника текстов
# ---------------------------------------------------------------------------

@dataclass
class TextSource:
    file_index: dict[int, Path] = field(default_factory=dict)   # id → .txt файл
    table_path: Path | None = None                              # CSV/parquet с текстом
    table_id_col: str | None = None
    table_text_col: str | None = None

    @property
    def has_files(self) -> bool:
        return bool(self.file_index)

    @property
    def has_table(self) -> bool:
        return self.table_path is not None


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _find_text_table(data_dir: Path) -> tuple[Path, str, str] | None:
    """Ищет CSV/parquet, где есть и id-колонка, и колонка с полным текстом."""
    from gutenberg_cluster.load import _COLUMN_CANDIDATES  # переиспользуем детект id

    for path in sorted(data_dir.rglob("*.csv")) + sorted(data_dir.rglob("*.parquet")):
        try:
            if path.suffix == ".parquet":
                import pyarrow.parquet as pq
                cols = list(pq.ParquetFile(path).schema.names)
            else:
                with open(path, encoding="utf-8", errors="ignore", newline="") as f:
                    cols = next(csv.reader(f))
        except Exception as e:  # noqa: BLE001
            log.debug("skip %s: %s", path.name, e)
            continue
        norm = {_norm(c): c for c in cols}
        id_col = next((norm[cand] for cand in _COLUMN_CANDIDATES["id"] if cand in norm), None)
        # текстовая колонка обязана отличаться от id (иначе 'Text#'→'text' даёт ложное совпадение)
        text_col = next((norm[n] for n in norm if n in _TEXT_COL_NAMES and norm[n] != id_col), None)
        if text_col and id_col:
            return path, id_col, text_col
    return None


def discover_text_source(data_dir: Path, min_files: int = 20) -> TextSource:
    """Автоопределение, как хранятся тексты: отдельные файлы и/или колонка таблицы."""
    data_dir = Path(data_dir)
    src = TextSource()

    txts = list(data_dir.rglob("*.txt"))
    if len(txts) >= min_files:
        for p in txts:
            m = _INT.search(p.stem)
            if m:
                src.file_index.setdefault(int(m.group()), p)
        log.info("Найдено %d .txt-файлов → индекс по id (%d уникальных id)",
                 len(txts), len(src.file_index))

    table = _find_text_table(data_dir)
    if table:
        src.table_path, src.table_id_col, src.table_text_col = table
        log.info("Найдена таблица с текстом: %s (id=%s, text=%s)",
                 src.table_path.name, src.table_id_col, src.table_text_col)

    if not src.has_files and not src.has_table:
        log.info("Локальных текстов не найдено — будет загрузка с gutenberg.org по id")
    return src


# ---------------------------------------------------------------------------
# Извлечение текста
# ---------------------------------------------------------------------------

_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)
_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)


def strip_pg_boilerplate(text: str) -> str:
    """Убирает шапку/подвал Project Gutenberg, оставляя тело книги."""
    s = _START.search(text)
    if s:
        text = text[s.end():]
    e = _END.search(text)
    if e:
        text = text[:e.start()]
    return text.strip()


def _read_table_texts(src: TextSource, ids: set[int]) -> dict[int, str]:
    """Тексты нужных id из таблицы за один проход (без загрузки всего в память)."""
    import pandas as pd

    found: dict[int, str] = {}
    path = src.table_path
    if path.suffix == ".parquet":
        import pyarrow.dataset as ds
        dataset = ds.dataset(path)
        table = dataset.to_table(columns=[src.table_id_col, src.table_text_col])
        for _id, _txt in zip(table[src.table_id_col].to_pylist(),
                             table[src.table_text_col].to_pylist()):
            gid = _INT.search(str(_id))
            if gid and int(gid.group()) in ids:
                found[int(gid.group())] = str(_txt)
    else:  # CSV чанками
        for chunk in pd.read_csv(path, dtype=str, chunksize=2000,
                                 usecols=[src.table_id_col, src.table_text_col]):
            for _id, _txt in zip(chunk[src.table_id_col], chunk[src.table_text_col]):
                gid = _INT.search(str(_id))
                if gid and int(gid.group()) in ids and int(gid.group()) not in found:
                    found[int(gid.group())] = "" if _txt is None else str(_txt)
            if len(found) == len(ids):
                break
    return found


def _download_pg(book_id: int, timeout: float = 30.0) -> str | None:
    """Скачивает текст книги с gutenberg.org, перебирая известные шаблоны URL."""
    for tmpl in _PG_URLS:
        url = tmpl.format(id=book_id)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if r.status == 200:
                    return r.read().decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001 — пробуем следующий шаблон
            log.debug("PG %s → %s", url, e)
    log.warning("Не удалось скачать книгу %d с gutenberg.org", book_id)
    return None


class BookTextResolver:
    """id → полный текст: локальные файлы → таблица → загрузка с gutenberg.org."""

    def __init__(self, data_dir: Path | None, allow_download: bool = True,
                 cache_dir: Path | None = None):
        self.source = discover_text_source(data_dir) if data_dir else TextSource()
        self.allow_download = allow_download
        self.cache_dir = Path(cache_dir) if cache_dir else None
        # каталог кэша создаём лениво — только когда реально что-то скачиваем

    def get_texts(self, ids) -> dict[int, str | None]:
        wanted = {int(i) for i in ids}
        out: dict[int, str | None] = {}

        for i in list(wanted):
            if i in self.source.file_index:
                out[i] = self.source.file_index[i].read_text(encoding="utf-8", errors="replace")
        remaining = wanted - out.keys()

        if remaining and self.source.has_table:
            out.update(_read_table_texts(self.source, remaining))
            remaining = wanted - out.keys()

        if remaining and self.allow_download:
            for i in remaining:
                cached = self.cache_dir / f"{i}.txt" if self.cache_dir else None
                if cached and cached.exists():
                    out[i] = cached.read_text(encoding="utf-8", errors="replace")
                    continue
                text = _download_pg(i)
                out[i] = text
                if text and cached:
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    cached.write_text(text, encoding="utf-8")

        for i in wanted - out.keys():
            out[i] = None
        return out

    def get_text(self, book_id: int) -> str | None:
        return self.get_texts([book_id])[int(book_id)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_data_dir(data_dir: str | None) -> Path | None:
    """Путь к датасету: явный аргумент, иначе config.yaml, иначе Kaggle."""
    if data_dir:
        return Path(data_dir)
    import yaml
    cfg_path = SCRIPT_DIR / "config.yaml"
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        d = (cfg.get("data") or {}).get("data_dir")
        if d:
            return Path(d)
        kaggle_id = (cfg.get("data") or {}).get("kaggle_id")
        if kaggle_id:
            import kagglehub
            log.info("Скачиваю датасет с Kaggle: %s", kaggle_id)
            return Path(kagglehub.dataset_download(kaggle_id))
    return None


def _inspect(data_dir: Path | None) -> None:
    """Печатает, как в датасете хранятся тексты (прямой ответ на 'как реализуется')."""
    if not data_dir or not Path(data_dir).exists():
        log.info("data_dir не задан/не существует — локальной раскладки нет, "
                 "текст берётся загрузкой с gutenberg.org по id")
        return
    data_dir = Path(data_dir)
    txts = list(data_dir.rglob("*.txt"))
    tables = sorted(data_dir.rglob("*.csv")) + sorted(data_dir.rglob("*.parquet"))
    print(f"\nДатасет: {data_dir}")
    print(f"  .txt файлов: {len(txts)}")
    for p in txts[:3]:
        print(f"    пример: {p.relative_to(data_dir)}  ({p.stat().st_size} байт)")
    print(f"  таблиц (csv/parquet): {len(tables)}")
    for p in tables:
        print(f"    {p.relative_to(data_dir)}  ({p.stat().st_size // 1024} КБ)")
    src = discover_text_source(data_dir)
    print("\nВывод:")
    if src.has_files:
        print(f"  → тексты ОТДЕЛЬНЫМИ файлами, {len(src.file_index)} шт., доступ по id из имени файла")
    if src.has_table:
        print(f"  → тексты в КОЛОНКЕ '{src.table_text_col}' файла {src.table_path.name} "
              f"(id-колонка '{src.table_id_col}')")
    if not src.has_files and not src.has_table:
        print("  → локальных текстов нет; book_text.py скачает по id с gutenberg.org")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None, help="папка датасета (иначе из config.yaml / Kaggle)")
    p.add_argument("--inspect", action="store_true", help="показать раскладку текстов и выйти")
    p.add_argument("--id", type=int, default=None, help="напечатать текст этой книги")
    p.add_argument("--cluster", default=None, help="выгрузить тексты книг этого кластера")
    p.add_argument("--clusters-json", default=str(SCRIPT_DIR / "outputs" / "clusters.json"))
    p.add_argument("--out", default=None, help="папка для сохранения текстов (для --cluster)")
    p.add_argument("--no-download", action="store_true", help="не ходить в gutenberg.org")
    p.add_argument("--strip-headers", action="store_true", help="убрать шапку/подвал Gutenberg")
    args = p.parse_args()

    data_dir = _resolve_data_dir(args.data_dir)

    if args.inspect:
        _inspect(data_dir)
        return

    resolver = BookTextResolver(
        data_dir, allow_download=not args.no_download,
        cache_dir=SCRIPT_DIR / "outputs" / "text_cache",
    )

    if args.id is not None:
        text = resolver.get_text(args.id)
        if text is None:
            log.error("Текст книги %d не найден", args.id)
            sys.exit(1)
        if args.strip_headers:
            text = strip_pg_boilerplate(text)
        sys.stdout.write(text)
        return

    if args.cluster:
        clusters = json.loads(Path(args.clusters_json).read_text(encoding="utf-8"))
        match = next((c for c in clusters if c["cluster_name"] == args.cluster), None)
        if not match:
            log.error("Кластер %r не найден в %s", args.cluster, args.clusters_json)
            sys.exit(1)
        out_dir = Path(args.out or (SCRIPT_DIR / "outputs" / "texts"))
        out_dir.mkdir(parents=True, exist_ok=True)
        texts = resolver.get_texts(match["ids"])
        n_ok = 0
        for book_id, text in texts.items():
            if text is None:
                continue
            if args.strip_headers:
                text = strip_pg_boilerplate(text)
            (out_dir / f"{book_id}.txt").write_text(text, encoding="utf-8")
            n_ok += 1
        log.info("Сохранено %d/%d текстов кластера %r → %s",
                 n_ok, len(match["ids"]), args.cluster, out_dir)
        return

    p.print_help()


if __name__ == "__main__":
    main()
