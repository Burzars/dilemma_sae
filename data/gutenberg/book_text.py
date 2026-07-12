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
import tarfile
import urllib.request
import zipfile
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
    archive_path: Path | None = None                            # zip/tar с текстами
    archive_index: dict[int, str] = field(default_factory=dict) # id → имя файла внутри архива

    @property
    def has_files(self) -> bool:
        return bool(self.file_index)

    @property
    def has_table(self) -> bool:
        return self.table_path is not None

    @property
    def has_archive(self) -> bool:
        return self.archive_path is not None and bool(self.archive_index)


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


def _archive_member_names(path: Path) -> list[str]:
    """Имена файлов внутри архива (zip — быстро, из оглавления; tar — сканом)."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            return [i.filename for i in z.infolist() if not i.is_dir()]
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as t:
            return [m.name for m in t.getmembers() if m.isfile()]
    return []


def _index_archive_members(names: list[str]) -> dict[int, str]:
    """id → имя файла внутри архива (id из имени файла; приоритет .txt)."""
    txt = [n for n in names if n.lower().endswith(".txt")]
    pool = txt if txt else names
    idx: dict[int, str] = {}
    for n in pool:
        m = _INT.search(n.rsplit("/", 1)[-1])
        if m:
            idx.setdefault(int(m.group()), n)
    return idx


def _find_text_archive(data_dir: Path, min_files: int) -> tuple[Path, dict[int, str]] | None:
    """Ищет zip/tar с текстами книг (файлы с id в имени)."""
    globs = ["*.zip", "*.tar", "*.tar.gz", "*.tgz", "*.tar.bz2"]
    for path in sorted(p for g in globs for p in data_dir.rglob(g)):
        try:
            idx = _index_archive_members(_archive_member_names(path))
        except Exception as e:  # noqa: BLE001
            log.debug("skip archive %s: %s", path.name, e)
            continue
        if len(idx) >= min_files:
            return path, idx
    return None


def discover_text_source(data_dir: Path, min_files: int = 20) -> TextSource:
    """Автоопределение раскладки текстов: файлы / колонка таблицы / архив zip-tar."""
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

    if not src.has_files:
        archive = _find_text_archive(data_dir, min_files)
        if archive:
            src.archive_path, src.archive_index = archive
            log.info("Найден архив с текстами: %s (%d файлов по id, без распаковки на диск)",
                     src.archive_path.name, len(src.archive_index))

    if not src.has_files and not src.has_table and not src.has_archive:
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


def _iter_table_texts(src: TextSource, wanted: set[int]):
    """Стриминговый обход таблицы: отдаёт (id, text) по одному, память ограничена.

    В отличие от _read_table_texts не копит всё в dict — для выгрузки десятков тысяч
    книг (иначе весь корпус текстов оказался бы в RAM).
    """
    path = src.table_path
    if path.suffix == ".parquet":
        import pyarrow.dataset as ds
        scanner = ds.dataset(path).scanner(columns=[src.table_id_col, src.table_text_col])
        for batch in scanner.to_batches():
            ids_col = batch.column(0).to_pylist()
            txt_col = batch.column(1).to_pylist()
            for _id, _txt in zip(ids_col, txt_col):
                m = _INT.search(str(_id))
                if m and int(m.group()) in wanted:
                    yield int(m.group()), "" if _txt is None else str(_txt)
    else:  # CSV чанками
        import pandas as pd
        for chunk in pd.read_csv(path, dtype=str, chunksize=2000,
                                 usecols=[src.table_id_col, src.table_text_col]):
            for _id, _txt in zip(chunk[src.table_id_col], chunk[src.table_text_col]):
                m = _INT.search(str(_id))
                if m and int(m.group()) in wanted:
                    yield int(m.group()), "" if _txt is None else str(_txt)


def _open_archive(path: Path):
    """Открывает архив на чтение; отдаёт ('zip'|'tar', handle) для random-access."""
    if zipfile.is_zipfile(path):
        return "zip", zipfile.ZipFile(path)
    return "tar", tarfile.open(path)


def _read_member(kind: str, handle, member: str) -> str:
    """Читает один файл внутри архива по имени (без распаковки остального)."""
    if kind == "zip":
        return handle.read(member).decode("utf-8", errors="replace")
    f = handle.extractfile(member)
    return f.read().decode("utf-8", errors="replace") if f else ""


def _read_archive_texts(src: TextSource, ids: set[int]) -> dict[int, str]:
    """Тексты нужных id из архива; архив открывается один раз."""
    out: dict[int, str] = {}
    kind, handle = _open_archive(src.archive_path)
    try:
        for i in ids:
            member = src.archive_index.get(i)
            if member is not None:
                out[i] = _read_member(kind, handle, member)
    finally:
        handle.close()
    return out


def _iter_archive_texts(src: TextSource, wanted: set[int]):
    """Стриминговый обход архива: (id, text) по одному, архив открыт один раз."""
    kind, handle = _open_archive(src.archive_path)
    try:
        for i, member in src.archive_index.items():
            if i in wanted:
                yield i, _read_member(kind, handle, member)
    finally:
        handle.close()


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

        if remaining and self.source.has_archive:
            out.update(_read_archive_texts(self.source, remaining))
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


def _slug(name: str) -> str:
    """Имя кластера → безопасное имя папки."""
    s = re.sub(r"[^\w.-]+", "_", str(name), flags=re.UNICODE).strip("_")
    return s[:80] or "cluster"


def dump_all(resolver: BookTextResolver, ids, out_dir: Path,
             cluster_of: dict[int, str] | None = None, strip: bool = False,
             skip_existing: bool = True, download: bool = True, workers: int = 8) -> None:
    """Массовая выгрузка текстов в файлы {out_dir}/[кластер/]{id}.txt.

    Стриминг (по одному тексту в памяти) + резюмируемость (пропуск уже записанных),
    поэтому годится для десятков тысяч книг и переживает прерывание.
    """
    from concurrent.futures import ThreadPoolExecutor

    out_dir = Path(out_dir)

    def target(i: int) -> Path:
        base = out_dir / _slug(cluster_of[i]) if (cluster_of and i in cluster_of) else out_dir
        return base / f"{i}.txt"

    wanted = {int(i) for i in ids}
    if skip_existing:
        already = {i for i in wanted if target(i).exists() and target(i).stat().st_size > 0}
        if already:
            log.info("Пропускаю %d уже сохранённых (резюмирую)", len(already))
        wanted -= already
    total = len(wanted)
    log.info("К выгрузке: %d текстов → %s", total, out_dir)

    done = 0
    missing: list[int] = []

    def write(i: int, text: str | None) -> None:
        nonlocal done
        if not text:
            missing.append(i)
            return
        if strip:
            text = strip_pg_boilerplate(text)
        t = target(i)
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text(text, encoding="utf-8")
        done += 1
        if done % 500 == 0:
            log.info("... сохранено %d/%d", done, total)

    src = resolver.source
    remaining = set(wanted)

    # 1) локальные файлы {id}.txt — просто копия с диска
    for i in list(remaining):
        if i in src.file_index:
            write(i, src.file_index[i].read_text(encoding="utf-8", errors="replace"))
            remaining.discard(i)

    # 2) колонка таблицы — один потоковый проход
    if remaining and src.has_table:
        for i, text in _iter_table_texts(src, remaining):
            if i in remaining:
                write(i, text)
                remaining.discard(i)

    # 3) архив zip/tar — random-access по id, без распаковки на диск
    if remaining and src.has_archive:
        for i, text in _iter_archive_texts(src, remaining):
            if i in remaining:
                write(i, text)
                remaining.discard(i)

    # 4) остаток — загрузка с gutenberg.org (только если тексты не в датасете)
    if remaining and download:
        log.warning("%d текстов нет локально — качаю с gutenberg.org в %d потоков "
                    "(медленно; для полного корпуса лучше иметь тексты в датасете)",
                    len(remaining), workers)
        rem_list = list(remaining)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, text in zip(rem_list, ex.map(_download_pg, rem_list)):
                write(i, text)
    elif remaining:
        missing.extend(remaining)

    log.info("ГОТОВО: записано %d, не найдено %d (из %d к выгрузке)", done, len(missing), total)
    if missing:
        miss_file = out_dir / "_missing_ids.txt"
        out_dir.mkdir(parents=True, exist_ok=True)
        miss_file.write_text("\n".join(map(str, sorted(missing))), encoding="utf-8")
        log.info("Список ненайденных id → %s", miss_file)


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
    archives = sorted(p for g in ("*.zip", "*.tar", "*.tar.gz", "*.tgz", "*.tar.bz2")
                      for p in data_dir.rglob(g))
    print(f"\nДатасет: {data_dir}")
    print(f"  .txt файлов: {len(txts)}")
    for p in txts[:3]:
        print(f"    пример: {p.relative_to(data_dir)}  ({p.stat().st_size} байт)")
    print(f"  таблиц (csv/parquet): {len(tables)}")
    for p in tables:
        print(f"    {p.relative_to(data_dir)}  ({p.stat().st_size // 1024} КБ)")
    print(f"  архивов (zip/tar): {len(archives)}")
    for p in archives:
        print(f"    {p.relative_to(data_dir)}  ({p.stat().st_size // 1024 // 1024} МБ)")
    src = discover_text_source(data_dir)
    print("\nВывод:")
    if src.has_files:
        print(f"  → тексты ОТДЕЛЬНЫМИ файлами, {len(src.file_index)} шт., доступ по id из имени файла")
    if src.has_table:
        print(f"  → тексты в КОЛОНКЕ '{src.table_text_col}' файла {src.table_path.name} "
              f"(id-колонка '{src.table_id_col}')")
    if src.has_archive:
        print(f"  → тексты в АРХИВЕ {src.archive_path.name}, {len(src.archive_index)} файлов; "
              f"читаются по id без распаковки")
    if not src.has_files and not src.has_table and not src.has_archive:
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
    p.add_argument("--dump-all", action="store_true",
                   help="выгрузить тексты ВСЕХ книг из clusters.json")
    p.add_argument("--by-cluster", action="store_true",
                   help="раскладывать по подпапкам кластеров (для --dump-all/--cluster)")
    p.add_argument("--limit", type=int, default=None, help="ограничить число книг (тест)")
    p.add_argument("--workers", type=int, default=8, help="потоков на загрузку с gutenberg.org")
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

    if args.dump_all:
        clusters = json.loads(Path(args.clusters_json).read_text(encoding="utf-8"))
        cluster_of = {int(i): c["cluster_name"] for c in clusters for i in c["ids"]}
        ids = list(cluster_of.keys())
        if args.limit:
            ids = ids[:args.limit]
        out_dir = Path(args.out or (SCRIPT_DIR / "outputs" / "texts"))
        dump_all(resolver, ids, out_dir,
                 cluster_of=cluster_of if args.by_cluster else None,
                 strip=args.strip_headers, download=not args.no_download, workers=args.workers)
        return

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
