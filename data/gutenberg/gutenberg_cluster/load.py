"""Загрузка и парсинг метаданных книг Gutenberg.

Kaggle-страница рендерится JS, точные имена колонок заранее неизвестны, поэтому
колонки детектим в рантайме (case-insensitive), а поле subject разбираем на
LoCC-коды и LCSH-темы регуляркой — это работает и когда коды лежат в отдельной
колонке, и когда они смешаны с темами в одном поле (как в примере пользователя).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

# lc_classes лежит рядом с пакетом (data/gutenberg/), добавляется в sys.path
# через cluster_books.py; при прямом импорте пробуем оба варианта.
try:
    from lc_classes import LC_CLASS, LC_SUBCLASS
except ImportError:  # pragma: no cover
    from ..lc_classes import LC_CLASS, LC_SUBCLASS  # type: ignore

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Детект колонок
# ---------------------------------------------------------------------------

# Кандидаты нормализованных имён по порядку предпочтения (первое найденное — берём).
_COLUMN_CANDIDATES: dict[str, list[str]] = {
    "id":        ["etextnumber", "text", "textno", "etextno", "etext", "ebookno",
                  "ebook", "gutenbergid", "bookid", "id", "no"],
    "title":     ["title", "booktitle", "name"],
    "author":    ["authors", "author", "creator"],
    "language":  ["language", "languages", "lang"],
    "type":      ["type", "mediatype"],
    "subject":   ["subjects", "subject"],
    "bookshelf": ["bookshelves", "bookshelf", "shelves", "shelf"],
    "locc":      ["locc", "loc", "lcc", "classification"],
}


def _norm(name: str) -> str:
    """Нормализация имени колонки: lower + только буквы/цифры ('Text#' → 'text')."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def detect_columns(columns: list[str]) -> dict[str, str]:
    """Сопоставляет канонические поля реальным именам колонок CSV.

    Возвращает {canonical: actual_column_name} только для найденных полей.
    """
    norm_to_actual: dict[str, str] = {}
    for col in columns:
        norm_to_actual.setdefault(_norm(col), col)  # первое вхождение выигрывает

    mapping: dict[str, str] = {}
    for canonical, candidates in _COLUMN_CANDIDATES.items():
        for cand in candidates:
            if cand in norm_to_actual:
                mapping[canonical] = norm_to_actual[cand]
                break
    return mapping


# ---------------------------------------------------------------------------
# Парсинг полей
# ---------------------------------------------------------------------------

_LOCC_WITH_DIGIT = re.compile(r"^[A-Z]{1,3}\d")   # 'E201', 'PS3537' — буквы+цифра
_ID_DIGITS = re.compile(r"\d+")
_BROWSING_PREFIX = re.compile(r"^browsing\s*:\s*", re.IGNORECASE)

# Немного полных названий языков → ISO-коды (на случай, если в колонке не коды).
_NAME2CODE = {
    "english": "en", "french": "fr", "german": "de", "spanish": "es",
    "italian": "it", "dutch": "nl", "portuguese": "pt", "latin": "la",
    "greek": "el", "russian": "ru", "finnish": "fi", "swedish": "sv",
    "danish": "da", "polish": "pl", "hungarian": "hu", "chinese": "zh",
    "japanese": "ja",
}


def _split_semicolon(raw: object) -> list[str]:
    """Разбивает поле на токены по ';' (ловит и '; ', и ' ; '), чистит пустые."""
    if raw is None or (isinstance(raw, float)):
        return []
    text = str(raw).strip()
    if not text or text.lower() in ("nan", "none"):
        return []
    return [tok.strip() for tok in text.split(";") if tok.strip()]


def _is_locc_code(token: str) -> bool:
    """True, если токен похож на LoCC-код (а не на LCSH-тему)."""
    t = token.strip()
    if not t or " " in t:                 # темы LCSH содержат пробелы/'--'
        return False
    if _LOCC_WITH_DIGIT.match(t):          # 'E201', 'JK75', 'PS3537'
        return True
    up = t.upper()
    # буквенные коды без цифр ('JK', 'PZ', 'E') — только если это известный LC-класс
    return up in LC_SUBCLASS or up in LC_CLASS


def parse_subject_field(raw: object) -> tuple[list[str], list[str]]:
    """Разбирает поле subject на (locc_codes, lcsh_subjects).

    Работает и когда коды смешаны с темами в одном поле, и когда поле — чисто темы.
    """
    locc: list[str] = []
    subjects: list[str] = []
    for tok in _split_semicolon(raw):
        if _is_locc_code(tok):
            locc.append(tok.upper())
        else:
            subjects.append(tok)
    return locc, subjects


def parse_bookshelf_field(raw: object) -> list[str]:
    """Список полок; срезает префикс 'Browsing:' и дедуплицирует, сохраняя порядок."""
    seen: set[str] = set()
    out: list[str] = []
    for tok in _split_semicolon(raw):
        shelf = _BROWSING_PREFIX.sub("", tok).strip()
        if shelf and shelf.lower() not in seen:
            seen.add(shelf.lower())
            out.append(shelf)
    return out


def parse_id(raw: object) -> int | None:
    """Достаёт целочисленный Gutenberg id ('PG1234'/'1234' → 1234)."""
    if raw is None:
        return None
    m = _ID_DIGITS.search(str(raw))
    return int(m.group()) if m else None


def _norm_langs(raw: object) -> set[str]:
    """Множество ISO-кодов языков строки ('en', 'English', "['en', 'fr']" → {en,fr})."""
    if raw is None:
        return set()
    codes: set[str] = set()
    for tok in re.split(r"[^A-Za-z]+", str(raw).lower()):
        if not tok:
            continue
        if len(tok) == 2:
            codes.add(tok)
        elif tok in _NAME2CODE:
            codes.add(_NAME2CODE[tok])
    return codes


# ---------------------------------------------------------------------------
# Поиск CSV и загрузка
# ---------------------------------------------------------------------------

def find_metadata_csv(data_dir: Path) -> Path:
    """Находит CSV с метаданными в папке датасета (или принимает путь к самому CSV)."""
    data_dir = Path(data_dir)
    if data_dir.is_file():
        return data_dir

    candidates = sorted(data_dir.rglob("*.csv")) + sorted(data_dir.rglob("*.csv.gz"))
    if not candidates:
        raise FileNotFoundError(f"CSV не найден в {data_dir}")
    if len(candidates) == 1:
        return candidates[0]

    # несколько CSV — берём тот, где есть title+subject и максимум колонок
    best: tuple[int, Path] | None = None
    for path in candidates:
        try:
            header = pd.read_csv(path, nrows=0)
        except Exception as e:  # noqa: BLE001
            log.warning("Не смог прочитать заголовок %s: %s", path.name, e)
            continue
        cols = detect_columns(list(header.columns))
        score = len(header.columns) + (1000 if {"title", "subject"} <= cols.keys() else 0)
        if best is None or score > best[0]:
            best = (score, path)
    if best is None:
        raise FileNotFoundError(f"Не удалось выбрать метаданные среди {len(candidates)} CSV в {data_dir}")
    return best[1]


def load_books(
    data_dir: Path,
    languages: list[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Читает CSV, детектит колонки, парсит поля, фильтрует по типу/языку.

    Возвращает DataFrame: id(int), title, author, language, subjects(list),
    locc(list), shelves(list). По одной строке на уникальный id.
    """
    csv_path = find_metadata_csv(data_dir)
    log.info("Читаю метаданные: %s", csv_path)
    raw = pd.read_csv(csv_path, dtype=str, on_bad_lines="skip")
    log.info("Строк в CSV: %d, колонок: %d", len(raw), len(raw.columns))

    cols = detect_columns(list(raw.columns))
    log.info("Детект колонок: %s", cols)
    for required in ("id", "subject"):
        if required not in cols:
            raise KeyError(
                f"В CSV не найдена колонка '{required}'. Колонки: {list(raw.columns)}"
            )

    df = pd.DataFrame()
    df["id"] = raw[cols["id"]].map(parse_id)
    df["title"] = raw[cols["title"]].fillna("") if "title" in cols else ""
    df["author"] = raw[cols["author"]].fillna("") if "author" in cols else ""
    df["language"] = raw[cols["language"]].fillna("") if "language" in cols else ""

    # subject → (locc, subjects); при наличии отдельной колонки locc — объединяем
    parsed = raw[cols["subject"]].map(parse_subject_field)
    df["locc"] = parsed.map(lambda p: p[0])
    df["subjects"] = parsed.map(lambda p: p[1])
    if "locc" in cols:
        extra = raw[cols["locc"]].map(lambda v: [c.upper() for c in _split_semicolon(v)])
        df["locc"] = [sorted(set(a) | set(b)) for a, b in zip(df["locc"], extra)]

    df["shelves"] = (
        raw[cols["bookshelf"]].map(parse_bookshelf_field)
        if "bookshelf" in cols else [[] for _ in range(len(raw))]
    )

    # тип: оставляем только текстовые книги
    n0 = len(df)
    if "type" in cols:
        is_text = raw[cols["type"]].fillna("Text").str.strip().str.lower().eq("text")
        df = df[is_text.values]
        log.info("Фильтр Type==Text: %d → %d", n0, len(df))

    # валидный id
    df = df[df["id"].notna()].copy()
    df["id"] = df["id"].astype(int)
    df = df.drop_duplicates(subset="id", keep="first")

    # язык
    if languages:
        wanted = {c.lower() for c in languages}
        lang_codes = df["language"].map(_norm_langs)
        keep = lang_codes.map(lambda s: (not s) or bool(s & wanted))
        n_before = len(df)
        n_unparsed = int((lang_codes.map(len) == 0).sum())
        df = df[keep.values].copy()
        log.info(
            "Фильтр языков %s: %d → %d (не распознан язык у %d — оставлены)",
            sorted(wanted), n_before, len(df), n_unparsed,
        )

    df = df.reset_index(drop=True)
    if limit is not None and limit < len(df):
        df = df.sample(n=limit, random_state=0).reset_index(drop=True)
        log.info("Ограничение limit=%d применено", limit)

    log.info("Готово: %d книг", len(df))
    return df
