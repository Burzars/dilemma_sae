"""Карта кодов Library of Congress Classification (LoCC) → человекочитаемые имена.

Датасет Gutenberg хранит в поле subject смесь LCSH-тем и LoCC-кодов
(`E201`, `JK`, `PS3537`, `TX714`, ...). Первая одна-три буквы кода задают класс
(`P` = Language and Literature) и подкласс (`PS` = American literature). Здесь —
статические карты для обогащения имён кластеров: по коду отдаём самый специфичный
известный ярлык (сначала 2-буквенный подкласс, потом 1-буквенный класс).

Ничего не импортирует — чистые данные + пара функций.
"""

from __future__ import annotations

import re

# Верхний уровень LC (одна буква).
LC_CLASS: dict[str, str] = {
    "A": "General Works",
    "B": "Philosophy, Psychology & Religion",
    "C": "Auxiliary Sciences of History",
    "D": "World History",
    "E": "History of the Americas",
    "F": "History of the Americas",
    "G": "Geography, Anthropology & Recreation",
    "H": "Social Sciences",
    "J": "Political Science",
    "K": "Law",
    "L": "Education",
    "M": "Music",
    "N": "Fine Arts",
    "P": "Language & Literature",
    "Q": "Science",
    "R": "Medicine",
    "S": "Agriculture",
    "T": "Technology",
    "U": "Military Science",
    "V": "Naval Science",
    "Z": "Bibliography & Library Science",
}

# Двухбуквенные подклассы (перечислены самые полезные для жанров/тем; список
# не исчерпывающий, но покрывает основную массу художки и науки в Gutenberg).
LC_SUBCLASS: dict[str, str] = {
    # A — General Works
    "AE": "Encyclopedias", "AI": "Indexes", "AM": "Museums",
    "AN": "Newspapers", "AP": "Periodicals", "AY": "Yearbooks & Almanacs",
    "AZ": "History of Scholarship",
    # B — Philosophy, Psychology, Religion
    "BC": "Logic", "BD": "Speculative Philosophy", "BF": "Psychology",
    "BH": "Aesthetics", "BJ": "Ethics", "BL": "Religions & Mythology",
    "BM": "Judaism", "BP": "Islam & Bahai", "BQ": "Buddhism",
    "BR": "Christianity", "BS": "The Bible", "BT": "Doctrinal Theology",
    "BV": "Practical Theology", "BX": "Christian Denominations",
    # C — Auxiliary Sciences of History
    "CB": "History of Civilization", "CC": "Archaeology", "CE": "Chronology",
    "CJ": "Numismatics", "CR": "Heraldry", "CS": "Genealogy", "CT": "Biography",
    # D — World History
    "DA": "History of Great Britain", "DB": "History of Austria",
    "DC": "History of France", "DD": "History of Germany",
    "DE": "Greco-Roman World", "DF": "History of Greece", "DG": "History of Italy",
    "DK": "History of Russia", "DL": "History of Northern Europe",
    "DP": "History of Spain & Portugal", "DR": "History of the Balkans",
    "DS": "History of Asia", "DT": "History of Africa", "DU": "History of Oceania",
    # E/F — Americas
    "DA_": "",  # placeholder to keep formatting tidy
    # G — Geography, Anthropology, Recreation
    "GB": "Physical Geography", "GC": "Oceanography", "GE": "Environmental Sciences",
    "GN": "Anthropology", "GR": "Folklore", "GT": "Manners & Customs",
    "GV": "Recreation & Sports",
    # H — Social Sciences
    "HA": "Statistics", "HB": "Economic Theory", "HC": "Economic History",
    "HD": "Industry, Land & Labor", "HE": "Transportation & Communications",
    "HF": "Commerce", "HG": "Finance", "HJ": "Public Finance", "HM": "Sociology",
    "HN": "Social History", "HQ": "Family, Marriage & Gender", "HS": "Societies",
    "HT": "Communities & Social Classes", "HV": "Social Welfare & Criminology",
    "HX": "Socialism & Communism",
    # J — Political Science
    "JA": "Political Science", "JC": "Political Theory", "JF": "Political Institutions",
    "JK": "US Politics & Government", "JL": "Politics of the Americas",
    "JN": "Politics of Europe", "JQ": "Politics of Asia & Africa",
    "JS": "Local Government", "JV": "Colonies & Emigration",
    "JX": "International Law", "JZ": "International Relations",
    # K — Law
    "KD": "Law of the United Kingdom", "KF": "Law of the United States",
    # L — Education
    "LA": "History of Education", "LB": "Theory & Practice of Education",
    "LC": "Special Aspects of Education",
    # M — Music
    "ML": "Literature on Music", "MT": "Musical Instruction",
    # N — Fine Arts
    "NA": "Architecture", "NB": "Sculpture", "NC": "Drawing & Illustration",
    "ND": "Painting", "NE": "Printmaking", "NK": "Decorative Arts",
    "NX": "Arts in General",
    # P — Language and Literature
    "PA": "Classical Languages & Literature", "PB": "Modern & Celtic Languages",
    "PC": "Romance Languages", "PD": "Germanic Languages", "PE": "English Language",
    "PF": "West Germanic Languages", "PG": "Slavic Languages & Literature",
    "PH": "Finno-Ugrian Languages", "PJ": "Oriental Languages",
    "PK": "Indo-Iranian Languages", "PL": "East Asian & African Languages",
    "PM": "Indigenous American Languages", "PN": "Literature, Drama & Journalism",
    "PQ": "Romance Literatures", "PR": "English Literature",
    "PS": "American Literature", "PT": "Germanic Literature",
    "PZ": "Fiction & Juvenile Literature",
    # Q — Science
    "QA": "Mathematics & Computing", "QB": "Astronomy", "QC": "Physics",
    "QD": "Chemistry", "QE": "Geology", "QH": "Natural History & Biology",
    "QK": "Botany", "QL": "Zoology", "QM": "Human Anatomy", "QP": "Physiology",
    "QR": "Microbiology",
    # R — Medicine
    "RA": "Public Health", "RC": "Internal Medicine", "RD": "Surgery",
    "RG": "Gynecology & Obstetrics", "RJ": "Pediatrics", "RM": "Therapeutics",
    "RS": "Pharmacy", "RT": "Nursing",
    # S — Agriculture
    "SB": "Plant Culture", "SD": "Forestry", "SF": "Animal Culture",
    "SH": "Fisheries & Aquaculture", "SK": "Hunting Sports",
    # T — Technology
    "TA": "Civil Engineering", "TC": "Hydraulic Engineering",
    "TD": "Environmental Engineering", "TE": "Highway Engineering",
    "TF": "Railroad Engineering", "TG": "Bridge Engineering",
    "TH": "Building Construction", "TJ": "Mechanical Engineering",
    "TK": "Electrical Engineering", "TL": "Motor Vehicles & Aeronautics",
    "TN": "Mining & Metallurgy", "TP": "Chemical Technology",
    "TR": "Photography", "TS": "Manufactures", "TT": "Handicrafts & Crafts",
    "TX": "Home Economics & Cooking",
    # U/V — Military & Naval
    "UA": "Armies", "UB": "Military Administration", "UD": "Infantry",
    "UG": "Military Engineering & Air Forces", "VA": "Navies", "VK": "Navigation",
    # Z — Bibliography
    "ZA": "Information Resources",
}
# убираем технический placeholder
LC_SUBCLASS.pop("DA_", None)

_CODE_RE = re.compile(r"^([A-Z]{1,3})")


def locc_class_name(code: str) -> str | None:
    """Имя LC-класса по коду (`PS3537` → 'American Literature', `E201` → 'History of the Americas').

    Возвращает самый специфичный известный ярлык: сначала 2-буквенный подкласс,
    иначе 1-буквенный класс, иначе None.
    """
    if not code:
        return None
    m = _CODE_RE.match(code.strip().upper())
    if not m:
        return None
    letters = m.group(1)
    # пробуем 2-буквенный подкласс (самый информативный), затем 1-буквенный класс
    if len(letters) >= 2 and letters[:2] in LC_SUBCLASS:
        return LC_SUBCLASS[letters[:2]]
    if letters[:1] in LC_CLASS:
        return LC_CLASS[letters[:1]]
    return None


def summarize_locc(codes: list[str], top: int = 2) -> list[str]:
    """Топ-N человекочитаемых LC-классов из списка кодов (по частоте, стабильно)."""
    from collections import Counter

    names = [locc_class_name(c) for c in codes]
    counter: Counter[str] = Counter(n for n in names if n)
    return [name for name, _ in counter.most_common(top)]
