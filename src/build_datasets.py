"""Два среза из ОДНОЙ генерации: full и balanced.

Из файлов, которые написал src/generate_answers.py (формат src/data.py +
служебное поле scenario_meta), делаем две папки, обе в том же формате — их
читает существующий data.py без изменений:

  full      — все ответы как есть. На нём обучается SAE.
              (Меток для SAE не нужно; больше данных = лучше реконструкция;
              дисбаланс SAE не «переобучает» — это не классификатор.)

  balanced  — андерсэмплинг по корпусу до равного числа Yes/No/? (по самому
              редкому классу, случайный выбор с фиксированным seed). На нём
              считаем Yes/No-контраст и отбираем триггеры — там МЕТКИ
              используются и шум редкого класса вреден.

scenario_meta переносится в оба среза без изменений (нужно для отбора
промптов по податливости в steering — Задача 3).

Запуск:
    python -m src.build_datasets \
        --in-dir data/results_dilemma_v2 \
        --full-dir data/results_dilemma_v2_full \
        --balanced-dir data/results_dilemma_v2_balanced \
        --seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_datasets")

VALID = ("Yes", "No", "?")
FILE_GLOB = "volunteer_data_llama_reasoning_*_res.json"


def _load_files(in_dir: Path) -> List[Tuple[Path, dict]]:
    """Грузит все файлы генерации, отсортированные по числовому индексу N."""
    files = list(in_dir.glob(FILE_GLOB))
    if not files:
        raise FileNotFoundError(f"Нет файлов {FILE_GLOB} в {in_dir}")

    def _idx(p: Path) -> int:
        m = re.search(r"_(\d+)_res\.json$", p.name)
        return int(m.group(1)) if m else -1

    files.sort(key=_idx)
    out = []
    for p in files:
        with open(p, encoding="utf-8") as f:
            out.append((p, json.load(f)))
    log.info("Загружено файлов: %d из %s", len(out), in_dir)
    return out


def _write_payload(out_dir: Path, name: str, prompt: str, meta, answers: List[dict]) -> None:
    payload = {"prompt": prompt}
    if meta is not None:
        payload["scenario_meta"] = meta
    payload["answers"] = answers
    with open(out_dir / name, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _dist(loaded: List[Tuple[Path, dict]]) -> Counter:
    c: Counter = Counter()
    for _, data in loaded:
        for a in data["answers"]:
            t = a.get("type")
            if t in VALID:
                c[t] += 1
    return c


def build_full(loaded: List[Tuple[Path, dict]], full_dir: Path) -> Counter:
    """Срез full: все ответы как есть (verbatim re-dump в отдельную папку)."""
    full_dir.mkdir(parents=True, exist_ok=True)
    for path, data in loaded:
        _write_payload(full_dir, path.name, data["prompt"],
                       data.get("scenario_meta"), list(data["answers"]))
    dist = _dist(loaded)
    log.info("FULL → %s | размеченных ответов: %d | %s",
             full_dir, sum(dist.values()), dict(dist))
    return dist


def build_balanced(
    loaded: List[Tuple[Path, dict]], balanced_dir: Path, seed: int = 0,
) -> Counter:
    """Срез balanced: глобальный андерсэмплинг до равного числа Yes/No/?.

    Балансируем по ВСЕМУ корпусу (а не пофайлово): берём min по классам и для
    каждого класса случайно оставляем ровно столько (fixed seed). Структура
    файлов/промптов/scenario_meta сохраняется — у каждого промпта остаётся его
    подмножество ответов (в исходном порядке).
    """
    balanced_dir.mkdir(parents=True, exist_ok=True)

    # records: (file_index_in_loaded, position_in_answers, label)
    by_label: Dict[str, List[Tuple[int, int]]] = {k: [] for k in VALID}
    for fi, (_, data) in enumerate(loaded):
        for pos, a in enumerate(data["answers"]):
            t = a.get("type")
            if t in VALID:
                by_label[t].append((fi, pos))

    counts = {k: len(v) for k, v in by_label.items()}
    keep_per_class = min(counts.values()) if counts else 0
    log.info("BALANCED: до баланса %s → оставляем по %d на класс",
             counts, keep_per_class)
    if keep_per_class == 0:
        log.warning("BALANCED: один из классов пуст (%s) — срез будет пустым. "
                    "Сгенерируйте больше редкого класса.", counts)

    rng = random.Random(seed)
    kept: set[Tuple[int, int]] = set()
    for label in VALID:
        pool = by_label[label]
        chosen = pool if len(pool) <= keep_per_class else rng.sample(pool, keep_per_class)
        kept.update(chosen)

    for fi, (path, data) in enumerate(loaded):
        answers = [a for pos, a in enumerate(data["answers"]) if (fi, pos) in kept]
        _write_payload(balanced_dir, path.name, data["prompt"],
                       data.get("scenario_meta"), answers)

    # Распределение по реально записанному (точное, на случай малых классов).
    real = _dist([(p, {"answers": [a for pos, a in enumerate(d["answers"])
                                    if (fi, pos) in kept]})
                  for fi, (p, d) in enumerate(loaded)])
    log.info("BALANCED → %s | ответов: %d | %s",
             balanced_dir, sum(real.values()), dict(real))
    return real


# ---------------------------------------------------------------------------
# Balanced-срез БЕЗ повторной экстракции: маска по позициям full-pack.
# Нужна на шаге анализа (Yes/No-контраст считаем на balanced, а активации
# снимаются один раз на FULL — см. README и scripts/03_analyze.py).
# ---------------------------------------------------------------------------

def _balanced_keys_from_dir(balanced_dir: Path) -> set:
    """{(file_idx, answer_text)} всех ответов в готовом balanced-срезе."""
    keys: set = set()
    for path, data in _load_files(balanced_dir):
        m = re.search(r"_(\d+)_res\.json$", path.name)
        fidx = int(m.group(1)) if m else -1
        for a in data["answers"]:
            keys.add((fidx, a.get("answer", "")))
    return keys


def balanced_sample_mask(
    samples: Sequence,
    sample_idx: np.ndarray,
    *,
    balanced_dir: "str | Path | None" = None,
    seed: int = 0,
    classes: Tuple[str, ...] = ("Yes", "No", "?"),
) -> Tuple[np.ndarray, dict]:
    """Булева маска по ПОЗИЦИЯМ full-pack, оставляющая class-balanced подвыборку ОТВЕТОВ.

    Балансировка на уровне ОТВЕТА (один reasoning = одна единица): берём по
    самому редкому классу и разворачиваем выбор обратно в позиции (все позиции
    выбранных ответов). Это даёт balanced-срез для контраста без повторного
    снятия активаций — тот же full-pack, только маска.

    Два источника выбора:
      * balanced_dir задан и существует → используем РОВНО те ответы, что лежат
        в готовом срезе (src/build_datasets), матчинг по (file_idx, answer).
        Гарантирует совпадение с артефактом-папкой.
      * иначе → детерминированный андерсэмплинг по самому редкому классу (тот же
        алгоритм, что build_balanced; fixed seed). Self-contained fallback.

    Returns
    -------
    (mask, info) : (np.ndarray[bool] длины N_positions,
                    {"counts_before": {...}, "kept_per_class": int|None,
                     "source": "dir"|"derive", "n_answers_kept": int})
    """
    sample_idx = np.asarray(sample_idx)

    ids_by_label: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        if s.label in classes:
            ids_by_label[s.label].append(i)
    counts_before = {c: len(ids_by_label.get(c, [])) for c in classes}

    kept_ids: set = set()
    info = {"counts_before": counts_before}

    bd = Path(balanced_dir) if balanced_dir is not None else None
    if bd is not None and bd.exists() and any(bd.glob(FILE_GLOB)):
        keys = _balanced_keys_from_dir(bd)
        for i, s in enumerate(samples):
            if s.label in classes and (s.file_idx, s.answer) in keys:
                kept_ids.add(i)
        info.update(source="dir", kept_per_class=None)
        log.info("balanced_sample_mask: источник=папка %s, ответов в срезе=%d",
                 bd, len(kept_ids))
    else:
        keep = min(counts_before.values()) if counts_before else 0
        rng = np.random.default_rng(seed)
        for c in classes:
            pool = ids_by_label.get(c, [])
            chosen = pool if len(pool) <= keep else rng.choice(pool, size=keep, replace=False)
            kept_ids.update(int(x) for x in chosen)
        info.update(source="derive", kept_per_class=int(keep))
        log.info("balanced_sample_mask: источник=derive(seed=%d), до=%s → по %d/класс",
                 seed, counts_before, keep)

    mask = np.fromiter((int(si) in kept_ids for si in sample_idx),
                       dtype=bool, count=len(sample_idx))
    info["n_answers_kept"] = len(kept_ids)
    return mask, info


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", default="data/results_dilemma_v2",
                   help="папка с файлами генерации (вход)")
    p.add_argument("--full-dir", default="data/results_dilemma_v2_full",
                   help="выход: full-срез (для обучения SAE)")
    p.add_argument("--balanced-dir", default="data/results_dilemma_v2_balanced",
                   help="выход: balanced-срез (для Yes/No-контраста)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed андерсэмплинга balanced (воспроизводимость)")
    args = p.parse_args()

    loaded = _load_files(Path(args.in_dir))
    build_full(loaded, Path(args.full_dir))
    build_balanced(loaded, Path(args.balanced_dir), seed=args.seed)
    log.info("ГОТОВО. SAE учим на --full-dir, контраст/триггеры — на --balanced-dir.")


if __name__ == "__main__":
    main()
