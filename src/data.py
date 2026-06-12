"""Загрузка данных Volunteer's Dilemma.

Формат входных файлов (8 шт.):
  volunteer_data_llama_reasoning_<N>_res.json:
    {
      "prompt": "...",
      "answers": [
        {"answer": "...", "type": "Yes" | "No" | "?"},
        ...
      ]
    }

Каждый ответ становится одним ReasoningSample. Неразмеченные ответы
(тип не "Yes"/"No"/"?") пропускаются.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

log = logging.getLogger(__name__)


@dataclass
class ReasoningSample:
    prompt: str
    answer: str
    label: str        # "Yes" | "No" | "?"
    file_idx: int
    answer_idx: int


def load_samples(results_dir: str | Path) -> List[ReasoningSample]:
    """Грузит все volunteer_data_llama_reasoning_<N>_res.json из директории.

    Returns
    -------
    list[ReasoningSample]
        ~640 примеров с дисбалансом ~547/61/32 Yes/?/No.
    """
    results_dir = Path(results_dir)
    # [0-9]* (а не [0-9]) — чтобы матчились и многозначные индексы: исходные
    # файлы 1..8 однозначны, но новый корпус (generate_answers, start-idx 9)
    # доходит до 9..18+, и [0-9] поймал бы только однозначные.
    files = sorted(results_dir.glob("volunteer_data_llama_reasoning_[0-9]*_res.json"))
    if not files:
        raise FileNotFoundError(
            f"JSON-файлов volunteer_data_llama_reasoning_*_res.json не найдено в {results_dir}"
        )

    log.info("Найдено %d JSON-файлов в %s", len(files), results_dir)

    samples: list[ReasoningSample] = []
    for path in files:
        m = re.search(r"_(\d+)_res\.json$", path.name)
        file_idx = int(m.group(1)) if m else -1
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        n_before = len(samples)
        for j, item in enumerate(data["answers"]):
            label = item.get("type") or ""
            if label not in ("Yes", "No", "?"):
                continue  # пропускаем неразмеченные
            samples.append(ReasoningSample(
                prompt=data["prompt"],
                answer=item["answer"],
                label=label,
                file_idx=file_idx,
                answer_idx=j,
            ))
        log.debug("  %s → %d ответов", path.name, len(samples) - n_before)

    log.info("Загружено reasoning-ответов: %d", len(samples))
    log.info("Распределение меток: %s", Counter(s.label for s in samples))
    return samples


def save_samples(samples: List[ReasoningSample], path: str | Path) -> None:
    """Сериализует список ReasoningSample в JSON.

    Тот же формат, что использовался в ноутбуке: список словарей с
    file_idx, answer_idx, label, prompt, answer.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in samples], f, ensure_ascii=False, indent=2)
    log.info("Сохранено %d samples → %s", len(samples), path)


def load_saved_samples(path: str | Path) -> List[ReasoningSample]:
    """Читает JSON, сохранённый через save_samples, обратно в список ReasoningSample."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    samples = [ReasoningSample(**d) for d in data]
    log.info("Загружено %d samples из %s", len(samples), path)
    return samples
