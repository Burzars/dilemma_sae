#!/usr/bin/env python3
"""LLM-судья (OpenRouter): по top контекстам каждого нейрона → одна
короткая тема, либо "NO CLEAR THEME" для шумных нейронов.

Читает artifacts/neuron_report_<mode>.json (от 03_analyze.py),
записывает:
  artifacts/neuron_themes_<mode>.json   — то же + поле "theme"
  artifacts/summary_<mode>.csv          — компактная сводка

Требует переменную окружения OPENROUTER_API_KEY.

Запуск:
    export OPENROUTER_API_KEY=sk-...
    python scripts/04_interpret.py
    python scripts/04_interpret.py --override interpret.judge_model=anthropic/claude-3-haiku
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from _common import parse_args_and_init


log = logging.getLogger(__name__)


def main() -> None:
    cfg, args = parse_args_and_init(
        "04_interpret",
        description="LLM-судья → темы нейронов (OpenRouter).",
    )

    from src.interpret import interpret_report

    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    mode = cfg["sae"]["mode"]
    report_path = artifacts_dir / f"neuron_report_{mode}.json"
    themes_path = artifacts_dir / f"neuron_themes_{mode}.json"
    summary_path = artifacts_dir / f"summary_{mode}.csv"

    if not report_path.exists():
        raise FileNotFoundError(
            f"Не найден отчёт {report_path}. Сначала запустите 03_analyze.py."
        )

    if themes_path.exists() and not args.force:
        log.info("Темы уже посчитаны: %s — пропускаю.", themes_path)
        log.info("Используйте --force, чтобы перепросить LLM.")
        with open(themes_path, encoding="utf-8") as f:
            themed = json.load(f)
    else:
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        log.info("Запускаем LLM-судью: %d нейронов, model=%s",
                 len(report), cfg["interpret"]["judge_model"])
        themed = interpret_report(
            report,
            judge_model=cfg["interpret"]["judge_model"],
            api_base=cfg["interpret"]["api_base"],
            max_snippets=cfg["interpret"]["max_snippets_per_neuron"],
        )
        with open(themes_path, "w", encoding="utf-8") as f:
            json.dump(themed, f, ensure_ascii=False, indent=2)
        log.info("Темы сохранены: %s", themes_path)

    # Компактная CSV-сводка
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "neuron_id", "kind", "theme",
            "fire_rate", "mean_active", "n_fires",
            "cohens_d_token", "cohens_d_sample",
        ])
        for item in themed:
            s = item["stats"]
            w.writerow([
                item["neuron_id"],
                item["kind"],
                item.get("theme", ""),
                f"{s['fire_rate']:.4f}",
                f"{s['mean_active']:.4f}",
                s["n_fires"],
                f"{s['cohens_d_token']:.3f}",
                f"{s['cohens_d_sample']:.3f}",
            ])
    log.info("CSV-сводка: %s", summary_path)


if __name__ == "__main__":
    main()
