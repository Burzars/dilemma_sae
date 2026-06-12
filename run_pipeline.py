#!/usr/bin/env python3
"""run_pipeline.py — единая точка входа для Задачи 2 (после локальной экстракции).

Предполагается, что активации УЖЕ сняты локально на FULL-срезе:

    # 0) (один раз) сгенерировать корпус и срезы:
    python -m src.generate_answers --out-dir data/results_dilemma_v2
    python -m src.build_datasets                       # full + balanced папки

    # 1) снять активации на FULL (тем же Qwen2.5-7B-Instruct):
    python scripts/01_extract.py                       # data.results_dir=...v2_full

    # 2) этот скрипт прогоняет остальное:
    python run_pipeline.py                             # 02 → 06 → 03 → 04
    python run_pipeline.py --smoke                     # дёшево, без судьи/EV-random
    python run_pipeline.py --skip 06_eval_ev           # пропустить шаг

Что делает (оркестрация существующих scripts/NN_*.py, чтобы не дублировать код):
  02_train_sae   — обучает SAE (topk, k=64, expansion=16 из cfg.sae) на FULL.
  06_eval_ev     — EV: random split 90/10 + leave-one-file-out → eval_ev.json,
                   ev_per_file.png (обобщает ли SAE).
  03_analyze     — features/neuron_statistics на FULL; Yes/No-контраст
                   (label_contrast + sample_level_contrast + find_contrast_neurons)
                   на BALANCED (contrast.slice=balanced); отчёт по yes/no/top.
  04_interpret   — LLM-судья → темы нейронов (нужен OPENROUTER_API_KEY).

Срезы (см. README): SAE и features учим на FULL; контраст и отбор триггеров —
на BALANCED. Это управляется конфигом (data.results_dir=full, contrast.slice).

--override прокидывается во ВСЕ шаги. --smoke прокидывается в 06/07-совместимые
шаги (02/03/04 его не знают и получат свои дешёвые override'ы).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"

# Шаги Задачи 2 по порядку. smoke=True → пробросить --smoke (если шаг его понимает).
STEPS = [
    {"name": "02_train_sae", "smoke_flag": False,
     "smoke_override": ["sae.epochs=2", "sae.expansion=4"]},
    {"name": "06_eval_ev", "smoke_flag": True, "smoke_override": []},
    {"name": "03_analyze", "smoke_flag": False, "smoke_override": []},
    {"name": "04_interpret", "smoke_flag": False,
     "smoke_override": ["interpret.max_snippets_per_neuron=3"]},
]


def _run(name: str, common: list[str], base_overrides: list[str],
         smoke: bool, step: dict) -> None:
    cmd = [sys.executable, str(SCRIPTS / f"{name}.py"), *common]
    if smoke and step["smoke_flag"]:
        cmd.append("--smoke")
    overrides = list(base_overrides)
    if smoke:
        overrides += step["smoke_override"]
    if overrides:
        cmd += ["--override", *overrides]
    print(f"\n{'='*70}\n▶ {name}\n  {' '.join(cmd)}\n{'='*70}", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=ROOT)
    if res.returncode != 0:
        raise SystemExit(f"Шаг {name} упал (код {res.returncode}). Останов.")
    print(f"✔ {name} за {time.time() - t0:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", "-c", default="configs/default.yaml")
    p.add_argument("--override", "-o", nargs="*", default=[],
                   help="прокинуть key=value во все шаги (как у scripts/*).")
    p.add_argument("--smoke", action="store_true",
                   help="дешёвый сквозной прогон (мало эпох/фолдов/сниппетов).")
    p.add_argument("--force", "-f", action="store_true",
                   help="прокинуть --force во все шаги (игнорировать кеш).")
    p.add_argument("--skip", nargs="*", default=[],
                   help="имена шагов для пропуска, напр.: --skip 06_eval_ev 04_interpret")
    p.add_argument("--only", nargs="*", default=[],
                   help="запустить только эти шаги (имена как в STEPS).")
    args = p.parse_args()

    base_overrides = list(args.override)

    common = ["--config", args.config]
    if args.force:
        common.append("--force")

    steps = [s for s in STEPS if s["name"] not in set(args.skip)]
    if args.only:
        steps = [s for s in steps if s["name"] in set(args.only)]
    if not steps:
        raise SystemExit("Нечего запускать (проверьте --skip/--only).")

    print("Пайплайн Задачи 2:", " → ".join(s["name"] for s in steps))
    if not (ROOT / "artifacts" / "activations.npz").exists():
        print("⚠ artifacts/activations.npz не найден — сначала прогоните "
              "scripts/01_extract.py на FULL-срезе.", flush=True)

    t0 = time.time()
    for step in steps:
        _run(step["name"], common, base_overrides, args.smoke, step)
    print(f"\nГОТОВО за {time.time() - t0:.1f}s. Дальше — steering: "
          f"python scripts/07_steering.py --smoke", flush=True)


if __name__ == "__main__":
    main()
