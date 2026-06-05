#!/usr/bin/env python3
"""06_eval_ev.py — held-out explained variance (leave-one-file-out + random split).

Цель: понять, обобщается ли SAE или overfit'ит. Для каждого отложенного файла
переобучаем SAE (гиперпараметры из cfg.sae) на 7 остальных файлах и меряем
explained variance (EV) на train и на отложенном файле. Дополнительно —
один random-split 90/10 ПО reasoning-ответам (baseline без сдвига домена).

EV (как в ТЗ):  EV = 1 − MSE / Var(X),
    MSE    = ((X − X̂) ** 2).mean(),  Var(X) = X.var(dim=0).mean().

Артефакты:
  artifacts/eval_ev.json           — train/test EV+MSE по фолдам и random-split
  artifacts/figures/ev_per_file.png

Время: ~8 переобучений SAE × ~3 мин на GPU ≈ 25 мин. GPU практически обязательна.

Запуск:
    python scripts/06_eval_ev.py                 # полный прогон (8 фолдов + random)
    python scripts/06_eval_ev.py --smoke         # дёшево: 2 фолда, 2 эпохи, без random
    python scripts/06_eval_ev.py --force         # пересчитать, игнорируя кеш
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from _common import parse_args_and_init


log = logging.getLogger(__name__)


def _add_args(parser):
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke-тест: 2 фолда, мало эпох, маленький expansion, без random-split.",
    )
    return parser


def main() -> None:
    cfg, args = parse_args_and_init(
        "06_eval_ev",
        description="Held-out EV (leave-one-file-out) для проверки обобщения SAE.",
        extra_args=_add_args,
    )

    from src.analysis import recon_ev
    from src.data import load_saved_samples
    from src.io_utils import load_activations
    from src.sae import sae_kwargs_for_mode, train_sae
    from src.visualize import plot_ev_per_file

    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    act_path = artifacts_dir / "activations.npz"
    samples_path = artifacts_dir / "samples.json"
    out_path = artifacts_dir / ("eval_ev_smoke.json" if args.smoke else "eval_ev.json")

    if out_path.exists() and not args.force:
        log.info("Результат уже есть: %s — пропускаю (--force чтобы пересчитать).", out_path)
        return
    if not act_path.exists():
        raise FileNotFoundError(f"Нет активаций: {act_path}. Сначала 01_extract.py.")
    if not samples_path.exists():
        raise FileNotFoundError(f"Нет samples.json: {samples_path}. Сначала 01_extract.py.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Устройство: %s", device)
    if device == "cpu":
        log.warning(
            "GPU недоступна — 06_eval_ev на CPU крайне медленный. "
            "Рекомендуется запускать на сервере с GPU (или только --smoke)."
        )

    # ---- Данные и привязка позиций к файлам
    pack = load_activations(act_path)
    samples = load_saved_samples(samples_path)
    hidden = pack["hidden"]
    sample_idx = pack["sample_idx"]
    file_of_sample = np.array([s.file_idx for s in samples], dtype=np.int64)
    file_of_pos = file_of_sample[sample_idx]

    # ---- Гиперпараметры SAE (наследуются из cfg.sae) + smoke/eval_ev-override
    sae_cfg = cfg["sae"]
    ev_cfg = cfg["eval_ev"]
    kwargs = sae_kwargs_for_mode(sae_cfg)
    expansion = sae_cfg["expansion"]
    epochs = ev_cfg["epochs"] if ev_cfg["epochs"] is not None else sae_cfg["epochs"]
    enc_bs = ev_cfg["encode_batch_size"]

    held_files = ev_cfg["held_files"]
    if held_files is None:
        held_files = sorted(int(f) for f in np.unique(file_of_sample))
    run_random = bool(ev_cfg["random_split"])

    if args.smoke:
        held_files = held_files[:2]
        epochs = min(2, epochs)
        expansion = 4
        run_random = False
        log.info("SMOKE: фолды=%s, epochs=%d, expansion=%d, random_split=off",
                 held_files, epochs, expansion)

    def _train_eval(train_mask: np.ndarray, test_mask: np.ndarray, seed: int) -> dict:
        sae, _hist = train_sae(
            hidden=hidden[train_mask],
            expansion=expansion,
            lr=sae_cfg["lr"],
            epochs=epochs,
            batch_size=sae_cfg["batch_size"],
            device=device,
            seed=seed,
            verbose=False,
            **kwargs,
        )
        tr = recon_ev(sae, hidden[train_mask], batch_size=enc_bs, device=device)
        te = recon_ev(sae, hidden[test_mask], batch_size=enc_bs, device=device)
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "train_ev": tr["ev"], "test_ev": te["ev"],
            "train_mse": tr["mse"], "test_mse": te["mse"],
            "n_train": tr["n"], "n_test": te["n"],
        }

    results: dict = {"meta": {
        "mode": sae_cfg["mode"], "expansion": expansion, "epochs": epochs,
        "sae_kwargs": kwargs, "smoke": bool(args.smoke),
    }, "leave_one_file_out": {}}

    # ---- Leave-one-file-out
    log.info("=== Leave-one-file-out: %d фолдов ===", len(held_files))
    for hf in held_files:
        test_mask = file_of_pos == hf
        train_mask = ~test_mask
        log.info("Фолд held_file=%d: train=%d поз., test=%d поз. — обучаю SAE ...",
                 hf, int(train_mask.sum()), int(test_mask.sum()))
        res = _train_eval(train_mask, test_mask, seed=sae_cfg["seed"])
        results["leave_one_file_out"][str(hf)] = res
        log.info(
            "file %d: train EV=%.4f, test EV=%.4f, gap=%.4f (test MSE=%.2f)",
            hf, res["train_ev"], res["test_ev"],
            res["test_ev"] - res["train_ev"], res["test_mse"],
        )

    # ---- Random split 90/10 по reasoning-ответам
    if run_random:
        log.info("=== Random split 90/10 (по reasoning-ответам) ===")
        rng = np.random.default_rng(ev_cfg["seed"])
        n_samples = len(samples)
        perm = rng.permutation(n_samples)
        n_test = max(1, int(round(n_samples * ev_cfg["random_split_frac"])))
        test_samples = set(int(i) for i in perm[:n_test])
        test_mask = np.array([int(si) in test_samples for si in sample_idx])
        train_mask = ~test_mask
        log.info("Random: train=%d поз. (%d отв.), test=%d поз. (%d отв.)",
                 int(train_mask.sum()), n_samples - n_test,
                 int(test_mask.sum()), n_test)
        res = _train_eval(train_mask, test_mask, seed=sae_cfg["seed"])
        results["random_split"] = res
        log.info("random: train EV=%.4f, test EV=%.4f, gap=%.4f",
                 res["train_ev"], res["test_ev"], res["test_ev"] - res["train_ev"])

    # ---- Сохранение + график
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info("EV-результаты сохранены: %s", out_path)

    fig_name = "ev_per_file_smoke" if args.smoke else "ev_per_file"
    plot_ev_per_file(
        results["leave_one_file_out"], figures_dir,
        name=fig_name, random_split=results.get("random_split"),
    )

    # ---- Итоговая сводка в лог
    lofo = results["leave_one_file_out"]
    mean_gap = float(np.mean([v["test_ev"] - v["train_ev"] for v in lofo.values()]))
    mean_test = float(np.mean([v["test_ev"] for v in lofo.values()]))
    log.info("ИТОГ: средний test EV=%.4f, средний gap (test−train)=%.4f", mean_test, mean_gap)


if __name__ == "__main__":
    main()
