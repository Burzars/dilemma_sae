#!/usr/bin/env python3
"""Генерация PNG-графиков из артефактов.

Создаёт в artifacts/figures/:
  train_curves_<mode>.png   — кривые recon / L0 / L1 по эпохам
  pareto.png                — Pareto (recon vs L0) для всех имеющихся SAE
  fire_rate_hist_<mode>.png — распределение fire_rate живых нейронов
  top_neurons_heatmap_<mode>.png — средние активации топ-нейронов по меткам

Запуск:
    python scripts/05_visualize.py
    python scripts/05_visualize.py --override sae.mode=topk
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from _common import parse_args_and_init


log = logging.getLogger(__name__)


def main() -> None:
    cfg, args = parse_args_and_init(
        "05_visualize",
        description="Генерация PNG-графиков в artifacts/figures/.",
    )

    from src.analysis import (
        find_contrast_neurons,
        label_contrast,
        sample_level_contrast,
    )
    from src.data import load_saved_samples
    from src.io_utils import load_activations, load_sae
    from src.visualize import (
        plot_fire_rate_hist,
        plot_pareto,
        plot_top_neurons_heatmap,
        plot_train_curves,
    )

    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    mode = cfg["sae"]["mode"]

    # ---- 1. train_curves для текущего mode (из ckpt)
    sae_path = artifacts_dir / f"sae_{mode}.pt"
    if sae_path.exists():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _, ckpt = load_sae(sae_path, device=device)
        plot_train_curves(ckpt["history"], mode, figures_dir)
    else:
        log.warning("SAE %s не найден, train_curves пропущен.", sae_path)

    # ---- 2. Pareto по всем имеющимся sae_*.pt
    all_sae_paths = sorted(artifacts_dir.glob("sae_*.pt"))
    if len(all_sae_paths) >= 1:
        results = {}
        for p in all_sae_paths:
            name = p.stem.replace("sae_", "")
            try:
                _, ckpt = load_sae(p, device="cpu")
                results[name] = {"history": ckpt["history"]}
            except Exception as e:
                log.warning("Не удалось прочитать %s: %s", p, e)
        if results:
            plot_pareto(results, figures_dir)

    # ---- 3. fire_rate_hist
    stats_path = artifacts_dir / f"neuron_stats_{mode}.npz"
    if stats_path.exists():
        npz = np.load(stats_path, allow_pickle=False)
        stats = {k: npz[k] for k in npz.files}
        plot_fire_rate_hist(stats, mode, figures_dir)
    else:
        log.warning(
            "neuron_stats_%s.npz не найден — пропускаю fire_rate_hist. "
            "Сначала запустите 03_analyze.py.", mode,
        )

    # ---- 4. Heatmap top-нейронов по меткам.
    # Нужны features + контраст, чтобы выбрать топ-нейроны.
    feat_path = artifacts_dir / f"features_{mode}.npz"
    if feat_path.exists():
        features = np.load(feat_path, allow_pickle=False)["features"]
        pack = load_activations(artifacts_dir / "activations.npz")
        samples = load_saved_samples(artifacts_dir / "samples.json")

        tok_contrast = label_contrast(
            features, pack["label"],
            pos_label=cfg["contrast"]["pos_label"],
            neg_label=cfg["contrast"]["neg_label"],
            min_fires=cfg["analysis"]["min_fires"],
        )
        smp_contrast = sample_level_contrast(
            features, pack["sample_idx"], samples,
            pos_label=cfg["contrast"]["pos_label"],
            neg_label=cfg["contrast"]["neg_label"],
        )
        yes_neurons, no_neurons = find_contrast_neurons(
            tok_contrast, smp_contrast,
            top_k=cfg["contrast"]["top_k"],
            pool_factor=cfg["contrast"]["pool_factor"],
        )
        combined = list(yes_neurons) + list(no_neurons)
        if combined:
            plot_top_neurons_heatmap(
                features, combined, pack["label"], figures_dir,
                name=f"top_neurons_heatmap_{mode}",
                title=f"Yes/No-trigger neurons mean activations ({mode})",
            )
    else:
        log.warning(
            "features_%s.npz не найден — пропускаю heatmap. "
            "Сначала запустите 03_analyze.py.", mode,
        )

    log.info("Готово. Все PNG в %s", figures_dir)


if __name__ == "__main__":
    main()
