"""Визуализация результатов пайплайна. Все функции сохраняют PNG в
указанный каталог — никакого matplotlib.show(), чтобы работало без
дисплея на сервере.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

log = logging.getLogger(__name__)


def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_train_curves(history: Dict[str, list], sae_name: str, out_dir: str | Path) -> Path:
    """Кривые recon / L0 / L1 по эпохам для одной SAE.

    Возвращает путь к PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _ensure_dir(out_dir)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    axes[0].plot(history["recon"], label=sae_name)
    axes[1].plot(history["l0"],    label=sae_name)
    axes[2].plot(history["l1"],    label=sae_name)

    titles = ["Reconstruction MSE", "L0 (active neurons)", "L1 (sum activations)"]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    plt.tight_layout()
    path = out_dir / f"train_curves_{sae_name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log.info("Сохранено: %s", path)
    return path


def plot_pareto(results: Dict[str, dict], out_dir: str | Path) -> Path:
    """Pareto-плот: финальная reconstruction vs финальный L0 для всех SAE.

    Args
    ----
    results : dict
        name → {"history": {"recon": [...], "l0": [...], ...}, ...}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, res in results.items():
        h = res["history"]
        ax.scatter(h["l0"][-1], h["recon"][-1], s=120, label=name)
        ax.annotate(
            name, (h["l0"][-1], h["recon"][-1]),
            xytext=(8, 5), textcoords="offset points",
        )
    ax.set_xlabel("L0 (active neurons)")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Pareto: разреженность ↔ реконструкция")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / "pareto.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log.info("Сохранено: %s", path)
    return path


def plot_fire_rate_hist(stats: dict, sae_name: str, out_dir: str | Path) -> Path:
    """Гистограмма fire_rate среди ЖИВЫХ нейронов (fire_rate > 0).

    Помогает увидеть: есть ли длинный хвост редких + горстка частых,
    или распределение бимодальное.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _ensure_dir(out_dir)
    fr = stats["fire_rate"]
    alive = fr[fr > 0]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(alive, bins=80, log=True, color="#3a6ea5", edgecolor="white")
    ax.set_xlabel("fire_rate (доля позиций, где нейрон активен)")
    ax.set_ylabel("число нейронов (log-scale)")
    ax.set_title(
        f"Распределение fire_rate, SAE={sae_name}  "
        f"(живых: {len(alive)}, мёртвых: {(fr == 0).sum()})"
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / f"fire_rate_hist_{sae_name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log.info("Сохранено: %s", path)
    return path


def plot_top_neurons_heatmap(
    features: np.ndarray,
    neuron_ids: Sequence[int],
    labels: np.ndarray,
    out_dir: str | Path,
    name: str = "top_neurons_heatmap",
    title: str | None = None,
) -> Path:
    """Хитмапа средних активаций топ-нейронов по группам меток.

    Строки — нейроны, колонки — метки (Yes / No / ?), значения — средние
    активации. Полезно увидеть, какие нейроны явно тяготеют к Yes vs No.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _ensure_dir(out_dir)
    f = features.astype(np.float32)
    unique_labels = sorted(set(labels.tolist()))
    means = np.zeros((len(neuron_ids), len(unique_labels)))
    for i, nid in enumerate(neuron_ids):
        for j, lab in enumerate(unique_labels):
            mask = labels == lab
            if mask.any():
                means[i, j] = f[mask, nid].mean()

    fig_h = max(3, 0.3 * len(neuron_ids) + 1)
    fig, ax = plt.subplots(figsize=(4.5, fig_h))
    im = ax.imshow(means, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(unique_labels)))
    ax.set_xticklabels(unique_labels)
    ax.set_yticks(range(len(neuron_ids)))
    ax.set_yticklabels([str(n) for n in neuron_ids])
    ax.set_xlabel("label")
    ax.set_ylabel("neuron id")
    ax.set_title(title or "Mean activation by label (top neurons)")
    fig.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log.info("Сохранено: %s", path)
    return path
