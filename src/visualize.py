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


def plot_ev_per_file(
    lofo: Dict[str, dict],
    out_dir: str | Path,
    name: str = "ev_per_file",
    random_split: dict | None = None,
) -> Path:
    """Bar chart train EV vs test EV по каждому held-out файлу (эксперимент 06).

    Args
    ----
    lofo : dict
        {"1": {"train_ev": .., "test_ev": ..}, ...} — leave-one-file-out.
    random_split : dict | None
        {"train_ev": .., "test_ev": ..} — необязательный baseline (90/10),
        рисуется отдельной парой столбцов справа.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _ensure_dir(out_dir)
    keys = sorted(lofo.keys(), key=lambda s: int(s))
    labels = [f"file {k}" for k in keys]
    train_ev = [lofo[k]["train_ev"] for k in keys]
    test_ev = [lofo[k]["test_ev"] for k in keys]
    if random_split is not None:
        labels.append("random\n90/10")
        train_ev.append(random_split["train_ev"])
        test_ev.append(random_split["test_ev"])

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(labels)), 4.2))
    ax.bar(x - w / 2, train_ev, w, label="train EV", color="#3a6ea5")
    ax.bar(x + w / 2, test_ev, w, label="test EV", color="#d1495b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("explained variance")
    ax.set_title("Held-out EV (leave-one-file-out): обобщение SAE")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for xi, (tr, te) in enumerate(zip(train_ev, test_ev)):
        ax.annotate(f"{te - tr:+.3f}", (xi, max(tr, te)),
                    ha="center", va="bottom", fontsize=7, color="#555")
    plt.tight_layout()
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log.info("Сохранено: %s", path)
    return path


def plot_steering_curves(
    curves: Dict[str, dict],
    out_dir: str | Path,
    name: str = "steering_curves",
) -> Path:
    """Линии p_yes vs alpha для каждого нейрона (эксперимент 07).

    Args
    ----
    curves : dict
        {neuron_label: {"alphas": [..], "p_yes": [..], "is_control": bool}}.
        Контрольный нейрон рисуется пунктиром.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, c in curves.items():
        order = np.argsort(c["alphas"])
        a = np.asarray(c["alphas"])[order]
        p = np.asarray(c["p_yes"])[order]
        style = "--" if c.get("is_control") else "-"
        ax.plot(a, p, style, marker="o", label=label)
    ax.axvline(0.0, color="#999", lw=0.8)
    ax.set_xlabel("alpha (сила вмешательства вдоль W_dec нейрона)")
    ax.set_ylabel("p(Yes)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Steering: p(Yes) vs alpha")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / f"{name}.png"
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
