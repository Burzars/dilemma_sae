#!/usr/bin/env python3
"""Анализ feature-векторов SAE: статистика, top контексты, Yes/No контраст.

На выходе:
  artifacts/features_<mode>.npz       — матрица features (encode_all)
  artifacts/neuron_stats_<mode>.npz   — fire_rate / mean_active / max_active / n_fires
  artifacts/neuron_report_<mode>.json — отчёт (top контексты для top нейронов
                                        + yes/no триггеры из контраста),
                                        который потом скармливаем LLM-судье
                                        в 04_interpret.py.

Запуск:
    python scripts/03_analyze.py
    python scripts/03_analyze.py --override sae.mode=jumprelu
    python scripts/03_analyze.py --override contrast.neg_label="?"
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from _common import parse_args_and_init


log = logging.getLogger(__name__)


def main() -> None:
    cfg, args = parse_args_and_init(
        "03_analyze",
        description="Анализ SAE-фич: статистика, top контексты, Yes/No контраст.",
    )

    from src.analysis import (
        balanced_contrast_arrays,
        build_report,
        encode_all,
        find_contrast_neurons,
        label_contrast,
        neuron_statistics_streaming,
        sample_level_contrast,
        select_top_neurons,
    )
    from src.data import load_saved_samples
    from src.io_utils import load_activations, load_sae

    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    mode = cfg["sae"]["mode"]
    sae_path = artifacts_dir / f"sae_{mode}.pt"
    feat_path = artifacts_dir / f"features_{mode}.npz"
    stats_path = artifacts_dir / f"neuron_stats_{mode}.npz"
    report_path = artifacts_dir / f"neuron_report_{mode}.json"

    if not sae_path.exists():
        raise FileNotFoundError(
            f"SAE не найден: {sae_path}. Сначала запустите 02_train_sae.py."
        )

    # ---- 1. Активации, SAE, samples
    pack = load_activations(artifacts_dir / "activations.npz")
    samples = load_saved_samples(artifacts_dir / "samples.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae, _ckpt = load_sae(sae_path, device=device)

    # ---- 2. Статистика нейронов (streaming, БЕЗ плотной матрицы N×d_hidden) ----
    # Плотная (228k × 57k) float16 ≈ 26 ГБ — на больших корпусах не держим её в
    # ОЗУ. Статистики считаем потоково по батчам на GPU (только (d_hidden,)-векторы).
    if stats_path.exists() and not args.force:
        log.info("Загружаем кеш статистики нейронов: %s", stats_path)
        npz = np.load(stats_path, allow_pickle=False)
        stats = {k: npz[k] for k in npz.files}
    else:
        log.info("Считаю статистику нейронов потоково (без матрицы N×d_hidden) ...")
        stats = neuron_statistics_streaming(
            sae, pack["hidden"],
            batch_size=cfg["analysis"]["encode_batch_size"], device=device,
        )
        np.savez_compressed(stats_path, **stats)
        log.info("Статистика сохранена: %s", stats_path)

    dead = int((stats["fire_rate"] == 0).sum())
    alive = int((stats["fire_rate"] > 0).sum())
    log.info("Мёртвых нейронов: %d / %d", dead, len(stats["fire_rate"]))
    log.info("Живых нейронов: %d", alive)
    if alive:
        fr_alive = stats["fire_rate"][stats["fire_rate"] > 0]
        log.info(
            "fire_rate среди живых: min=%.4f, median=%.4f, mean=%.4f, max=%.4f",
            fr_alive.min(), float(np.median(fr_alive)),
            fr_alive.mean(), fr_alive.max(),
        )

    # ---- 3. Top "интересных" нейронов (из статистики FULL) ----
    top_neurons = select_top_neurons(
        stats,
        k=cfg["analysis"]["top_neurons_k"],
        min_fires=cfg["analysis"]["min_fires"],
        max_fire_rate=cfg["analysis"]["max_fire_rate"],
    )
    log.info("Выбрано top-%d интересных нейронов.", len(top_neurons))

    # ---- 4. Кодируем ТОЛЬКО balanced-срез активаций → features ----
    # SAE/статистика — на FULL; контраст и top-контексты — на balanced-срезе.
    # Кодируем лишь срез (~равные Yes/No/?), а не весь корпус → экономим память.
    log.info(
        "Yes/No контраст: pos=%s, neg=%s, slice=%s",
        cfg["contrast"]["pos_label"], cfg["contrast"]["neg_label"],
        cfg["contrast"].get("slice", "balanced"),
    )
    c_hidden, c_labels, c_sample_idx, c_token_pos = balanced_contrast_arrays(
        pack["hidden"], pack["label"], pack["sample_idx"], pack["token_pos"], samples,
        slice=cfg["contrast"].get("slice", "balanced"),
        balanced_dir=cfg["contrast"].get("balanced_dir"),
        balanced_seed=cfg["contrast"].get("balanced_seed", 0),
    )
    log.info("Кодирую balanced-срез: %d позиций → features ...", c_hidden.shape[0])
    c_features = encode_all(
        sae, c_hidden,
        batch_size=cfg["analysis"]["encode_batch_size"], device=device,
    )

    # Sanity check для TopK (на срезе — доля нулей одинакова для любого набора позиций)
    if sae.mode == "topk":
        expected = 1 - sae.k / sae.d_hidden
        actual = float((c_features == 0).mean())
        log.info("TopK sanity: доля нулей = %.4f, ожидалось %.4f", actual, expected)
        if abs(actual - expected) > 0.001:
            log.warning("Доля нулей не совпадает с ожидаемой для TopK! "
                        "actual=%.4f, expected=%.4f", actual, expected)

    # ---- 5. Yes/No контраст (token + sample) на срезе ----
    tok_contrast = label_contrast(
        c_features, c_labels,
        pos_label=cfg["contrast"]["pos_label"],
        neg_label=cfg["contrast"]["neg_label"],
        min_fires=cfg["analysis"]["min_fires"],
    )
    smp_contrast = sample_level_contrast(
        c_features, c_sample_idx, samples,
        pos_label=cfg["contrast"]["pos_label"],
        neg_label=cfg["contrast"]["neg_label"],
    )
    log.info(
        "Sample-level: n_pos=%d, n_neg=%d",
        smp_contrast["n_pos"], smp_contrast["n_neg"],
    )

    yes_neurons, no_neurons = find_contrast_neurons(
        tok_contrast, smp_contrast,
        top_k=cfg["contrast"]["top_k"],
        pool_factor=cfg["contrast"]["pool_factor"],
    )

    # ---- 6. Сборка отчёта для LLM-судьи (top-контексты из balanced-среза) ----
    log.info("Собираем отчёт по нейронам ...")
    # Для top контекстов нужен токенайзер — грузим только его (без модели)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    report = (
        build_report(
            yes_neurons, "yes_trigger",
            c_features, c_sample_idx, c_token_pos, samples, tokenizer,
            stats, tok_contrast, smp_contrast,
            top_k=cfg["analysis"]["top_contexts_k"],
            window=cfg["analysis"]["context_window"] + 10,  # для отчёта чуть шире окно
        )
        + build_report(
            no_neurons, "no_trigger",
            c_features, c_sample_idx, c_token_pos, samples, tokenizer,
            stats, tok_contrast, smp_contrast,
            top_k=cfg["analysis"]["top_contexts_k"],
            window=cfg["analysis"]["context_window"] + 10,
        )
        # Дополнительно — top по mean_active * log(1+n_fires) (общие "интересные")
        + build_report(
            list(top_neurons), "top_active",
            c_features, c_sample_idx, c_token_pos, samples, tokenizer,
            stats, tok_contrast, smp_contrast,
            top_k=cfg["analysis"]["top_contexts_k"],
            window=cfg["analysis"]["context_window"] + 10,
        )
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log.info("Отчёт сохранён: %s (нейронов: %d)", report_path, len(report))


if __name__ == "__main__":
    main()
