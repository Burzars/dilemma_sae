#!/usr/bin/env python3
"""Извлечение hidden states LLM на reasoning-ответах Volunteer's Dilemma.

Если artifacts/activations.npz уже существует и не указан --force,
шаг пропускается. samples.json (метаданные текстов) пишутся всегда.

Запуск:
    python scripts/01_extract.py
    python scripts/01_extract.py --override extraction.batch_size=8
    python scripts/01_extract.py --force
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from _common import parse_args_and_init


log = logging.getLogger(__name__)


def main() -> None:
    cfg, args = parse_args_and_init(
        "01_extract",
        description="Извлечение hidden states LLM из reasoning-ответов.",
    )

    # Импорты после init, чтобы логгер уже работал.
    from src.data import load_samples, save_samples
    from src.extract import ActivationConfig, extract_activations
    from src.io_utils import save_activations

    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    act_path = artifacts_dir / "activations.npz"
    samples_path = artifacts_dir / "samples.json"

    # Кеширование — главный смысл скрипта.
    if act_path.exists() and not args.force:
        log.info("Активации уже посчитаны: %s — пропускаю.", act_path)
        log.info("Используйте --force, чтобы пересчитать заново.")
        return

    # ---- 1. Загрузка данных
    log.info("Загружаем reasoning-ответы из %s ...", cfg["data"]["results_dir"])
    samples = load_samples(cfg["data"]["results_dir"])
    # Сохраним сериализованные samples в любом случае — нужны на шагах
    # анализа и визуализации, чтобы декодировать сниппеты.
    save_samples(samples, samples_path)

    # ---- 2. Загрузка LLM
    log.info("Загружаем модель %s ...", cfg["model"]["name"])
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    dtype = dtype_map[cfg["model"]["dtype"]]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        torch_dtype=dtype,
        device_map=cfg["model"]["device_map"],
    )
    model.eval()
    log.info(
        "Модель загружена: d_model=%d, n_layers=%d",
        model.config.hidden_size, model.config.num_hidden_layers,
    )

    # ---- 3. Экстракция активаций
    act_cfg = ActivationConfig(
        layer_idx=cfg["extraction"]["layer_idx"],
        max_length=cfg["extraction"]["max_length"],
        stride=cfg["extraction"]["stride"],
        min_position=cfg["extraction"]["min_position"],
        batch_size=cfg["extraction"]["batch_size"],
        only_assistant_positions=cfg["extraction"]["only_assistant_positions"],
    )
    log.info("Запускаем полное извлечение активаций ... (cfg=%s)", act_cfg)
    pack = extract_activations(samples, model, tokenizer, act_cfg)

    # ---- 4. Сохранение
    save_activations(pack, act_path)
    log.info(
        "Готово. Активаций: %d, hidden.shape=%s, ~%.1f МБ.",
        pack["hidden"].shape[0], pack["hidden"].shape,
        pack["hidden"].nbytes / 1e6,
    )


if __name__ == "__main__":
    main()
