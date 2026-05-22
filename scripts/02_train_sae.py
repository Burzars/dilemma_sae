#!/usr/bin/env python3
"""Обучение одной SAE (mode из config) на закешированных активациях.

Артефакт: artifacts/sae_<mode>.pt. Если файл уже есть и нет --force,
шаг пропускается.

Запуск:
    python scripts/02_train_sae.py                              # TopK по умолчанию
    python scripts/02_train_sae.py --override sae.mode=relu_l1
    python scripts/02_train_sae.py --override sae.mode=topk sae.k=16
    python scripts/02_train_sae.py --override sae.mode=jumprelu \\
                                            sae.l0_coef=2e-2 sae.theta_init=1.0

Чтобы обучить три варианта подряд:
    for mode in relu_l1 topk jumprelu; do
        python scripts/02_train_sae.py --override sae.mode=$mode
    done
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from _common import parse_args_and_init


log = logging.getLogger(__name__)


def _sae_kwargs(sae_cfg: dict) -> dict:
    """Собирает только релевантные для текущего режима kwargs SAE.

    Параметры других режимов всё равно валидно принимаются конструктором,
    но передавать их явно — лишний шум в логах.
    """
    mode = sae_cfg["mode"]
    kw = {"mode": mode}
    if mode == "relu_l1":
        kw["l1_coef"] = sae_cfg["l1_coef"]
    elif mode == "topk":
        kw["k"] = sae_cfg["k"]
    elif mode == "jumprelu":
        kw["l0_coef"] = sae_cfg["l0_coef"]
        kw["theta_init"] = sae_cfg["theta_init"]
        kw["ste_eps"] = sae_cfg["ste_eps"]
    return kw


def main() -> None:
    cfg, args = parse_args_and_init(
        "02_train_sae",
        description="Обучение одного SAE из cfg.sae.mode.",
    )

    from src.io_utils import load_activations, save_sae
    from src.sae import train_sae

    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    sae_cfg = cfg["sae"]
    mode = sae_cfg["mode"]
    sae_path = artifacts_dir / f"sae_{mode}.pt"
    act_path = artifacts_dir / "activations.npz"

    if sae_path.exists() and not args.force:
        log.info("SAE уже обучен: %s — пропускаю.", sae_path)
        log.info("Используйте --force, чтобы переобучить.")
        return

    if not act_path.exists():
        raise FileNotFoundError(
            f"Не найдены активации: {act_path}. "
            "Сначала запустите scripts/01_extract.py."
        )

    pack = load_activations(act_path)
    hidden = pack["hidden"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Устройство: %s", device)
    if torch.cuda.is_available():
        log.info("GPU: %s", torch.cuda.get_device_name(0))

    kwargs = _sae_kwargs(sae_cfg)
    log.info("Параметры SAE: %s", kwargs)
    log.info(
        "Обучение: expansion=%d, epochs=%d, batch_size=%d, lr=%g, seed=%d",
        sae_cfg["expansion"], sae_cfg["epochs"], sae_cfg["batch_size"],
        sae_cfg["lr"], sae_cfg["seed"],
    )

    sae, history = train_sae(
        hidden=hidden,
        expansion=sae_cfg["expansion"],
        lr=sae_cfg["lr"],
        epochs=sae_cfg["epochs"],
        batch_size=sae_cfg["batch_size"],
        device=device,
        seed=sae_cfg["seed"],
        **kwargs,
    )

    save_sae(sae, history, kwargs, sae_path)
    log.info(
        "Финальные метрики: recon=%.3f, L0=%.1f, L1=%.1f",
        history["recon"][-1], history["l0"][-1], history["l1"][-1],
    )


if __name__ == "__main__":
    main()
