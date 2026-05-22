"""IO-утилиты: YAML-конфиги, --override, save/load активаций и SAE,
настройка логирования.

Конфиг загружается из YAML, опционально переопределяется парами
key=value (с поддержкой dotted-нотации: sae.k=16). Значения парсятся
из строки в Python-типы (int → float → bool → str).
"""

from __future__ import annotations

import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml


# ----------------------------------------------------------------------------
# Конфиги
# ----------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    """Читает YAML-файл и возвращает словарь."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _coerce(value: str) -> Any:
    """Парсит строку из CLI в Python-тип.

    Порядок попыток: int → float → bool → None → str.
    """
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    if value.lower() in ("none", "null"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    """Применяет к cfg переопределения вида "sae.k=16".

    Возвращает новый словарь (deepcopy). Поддерживает вложенные ключи.
    """
    cfg = deepcopy(cfg)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Неверный формат override: {ov!r}. Ожидаю key=value.")
        key, _, raw_val = ov.partition("=")
        key = key.strip()
        val = _coerce(raw_val.strip())

        # Спустимся по dotted-ключам, создавая словари при необходимости
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = val
    return cfg


# ----------------------------------------------------------------------------
# Логирование
# ----------------------------------------------------------------------------

def setup_logging(script_name: str, cfg: dict) -> Path:
    """Настраивает корневой логгер: stdout + файл artifacts/logs/<script_name>.log.

    Возвращает путь к log-файлу.
    """
    level_name = cfg.get("logging", {}).get("level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)
    logs_dir = Path(cfg["paths"]["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{script_name}.log"

    # Полностью пересоберём root logger, чтобы повторные вызовы не
    # копили хендлеры (актуально, например, в ноутбуке).
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if cfg.get("logging", {}).get("console", True):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    if cfg.get("logging", {}).get("file", True):
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    logging.getLogger(__name__).info(
        "Логирование инициализировано: %s (level=%s)", log_path, level_name,
    )
    return log_path


# ----------------------------------------------------------------------------
# Активации (npz)
# ----------------------------------------------------------------------------

def save_activations(pack: dict, path: str | Path) -> None:
    """Сохраняет dict с ключами hidden/sample_idx/token_pos/label в .npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        hidden=pack["hidden"],
        sample_idx=pack["sample_idx"],
        token_pos=pack["token_pos"],
        label=pack["label"],
    )
    logging.getLogger(__name__).info(
        "Активации сохранены: %s (hidden.shape=%s)", path, pack["hidden"].shape,
    )


def load_activations(path: str | Path) -> dict:
    """Загружает то, что сохранил save_activations."""
    path = Path(path)
    npz = np.load(path, allow_pickle=False)
    pack = {
        "hidden":     npz["hidden"],
        "sample_idx": npz["sample_idx"],
        "token_pos":  npz["token_pos"],
        "label":      npz["label"],
    }
    logging.getLogger(__name__).info(
        "Активации загружены: %s (hidden.shape=%s)", path, pack["hidden"].shape,
    )
    return pack


# ----------------------------------------------------------------------------
# SAE (pt)
# ----------------------------------------------------------------------------

def save_sae(sae, history: dict, mode_cfg: dict, path: str | Path) -> None:
    """Сохраняет state_dict + параметры SAE + train-history.

    Формат файла:
      {
        "state_dict": ...,
        "d_in":       int,
        "d_hidden":   int,
        "mode":       str,
        "config":     dict   # параметры конструктора SAE
        "history":    dict   # recon / l1 / l0 / total по эпохам
      }
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": sae.state_dict(),
        "d_in":       sae.d_in,
        "d_hidden":   sae.d_hidden,
        "mode":       sae.mode,
        "config":     mode_cfg,
        "history":    history,
    }, path)
    logging.getLogger(__name__).info("SAE сохранён: %s", path)


def load_sae(path: str | Path, device: str = "cuda"):
    """Загружает SAE с диска. Возвращает (sae, ckpt_dict).

    Восстанавливает архитектуру по полям d_in / d_hidden / mode / config.
    """
    from .sae import SparseAutoencoder

    path = Path(path)
    # weights_only=False — нам нужен dict с историей, не только тензоры.
    # Это безопасно, потому что чекпойнты создаём мы сами.
    ckpt = torch.load(path, map_location=device, weights_only=False)

    sae_kwargs = dict(ckpt["config"])
    # mode идёт отдельно — он не должен дублироваться в config; но если
    # дублируется, mode из top-level выигрывает.
    sae_kwargs.pop("mode", None)
    sae = SparseAutoencoder(
        d_in=ckpt["d_in"],
        d_hidden=ckpt["d_hidden"],
        mode=ckpt["mode"],
        **sae_kwargs,
    ).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    logging.getLogger(__name__).info(
        "SAE загружён: %s (mode=%s, d_hidden=%d)",
        path, ckpt["mode"], ckpt["d_hidden"],
    )
    return sae, ckpt
