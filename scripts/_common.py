"""Общий CLI-парсер для всех скриптов проекта.

Использование в скрипте:

    from _common import parse_args_and_init
    cfg, args = parse_args_and_init("01_extract")
    # cfg — финальный dict, с применёнными --override
    # args — argparse.Namespace (содержит args.force и пр.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Делаем src/ импортируемым, когда скрипт запущен напрямую.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import apply_overrides, load_config, setup_logging  # noqa: E402


def build_argparser(script_name: str, description: str = "") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=script_name, description=description)
    p.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default.yaml",
        help="Путь к YAML-конфигу (по умолчанию configs/default.yaml).",
    )
    p.add_argument(
        "--override", "-o",
        nargs="*",
        default=[],
        help=(
            "Переопределение параметров вида key=value (поддерживается "
            "dotted-нотация). Пример: --override sae.mode=topk sae.k=16"
        ),
    )
    p.add_argument(
        "--force", "-f",
        action="store_true",
        help="Не пропускать шаг, даже если артефакт уже есть на диске.",
    )
    return p


def parse_args_and_init(
    script_name: str,
    description: str = "",
    extra_args=None,
):
    """Парсит CLI, грузит конфиг, применяет --override, поднимает логирование.

    Args
    ----
    extra_args : callable, optional
        Функция (parser) → parser, добавляющая специфичные для скрипта
        аргументы. Например: добавить --sae-name к 02_train_sae.

    Returns
    -------
    (cfg: dict, args: argparse.Namespace)
    """
    parser = build_argparser(script_name, description)
    if extra_args is not None:
        parser = extra_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    # Проинициализируем каталоги
    for key in ("artifacts_dir", "figures_dir", "logs_dir"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)

    setup_logging(script_name, cfg)
    return cfg, args
