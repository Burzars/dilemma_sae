#!/usr/bin/env python3
"""07_steering.py — причинная проверка триггерных нейронов через activation steering.

Вмешиваемся в резстрим Qwen на том же слое, откуда брались активации:
    h_new = h + alpha * sae.W_dec[neuron_id]
и смотрим, сдвигается ли решение модели (доля Yes), классифицируя генерации
LLM-судьёй (Yes/No/?). Ожидание: у Yes-триггера p(Yes) растёт с alpha, у
контрольного нейрона — кривая плоская.

ВАЖНО про нейроны: индексы признаков SAE привязаны к КОНКРЕТНОМУ обученному
SAE. После переобучения старые ID невалидны, поэтому по умолчанию цели
выбираются ПРОГРАММНО из Yes/No-контраста текущего SAE (+ один случайный
контроль похожей частоты). Явный список можно задать в steering.neuron_ids
(например, для приложенного «родного» sae_topk.pt).

Артефакты:
  artifacts/steering_results.json   — по нейрону → по alpha → по промпту counts+answers
  artifacts/steering_summary.csv    — neuron_id, theme, kind, alpha, yes/no/q, p_yes
  artifacts/figures/steering_curves.png

Стоимость полного прогона велика (≈1000 генераций + ≈1000 запросов судьи, ~3 ч).
Результат кешируется ПОЯЧЕЙКОВО (neuron × alpha) — повторный запуск дописывает
недостающее. Сначала прогоните --smoke.

Запуск:
    export OPENROUTER_API_KEY=sk-...
    python scripts/07_steering.py --smoke      # 1 нейрон × 2 alpha × 2 промпта × 1 ген = 4
    python scripts/07_steering.py              # полный прогон
    python scripts/07_steering.py --force      # игнорировать кеш
"""

from __future__ import annotations

import asyncio
import csv
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
        help="Smoke-тест: 1 нейрон × 2 alpha × 2 промпта × 1 генерация = 4 генерации.",
    )
    return parser


def _alpha_key(alpha: float) -> str:
    return f"alpha_{alpha:+.1f}"


def _load_themes_map(artifacts_dir: Path, mode: str) -> dict:
    """{neuron_id: theme} из neuron_themes_<mode>.json, если он есть (иначе {})."""
    p = artifacts_dir / f"neuron_themes_{mode}.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        themed = json.load(f)
    out = {}
    for item in themed:
        t = item.get("theme", "")
        if t and not t.startswith("ERROR") and t != "NO CLEAR THEME":
            out.setdefault(int(item["neuron_id"]), t)
    return out


def _load_or_encode_features(sae, pack, cfg, artifacts_dir, mode, device):
    """features_<mode>.npz из 03, иначе считаем encode_all (на лету)."""
    from src.analysis import encode_all
    fp = artifacts_dir / f"features_{mode}.npz"
    if fp.exists():
        log.info("Загружаю кеш features: %s", fp)
        return np.load(fp, allow_pickle=False)["features"]
    log.info("features_%s.npz не найден — кодирую активации на лету ...", mode)
    return encode_all(sae, pack["hidden"],
                      batch_size=cfg["analysis"]["encode_batch_size"], device=device)


def _select_neurons(cfg, mode, artifacts_dir, sae, pack, samples, device) -> list[dict]:
    """Список целей: [{neuron_id, kind, theme, is_control}]."""
    st = cfg["steering"]
    themes_map = _load_themes_map(artifacts_dir, mode)
    neurons: list[dict] = []
    chosen: set[int] = set()

    explicit = [int(n) for n in st["neuron_ids"]]
    if explicit:
        log.info("Использую явные neuron_ids из конфига: %s", explicit)
        for nid in explicit:
            neurons.append({"neuron_id": nid, "kind": "explicit",
                            "theme": themes_map.get(nid, ""), "is_control": False})
            chosen.add(nid)
    else:
        log.info("Авто-выбор целей из Yes/No-контраста (n_yes=%d, n_no=%d) ...",
                 st["n_yes"], st["n_no"])
        from src.analysis import (
            find_contrast_neurons, label_contrast, sample_level_contrast,
        )
        features = _load_or_encode_features(sae, pack, cfg, artifacts_dir, mode, device)
        tok = label_contrast(
            features, pack["label"],
            pos_label=cfg["contrast"]["pos_label"], neg_label=cfg["contrast"]["neg_label"],
            min_fires=cfg["analysis"]["min_fires"],
        )
        smp = sample_level_contrast(
            features, pack["sample_idx"], samples,
            pos_label=cfg["contrast"]["pos_label"], neg_label=cfg["contrast"]["neg_label"],
        )
        yes_n, no_n = find_contrast_neurons(
            tok, smp,
            top_k=max(st["n_yes"], st["n_no"], cfg["contrast"]["top_k"]),
            pool_factor=cfg["contrast"]["pool_factor"],
        )
        for nid in yes_n[: st["n_yes"]]:
            neurons.append({"neuron_id": int(nid), "kind": "yes_trigger",
                            "theme": themes_map.get(int(nid), ""), "is_control": False})
            chosen.add(int(nid))
        for nid in no_n[: st["n_no"]]:
            if int(nid) in chosen:
                continue
            neurons.append({"neuron_id": int(nid), "kind": "no_trigger",
                            "theme": themes_map.get(int(nid), ""), "is_control": False})
            chosen.add(int(nid))

    # Случайный контроль похожей частоты
    if st["add_random_control"]:
        from src.steering import pick_random_control_neuron
        fr = _fire_rate(cfg, mode, artifacts_dir, sae, pack, device)
        lo, hi = st["random_fire_rate_range"]
        rc = pick_random_control_neuron(fr, lo, hi, exclude=chosen, seed=st["random_seed"])
        neurons.append({"neuron_id": rc, "kind": "control",
                        "theme": "random control", "is_control": True})
        log.info("Контрольный нейрон: %d (fire_rate=%.4f)", rc, float(fr[rc]))

    return neurons


def _fire_rate(cfg, mode, artifacts_dir, sae, pack, device) -> np.ndarray:
    """fire_rate из neuron_stats_<mode>.npz, иначе считаем по features."""
    sp = artifacts_dir / f"neuron_stats_{mode}.npz"
    if sp.exists():
        return np.load(sp, allow_pickle=False)["fire_rate"]
    from src.analysis import neuron_statistics
    features = _load_or_encode_features(sae, pack, cfg, artifacts_dir, mode, device)
    return neuron_statistics(features)["fire_rate"]


def _load_prompts(samples) -> tuple[list[int], list[str]]:
    """По одному промпту на file_idx, упорядоченные по file_idx."""
    by_file: dict[int, str] = {}
    for s in samples:
        by_file.setdefault(s.file_idx, s.prompt)
    keys = sorted(by_file.keys())
    return keys, [by_file[k] for k in keys]


def main() -> None:
    cfg, args = parse_args_and_init(
        "07_steering",
        description="Activation steering триггерных нейронов SAE (proof-of-concept).",
        extra_args=_add_args,
    )

    from src.data import load_saved_samples
    from src.interpret import classify_decisions_async
    from src.io_utils import load_activations, load_sae
    from src.steering import generate_batched, resolve_steer_layer, steering_hook

    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    mode = cfg["sae"]["mode"]
    sae_path = artifacts_dir / f"sae_{mode}.pt"
    suffix = "_smoke" if args.smoke else ""
    results_path = artifacts_dir / f"steering_results{suffix}.json"
    summary_path = artifacts_dir / f"steering_summary{suffix}.csv"

    if not sae_path.exists():
        raise FileNotFoundError(
            f"SAE не найден: {sae_path}. Сначала 02_train_sae.py (и желательно 03_analyze.py)."
        )

    st = cfg["steering"]
    alphas = list(st["alphas"])
    n_gen = st["n_generations"]
    gen_bs = st["gen_batch_size"]
    log_every = st["log_every"]
    judge_model = st["judge_model"]
    api_base = cfg["interpret"]["api_base"]
    concurrency = st["judge_concurrency"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Устройство: %s", device)
    if device == "cpu":
        log.warning("GPU недоступна — генерация на CPU очень медленная (это POC для сервера).")

    # ---- Данные, SAE, модель
    pack = load_activations(artifacts_dir / "activations.npz")
    samples = load_saved_samples(artifacts_dir / "samples.json")
    file_keys, prompts = _load_prompts(samples)
    sae, _ckpt = load_sae(sae_path, device=device)

    neurons = _select_neurons(cfg, mode, artifacts_dir, sae, pack, samples, device)

    # ---- Smoke-урезание
    if args.smoke:
        neurons = neurons[:1]
        alphas = [0.0, 5.0]
        prompts = prompts[:2]
        file_keys = file_keys[:2]
        n_gen = 1
        log.info("SMOKE: нейроны=%s, alphas=%s, промптов=%d, n_gen=%d",
                 [n["neuron_id"] for n in neurons], alphas, len(prompts), n_gen)

    log.info("Цели (%d): %s", len(neurons),
             [(n["neuron_id"], n["kind"]) for n in neurons])
    log.info("alphas=%s, промптов=%d, генераций/ячейку=%d", alphas, len(prompts), n_gen)

    # ---- Модель Qwen
    log.info("Загружаю модель %s ...", cfg["model"]["name"])
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        torch_dtype=dtype_map[cfg["model"]["dtype"]],
        device_map=cfg["model"]["device_map"],
    )
    model.eval()

    module_idx = resolve_steer_layer(model, cfg["extraction"]["layer_idx"])
    log.info("Стиринг на model.model.layers[%d] (layer_idx=%s, n_layers=%d) — "
             "его выход = hidden_states[%s], откуда брались активации.",
             module_idx, cfg["extraction"]["layer_idx"],
             model.config.num_hidden_layers, cfg["extraction"]["layer_idx"])

    # ---- Кеш/резюмирование
    if results_path.exists() and not args.force:
        with open(results_path, encoding="utf-8") as f:
            results = json.load(f)
        log.info("Подгружен частичный результат: %s — недостающее досчитаю.", results_path)
    else:
        results = {}

    total_cells = len(neurons) * len(alphas)
    gen_counter = {"n": 0}
    cell_no = 0

    for nrec in neurons:
        nid = nrec["neuron_id"]
        nkey = f"neuron_{nid}"
        results.setdefault(nkey, {
            "neuron_id": nid, "kind": nrec["kind"],
            "theme": nrec["theme"], "is_control": nrec["is_control"],
        })
        vec = sae.W_dec[nid].detach()   # (d_in,), единичная норма; hook сам приведёт dtype/device

        for alpha in alphas:
            cell_no += 1
            akey = _alpha_key(alpha)
            done = results[nkey].get(akey)
            if done and not args.force and len(
                [k for k in done if k.startswith("prompt_")]) == len(prompts):
                log.info("[%d/%d] %s %s — уже в кеше, пропускаю.",
                         cell_no, total_cells, nkey, akey)
                continue

            log.info("[%d/%d] нейрон %d (%s) alpha=%+.1f — генерация ...",
                     cell_no, total_cells, nid, nrec["kind"], alpha)

            answers_by_prompt = {i: [] for i in range(len(prompts))}
            with steering_hook(model, module_idx, vec, alpha):
                for g in range(n_gen):
                    outs = generate_batched(
                        model, tokenizer, prompts,
                        max_new_tokens=st["max_new_tokens"],
                        temperature=st["temperature"],
                        seed=st["base_seed"] + g,
                        batch_size=gen_bs,
                    )
                    for i, txt in enumerate(outs):
                        answers_by_prompt[i].append(txt)
                    gen_counter["n"] += len(prompts)
                    if gen_counter["n"] % log_every < len(prompts):
                        log.info("    ... всего генераций: %d", gen_counter["n"])

            # Классификация судьёй (async, параллельно)
            flat = [t for i in range(len(prompts)) for t in answers_by_prompt[i]]
            labels = asyncio.run(classify_decisions_async(
                flat, judge_model=judge_model, api_base=api_base, concurrency=concurrency,
            ))

            cell = {}
            k = 0
            for i in range(len(prompts)):
                m = len(answers_by_prompt[i])
                labs = labels[k:k + m]
                k += m
                cell[f"prompt_{file_keys[i]}"] = {
                    "yes_count": labs.count("Yes"),
                    "no_count": labs.count("No"),
                    "q_count": labs.count("?"),
                    "answers": answers_by_prompt[i],
                    "labels": labs,
                }
            results[nkey][akey] = cell

            # Инкрементальное сохранение (чтобы сбой не терял прогресс)
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            ty = sum(c["yes_count"] for c in cell.values())
            tot = sum(c["yes_count"] + c["no_count"] + c["q_count"] for c in cell.values())
            log.info("    нейрон %d alpha=%+.1f: yes=%d/%d (p_yes=%.3f)",
                     nid, alpha, ty, tot, ty / tot if tot else float("nan"))

    # ---- Сводная CSV + кривые
    _write_summary_and_plot(results, neurons, alphas, summary_path, figures_dir, suffix)
    log.info("Готово. Результаты: %s ; сводка: %s", results_path, summary_path)


def _write_summary_and_plot(results, neurons, alphas, summary_path, figures_dir, suffix):
    from src.visualize import plot_steering_curves

    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["neuron_id", "theme", "kind", "is_control",
                    "alpha", "total_yes", "total_no", "total_q", "p_yes"])
        curves: dict[str, dict] = {}
        for nrec in neurons:
            nid = nrec["neuron_id"]
            nkey = f"neuron_{nid}"
            label = f"{nid} ({nrec['kind']})"
            curves[label] = {"alphas": [], "p_yes": [], "is_control": nrec["is_control"]}
            for alpha in alphas:
                cell = results.get(nkey, {}).get(_alpha_key(alpha))
                if not cell:
                    continue
                prompt_cells = [v for kk, v in cell.items() if kk.startswith("prompt_")]
                ty = sum(c["yes_count"] for c in prompt_cells)
                tn = sum(c["no_count"] for c in prompt_cells)
                tq = sum(c["q_count"] for c in prompt_cells)
                tot = ty + tn + tq
                p_yes = ty / tot if tot else float("nan")
                w.writerow([nid, nrec["theme"], nrec["kind"], nrec["is_control"],
                            f"{alpha:+.1f}", ty, tn, tq, f"{p_yes:.4f}"])
                curves[label]["alphas"].append(alpha)
                curves[label]["p_yes"].append(p_yes)

    plot_steering_curves(curves, figures_dir, name=f"steering_curves{suffix}")


if __name__ == "__main__":
    main()
