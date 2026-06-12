#!/usr/bin/env python3
"""07_steering.py — причинная проверка триггерных нейронов через activation steering.

Вмешиваемся в резстрим Qwen на том же слое, откуда брались активации:
    h_new = h + alpha * sae.W_dec[neuron_id]
и смотрим, сдвигается ли решение модели (доля Yes), классифицируя генерации
LLM-судьёй (Yes/No/?). Ожидание: у Yes-триггера p(Yes) растёт с alpha, у
контрольного нейрона — кривая плоская.

Корректный протокол (Задача 3):
  * alpha масштабируется от данных: сетка = доли median||h|| на слое -5
    (src/steering.alpha_scale_grid), а не магические ±2/±5 (~3% нормы → не двигало).
  * temperature=1.0, РАЗНЫЙ seed на каждый сэмпл → N генераций независимы.
  * >= 30 генераций на (нейрон, alpha, промпт); на каждую ячейку — bootstrap-CI
    (95%, percentile) долей Yes/No/?.
  * промпты отбираются по податливости АВТОМАТИЧЕСКИ из бейзлайна (alpha=0):
    push_down (Yes-смещённые, толкаем к No) и push_up (?/No-смещённые, к Yes).
  * критерий «сработало»: кривая триггера монотонна по alpha И её CI не
    пересекается с CI контроля на крайних alpha. Иначе — честный вывод
    «эффект неотличим от нуля/контроля». Вердикт печатается в лог.

ВАЖНО про нейроны: индексы признаков SAE привязаны к КОНКРЕТНОМУ обученному
SAE. После переобучения старые ID невалидны, поэтому по умолчанию цели
выбираются ПРОГРАММНО из Yes/No-контраста (на BALANCED-срезе) текущего SAE
(+ один случайный контроль). Явный список — в steering.neuron_ids.

Артефакты:
  artifacts/steering_results.json   — _meta (median_norm, сетка alpha, наборы
                                       промптов), _baseline, и по нейрону → по alpha:
                                       agg + per_prompt {counts Yes/No/?, доли, CI}
  artifacts/steering_summary.csv    — neuron, kind, alpha, n, yes/no/q, p_yes + CI
  artifacts/figures/steering_curves.png  — p(Yes) vs alpha с полосами CI + контроль

Результат кешируется ПОЯЧЕЙКОВО (neuron × alpha) — повторный запуск дописывает
недостающее. Сначала прогоните --smoke.

Запуск:
    export OPENROUTER_API_KEY=sk-...
    python scripts/07_steering.py --smoke      # дёшево: 1 нейрон × 3 alpha, n_gen=1
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
            contrast_feature_slice, find_contrast_neurons,
            label_contrast, sample_level_contrast,
        )
        features = _load_or_encode_features(sae, pack, cfg, artifacts_dir, mode, device)
        # Контраст — на BALANCED-срезе (как в 03_analyze), а features/fire_rate
        # остаются на FULL. Балансировку выбираем тем же конфигом contrast.slice.
        c_features, c_labels, c_sample_idx = contrast_feature_slice(
            features, pack["label"], pack["sample_idx"], samples,
            slice=cfg["contrast"].get("slice", "balanced"),
            balanced_dir=cfg["contrast"].get("balanced_dir"),
            balanced_seed=cfg["contrast"].get("balanced_seed", 0),
        )
        tok = label_contrast(
            c_features, c_labels,
            pos_label=cfg["contrast"]["pos_label"], neg_label=cfg["contrast"]["neg_label"],
            min_fires=cfg["analysis"]["min_fires"],
        )
        smp = sample_level_contrast(
            c_features, c_sample_idx, samples,
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


def _load_scenario_meta(results_dir) -> dict:
    """{file_idx: scenario_meta} из файлов генерации (для отбора по податливости)."""
    import re
    out: dict[int, dict] = {}
    p = Path(results_dir)
    if not p.exists():
        return out
    for fp in p.glob("volunteer_data_llama_reasoning_*_res.json"):
        m = re.search(r"_(\d+)_res\.json$", fp.name)
        if not m:
            continue
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(data.get("scenario_meta"), dict):
            out[int(m.group(1))] = data["scenario_meta"]
    return out


def _run_cell(model, tokenizer, module_idx, vec, alpha, prompts, file_keys, *,
              n_gen, st, gen_counter, log_every, judge_model, api_base, concurrency):
    """Генерация + разметка для одной ячейки (alpha, набор промптов).

    Каждый из n_gen сэмплов на промпт получает СВОЙ seed (base_seed+g) при
    temperature=1.0 → сэмплы независимы. Возвращает raw-словарь
    {prompt_<fk>: {"answers": [...], "labels": [...]}}.
    """
    from src.interpret import classify_decisions_async
    from src.steering import generate_batched, steering_hook
    answers_by_prompt = {i: [] for i in range(len(prompts))}
    with steering_hook(model, module_idx, vec, alpha):
        for g in range(n_gen):
            outs = generate_batched(
                model, tokenizer, prompts,
                max_new_tokens=st["max_new_tokens"],
                temperature=st["temperature"],
                seed=st["base_seed"] + g,
                batch_size=st["gen_batch_size"],
            )
            for i, txt in enumerate(outs):
                answers_by_prompt[i].append(txt)
            gen_counter["n"] += len(prompts)
            if gen_counter["n"] % log_every < len(prompts):
                log.info("    ... всего генераций: %d", gen_counter["n"])

    flat = [t for i in range(len(prompts)) for t in answers_by_prompt[i]]
    labels = asyncio.run(classify_decisions_async(
        flat, judge_model=judge_model, api_base=api_base, concurrency=concurrency,
    ))
    raw, k = {}, 0
    for i in range(len(prompts)):
        m = len(answers_by_prompt[i])
        raw[f"prompt_{file_keys[i]}"] = {
            "answers": answers_by_prompt[i], "labels": labels[k:k + m],
        }
        k += m
    return raw


def _summarize_labels(labels, st) -> dict:
    """counts Yes/No/?, доли и bootstrap-CI по списку меток."""
    from src.steering import bootstrap_ci_proportion
    n = len(labels)
    counts = {"Yes": labels.count("Yes"), "No": labels.count("No"), "?": labels.count("?")}
    props = {k: (counts[k] / n if n else float("nan")) for k in counts}
    ci = {}
    for tgt in ("Yes", "No", "?"):
        _, lo, hi = bootstrap_ci_proportion(
            labels, target=tgt, n_boot=st["bootstrap_n"],
            ci=st["bootstrap_ci"], seed=st["bootstrap_seed"],
        )
        ci[tgt] = [lo, hi]
    return {"n": n, "counts": counts, "props": props, "ci": ci}


def _finalize_cell(raw, alpha, st) -> dict:
    """raw-ячейку → JSON-ячейка с counts/долями/CI (per-prompt + агрегат)."""
    per_prompt, pooled = {}, []
    for pk, d in raw.items():
        labs = list(d["labels"])
        pooled += labs
        per_prompt[pk] = {**_summarize_labels(labs, st),
                          "answers": d["answers"], "labels": labs}
    return {"alpha": float(alpha), "agg": _summarize_labels(pooled, st),
            "per_prompt": per_prompt}


def _curve_from_results(results, nkey, alphas, target="Yes") -> dict:
    """Кривая агрегата p(target) vs alpha + CI для одного нейрона."""
    a_out, p_out, ci_out = [], [], []
    for alpha in alphas:
        cell = results.get(nkey, {}).get(_alpha_key(alpha))
        if not cell or "agg" not in cell:
            continue
        a_out.append(alpha)
        p_out.append(cell["agg"]["props"][target])
        lo, hi = cell["agg"]["ci"][target]
        ci_out.append((lo, hi))
    return {"alphas": a_out, "p_yes": p_out, "ci": ci_out}


def main() -> None:
    cfg, args = parse_args_and_init(
        "07_steering",
        description="Activation steering триггерных нейронов SAE (proof-of-concept).",
        extra_args=_add_args,
    )

    from src.data import load_saved_samples
    from src.io_utils import load_activations, load_sae
    from src.steering import (
        alpha_scale_grid, resolve_steer_layer, select_susceptible_prompts,
    )

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
    n_gen = st["n_generations"]
    log_every = st["log_every"]
    judge_model = st["judge_model"]
    api_base = cfg["interpret"]["api_base"]
    concurrency = st["judge_concurrency"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Устройство: %s", device)
    if device == "cpu":
        log.warning("GPU недоступна — генерация на CPU очень медленная (это POC для сервера).")

    # ---- Данные, SAE
    pack = load_activations(artifacts_dir / "activations.npz")
    samples = load_saved_samples(artifacts_dir / "samples.json")
    file_keys, prompts = _load_prompts(samples)
    meta_by_file = _load_scenario_meta(cfg["data"]["results_dir"])
    sae, _ckpt = load_sae(sae_path, device=device)

    neurons = _select_neurons(cfg, mode, artifacts_dir, sae, pack, samples, device)

    # ---- Сетка alpha: доли от медианы нормы ||h|| (Задача 3), если alphas не задан явно
    if st.get("alphas"):
        alphas = [float(a) for a in st["alphas"]]
        median_norm = float("nan")
        log.info("alphas заданы явно в конфиге: %s", alphas)
    else:
        alphas, median_norm = alpha_scale_grid(pack["hidden"], st["alpha_fractions"])
        log.info("median||h|| на слое экстракции = %.2f; alpha_fractions=%s",
                 median_norm, st["alpha_fractions"])
        log.info("Сетка alpha (доли нормы): %s", [round(a, 2) for a in alphas])

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
    log.info("Стиринг на model.model.layers[%d] (layer_idx=%s, n_layers=%d).",
             module_idx, cfg["extraction"]["layer_idx"], model.config.num_hidden_layers)

    # ---- Smoke-урезание (после расчёта сетки)
    if args.smoke:
        neurons = neurons[:1]
        alphas = [alphas[0], 0.0, alphas[-1]]
        prompts, file_keys = prompts[:4], file_keys[:4]
        n_gen = 1
        st = {**st, "prompts_per_set": 1}
        log.info("SMOKE: нейрон=%s, alphas=%s, кандидат-промптов=%d, n_gen=%d",
                 [n["neuron_id"] for n in neurons], [round(a, 2) for a in alphas],
                 len(prompts), n_gen)

    # ---- Кеш/резюмирование
    if results_path.exists() and not args.force:
        with open(results_path, encoding="utf-8") as f:
            results = json.load(f)
        log.info("Подгружен частичный результат: %s — недостающее досчитаю.", results_path)
    else:
        results = {}

    gen_counter = {"n": 0}

    def _save():
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # ---- (1) Бейзлайн alpha=0 по ВСЕМ промптам (нейрон-независим: hook не вешается) ----
    base_keys = [f"prompt_{fk}" for fk in file_keys]
    if ("_baseline" in results and not args.force
            and all(k in results["_baseline"].get("per_prompt", {}) for k in base_keys)):
        baseline = results["_baseline"]
        log.info("Бейзлайн (alpha=0) взят из кеша.")
    else:
        log.info("Считаю бейзлайн alpha=0 по %d промптам (%d ген/промпт) ...",
                 len(prompts), n_gen)
        raw = _run_cell(model, tokenizer, module_idx, None, 0.0, prompts, file_keys,
                        n_gen=n_gen, st=st, gen_counter=gen_counter, log_every=log_every,
                        judge_model=judge_model, api_base=api_base, concurrency=concurrency)
        baseline = _finalize_cell(raw, 0.0, st)
        results["_baseline"] = baseline
        _save()

    baseline_p_yes = {fk: baseline["per_prompt"][f"prompt_{fk}"]["props"]["Yes"]
                      for fk in file_keys}
    log.info("Бейзлайн p_yes по промптам: %s",
             {fk: round(v, 2) for fk, v in baseline_p_yes.items()})

    # ---- (2) Отбор промптов по податливости (по бейзлайну) ----
    if st.get("prompt_selection", "susceptible") == "all":
        sel_sets = {"push_down": list(file_keys), "push_up": []}
    else:
        sel_sets = select_susceptible_prompts(
            baseline_p_yes, k_each=st["prompts_per_set"],
            down_yes_min=st["down_yes_min"], up_yes_max=st["up_yes_max"],
        )
    sel_keys = sorted(set(sel_sets["push_down"]) | set(sel_sets["push_up"]))
    key_to_prompt = dict(zip(file_keys, prompts))
    sel_prompts = [key_to_prompt[k] for k in sel_keys]
    results["_meta"] = {
        "median_norm": median_norm, "alphas": alphas, "n_generations": n_gen,
        "temperature": st["temperature"], "prompt_sets": sel_sets,
        "scenario_meta": {str(k): meta_by_file.get(k, {}) for k in sel_keys},
    }
    _save()
    log.info("Отобраны промпты: push_down(к No, Yes-bias)=%s | push_up(к Yes, ?/No-bias)=%s",
             sel_sets["push_down"], sel_sets["push_up"])

    # ---- (3) Свип alpha по отобранным промптам для каждого нейрона ----
    total_cells = len(neurons) * len(alphas)
    cell_no = 0
    for nrec in neurons:
        nid = nrec["neuron_id"]
        nkey = f"neuron_{nid}"
        results.setdefault(nkey, {
            "neuron_id": nid, "kind": nrec["kind"],
            "theme": nrec["theme"], "is_control": nrec["is_control"],
        })
        vec = sae.W_dec[nid].detach()

        for alpha in alphas:
            cell_no += 1
            akey = _alpha_key(alpha)
            done = results[nkey].get(akey)
            if (done and not args.force
                    and all(f"prompt_{k}" in done.get("per_prompt", {}) for k in sel_keys)):
                log.info("[%d/%d] %s %s — в кеше, пропускаю.", cell_no, total_cells, nkey, akey)
                continue

            if abs(alpha) < 1e-9:
                # alpha=0 нейрон-независим → переиспользуем бейзлайн (без генерации)
                sub = {pk: baseline["per_prompt"][pk]
                       for pk in (f"prompt_{k}" for k in sel_keys)}
                cell = {"alpha": 0.0,
                        "agg": _summarize_labels(
                            [l for d in sub.values() for l in d["labels"]], st),
                        "per_prompt": sub}
            else:
                log.info("[%d/%d] нейрон %d (%s) alpha=%+.2f — генерация ...",
                         cell_no, total_cells, nid, nrec["kind"], alpha)
                raw = _run_cell(model, tokenizer, module_idx, vec, alpha,
                                sel_prompts, sel_keys, n_gen=n_gen, st=st,
                                gen_counter=gen_counter, log_every=log_every,
                                judge_model=judge_model, api_base=api_base,
                                concurrency=concurrency)
                cell = _finalize_cell(raw, alpha, st)

            results[nkey][akey] = cell
            _save()
            agg = cell["agg"]
            log.info("    нейрон %d alpha=%+.2f: p_yes=%.3f CI=[%.2f,%.2f] (n=%d)",
                     nid, alpha, agg["props"]["Yes"], agg["ci"]["Yes"][0],
                     agg["ci"]["Yes"][1], agg["n"])

    # ---- (4) Сводка, график с CI + контроль, формальный критерий ----
    _summarize_and_plot(results, neurons, alphas, summary_path, figures_dir, suffix)
    log.info("Готово. Результаты: %s ; сводка: %s", results_path, summary_path)


def _summarize_and_plot(results, neurons, alphas, summary_path, figures_dir, suffix):
    from src.steering import steering_verdict
    from src.visualize import plot_steering_curves

    # CSV: агрегат по отобранным промптам, с CI на p_yes
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["neuron_id", "theme", "kind", "is_control", "alpha", "n",
                    "yes", "no", "q", "p_yes", "p_yes_lo", "p_yes_hi"])
        for nrec in neurons:
            nkey = f"neuron_{nrec['neuron_id']}"
            for alpha in alphas:
                cell = results.get(nkey, {}).get(_alpha_key(alpha))
                if not cell or "agg" not in cell:
                    continue
                a = cell["agg"]
                lo, hi = a["ci"]["Yes"]
                w.writerow([nrec["neuron_id"], nrec["theme"], nrec["kind"],
                            nrec["is_control"], f"{alpha:+.2f}", a["n"],
                            a["counts"]["Yes"], a["counts"]["No"], a["counts"]["?"],
                            f"{a['props']['Yes']:.4f}", f"{lo:.4f}", f"{hi:.4f}"])

    # Кривые p(Yes) vs alpha с полосами CI; контроль — пунктиром
    curves: dict[str, dict] = {}
    control_curve = None
    for nrec in neurons:
        nkey = f"neuron_{nrec['neuron_id']}"
        c = _curve_from_results(results, nkey, alphas, target="Yes")
        c["is_control"] = nrec["is_control"]
        label = f"{nrec['neuron_id']} ({nrec['kind']})"
        curves[label] = c
        if nrec["is_control"]:
            control_curve = c
    plot_steering_curves(curves, figures_dir, name=f"steering_curves{suffix}")

    # Формальный критерий: триггер сработал ⇔ монотонная кривая И CI не пересекает
    # CI контроля на крайних alpha. Иначе — честный вывод «неотличим от нуля».
    if control_curve is None:
        log.warning("Контрольный нейрон отсутствует — критерий не применяется "
                    "(добавьте steering.add_random_control=true).")
        return
    log.info("=== Вердикт steering (критерий: монотонность + разделение CI с контролем) ===")
    for nrec in neurons:
        if nrec["is_control"]:
            continue
        nkey = f"neuron_{nrec['neuron_id']}"
        trig = _curve_from_results(results, nkey, alphas, target="Yes")
        if len(trig["alphas"]) < 2:
            continue
        worked, reason, _ = steering_verdict(trig, control_curve)
        log.info("нейрон %d (%s): %s", nrec["neuron_id"], nrec["kind"], reason)


if __name__ == "__main__":
    main()
