"""Activation steering: причинная проверка триггерных нейронов SAE.

Идея (proof-of-concept, не бенчмарк): на том же слое, откуда брались
активации (`extraction.layer_idx`, по умолчанию -5), к выходу декодер-блока
добавляется направление нейрона из декодера SAE:

    h_new = h + alpha * v,        v = sae.W_dec[neuron_id]   (единичной нормы)

и проверяется, сдвигается ли решение модели (Yes/No/?). Если усиление
Yes-триггера повышает долю Yes, а контрольный нейрон — нет, это довод за
причинную (а не просто корреляционную) роль признака.

Хук вешается на модуль `model.model.layers[L]`, ВЫХОД которого совпадает с
`hidden_states[layer_idx]` (см. resolve_steer_layer) — то есть мы вмешиваемся
ровно в тот тензор, который кодировал SAE.
"""

from __future__ import annotations

import contextlib
import logging
from typing import List, Sequence

import numpy as np
import torch

log = logging.getLogger(__name__)


def resolve_steer_layer(model, layer_idx: int) -> int:
    """Индекс блока model.model.layers[*], выход которого = hidden_states[layer_idx].

    `output_hidden_states=True` даёт кортеж длины n_layers+1:
      hidden_states[0]  = эмбеддинги,
      hidden_states[i]  = выход model.model.layers[i-1]   (для i >= 1).
    Поэтому для слоя layer_idx (можно отрицательный) приводим к положительному
    hs-индексу и вычитаем 1. Для Qwen2.5-3B (36 слоёв): layer_idx=-5 →
    hidden_states[32] → блок layers[31].
    """
    n_layers = int(model.config.num_hidden_layers)
    hs_len = n_layers + 1
    hs_idx = layer_idx if layer_idx >= 0 else hs_len + layer_idx
    if not (1 <= hs_idx <= n_layers):
        raise ValueError(
            f"layer_idx={layer_idx} → hidden_states[{hs_idx}] не соответствует "
            f"декодер-блоку (0 — эмбеддинги; допустимо 1..{n_layers})."
        )
    return hs_idx - 1


def get_decoder_layers(model):
    """Список декодер-блоков (model.model.layers у Qwen/Llama-подобных)."""
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise AttributeError(
            "Не найден model.model.layers — проверьте архитектуру модели."
        )
    return layers


@contextlib.contextmanager
def steering_hook(model, module_idx: int, vec: torch.Tensor, alpha: float):
    """Контекст: на время добавляет alpha*vec к выходу блока module_idx.

    alpha == 0.0 → хук не вешается (это и есть чистый baseline). Хук правит
    output[0] на ВСЕХ позициях, включая токены, которые генерируются пошагово
    (forward блока вызывается и при декодинге с KV-кэшем).
    """
    if alpha == 0.0:
        yield
        return

    layer = get_decoder_layers(model)[module_idx]
    delta = alpha * vec  # форма (d_in,)

    def _hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hs = output[0]
            hs = hs + delta.to(dtype=hs.dtype, device=hs.device)
            return (hs,) + tuple(output[1:])
        hs = output
        return hs + delta.to(dtype=hs.dtype, device=hs.device)

    handle = layer.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def generate_batched(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 400,
    temperature: float = 1.0,
    seed: int = 0,
    batch_size: int = 4,
) -> List[str]:
    """Батч-генерация ответов (chat-template + left-padding), один seed на вызов.

    Возвращает список строк — только сгенерированный ассистентский текст
    (input-часть отрезается). Стриминг по батчам ради памяти на T4.

    Стохастичность и независимость сэмплов: temperature=1.0 по умолчанию (было
    0.7 — слишком «узко», сэмплы почти совпадали). Несколько генераций на ОДИН
    промпт получают РАЗНЫЕ ответы за счёт РАЗНОГО seed на каждый вызов: вызывайте
    функцию по разу на сэмпл с seed=base_seed+g (см. scripts/07_steering.py).
    Так N генераций на (промпт, alpha) действительно независимы.
    """
    device = next(model.parameters()).device
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # критично для корректной батч-генерации

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    outs: list[str] = []
    try:
        for lo in range(0, len(prompts), batch_size):
            batch = list(prompts[lo:lo + batch_size])
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in batch
            ]
            enc = tokenizer(
                texts, return_tensors="pt", padding=True, add_special_tokens=False,
            )
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = gen[:, input_ids.shape[1]:]
            outs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = prev_side
    return outs


def pick_random_control_neuron(
    fire_rate: np.ndarray,
    lo: float,
    hi: float,
    exclude: Sequence[int] = (),
    seed: int = 0,
) -> int:
    """Случайный нейрон с fire_rate в [lo, hi], не входящий в exclude.

    Контроль для steering: «обычный» признак похожей частоты, для которого
    причинного эффекта на решение не ожидается.
    """
    pool = np.where((fire_rate >= lo) & (fire_rate <= hi))[0]
    pool = np.array([int(n) for n in pool if int(n) not in set(exclude)], dtype=np.int64)
    if pool.size == 0:
        raise ValueError(
            f"Нет нейронов с fire_rate в [{lo}, {hi}] вне exclude — "
            "ослабьте диапазон steering.random_fire_rate_range."
        )
    rng = np.random.default_rng(seed)
    return int(rng.choice(pool))


# ===========================================================================
# Корректный протокол оценки steering (Задача 3): масштаб alpha от нормы,
# bootstrap-CI, отбор промптов по податливости, формальный критерий «сработало».
# Все функции — чистый numpy (без модели), чтобы их можно было юнит-тестировать.
# ===========================================================================

def alpha_scale_grid(
    hidden: np.ndarray,
    fractions: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> tuple[list[float], float]:
    """Сетка alpha как ДОЛИ от медианы нормы hidden на слое экстракции.

    Прежние фиксированные ±2/±5 были ~3% от типичной нормы ||h|| (≈150), поэтому
    стиринг почти не двигал решение. Масштабируем от данных:

        m = median_i ||h_i||_2                     (по строкам матрицы активаций)
        alphas = sorted({ ±a·m : a ∈ fractions } ∪ {0})

    Это даёт значения на ~2 порядка крупнее прежних. Возвращает (alphas, m).
    """
    h = np.asarray(hidden)
    norms = np.linalg.norm(h.astype(np.float32, copy=False), axis=1)
    m = float(np.median(norms))
    fr = sorted({float(a) for a in fractions if a > 0})
    alphas = sorted([-a * m for a in fr] + [0.0] + [a * m for a in fr])
    return alphas, m


def bootstrap_ci_proportion(
    labels: Sequence[str],
    target: str = "Yes",
    n_boot: int = 1000,
    ci: float = 95.0,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile-bootstrap доверительный интервал доли `target` среди labels.

    Просто и достаточно: ресэмплим индексы с возвращением n_boot раз и берём
    перцентили доли. labels — список меток "Yes"/"No"/"?". Возвращает
    (p, lo, hi); при пустом входе — (nan, nan, nan).
    """
    labs = list(labels)
    n = len(labs)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    ind = np.fromiter((1.0 if l == target else 0.0 for l in labs),
                      dtype=np.float64, count=n)
    p = float(ind.mean())
    rng = np.random.default_rng(seed)
    boot = ind[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    half = (100.0 - ci) / 2.0
    return p, float(np.percentile(boot, half)), float(np.percentile(boot, 100.0 - half))


def select_susceptible_prompts(
    baseline_p_yes: dict,
    *,
    k_each: int = 3,
    down_yes_min: float = 0.55,
    up_yes_max: float = 0.45,
) -> dict:
    """Делит промпты на два набора по податливости, опираясь на бейзлайн (alpha=0).

    baseline_p_yes : {prompt_key: p_yes при alpha=0}.

    push_down (толкаем к No): Yes-смещённый бейзлайн — есть куда двигать ВНИЗ.
        кандидаты p_yes >= down_yes_min; берём top-k по p_yes.
    push_up (толкаем к Yes): высокий ?/No бейзлайн — есть куда двигать ВВЕРХ.
        кандидаты p_yes <= up_yes_max; берём bottom-k по p_yes.

    Если порог отсекает меньше k_each — добираем крайними по ранжировке (наборы
    не должны быть пустыми; об этом пишем в лог на стороне вызова).
    Возвращает {"push_down": [keys], "push_up": [keys]}.
    """
    keys_sorted = sorted(baseline_p_yes, key=lambda k: baseline_p_yes[k])  # по возр. p_yes
    up = [k for k in keys_sorted if baseline_p_yes[k] <= up_yes_max][:k_each]
    if len(up) < k_each:
        up = keys_sorted[:k_each]
    down = [k for k in reversed(keys_sorted) if baseline_p_yes[k] >= down_yes_min][:k_each]
    if len(down) < k_each:
        down = list(reversed(keys_sorted))[:k_each]
    return {"push_down": down, "push_up": up}


def _ci_disjoint(a: tuple, b: tuple) -> bool:
    (alo, ahi), (blo, bhi) = a, b
    return ahi < blo or bhi < alo


def steering_verdict(
    trigger: dict,
    control: dict,
    monotonic_tol: float = 0.05,
) -> tuple[bool, str, dict]:
    """Формальный критерий «steering сработал» для одного триггера vs контроль.

    trigger/control: {"alphas": [...], "p_yes": [...], "ci": [(lo,hi), ...]} —
    ci относится к p_yes. Считаем сработавшим, ТОЛЬКО если ОБА условия:

      (1) p_yes триггера монотонна по alpha (с допуском monotonic_tol на шум);
      (2) на КРАЙНИХ alpha CI триггера НЕ пересекается с CI контроля хотя бы на
          одном конце.

    Иначе — честный вывод «эффект неотличим от нуля/контроля».
    Возвращает (worked, reason, details).
    """
    a = np.asarray(trigger["alphas"], dtype=float)
    order = np.argsort(a)
    p = np.asarray(trigger["p_yes"], dtype=float)[order]
    ci = [trigger["ci"][i] for i in order]

    diffs = np.diff(p)
    overall = float(p[-1] - p[0]) if p.size >= 2 else 0.0
    if overall >= 0:
        monotone = bool(np.all(diffs >= -monotonic_tol))
    else:
        monotone = bool(np.all(diffs <= monotonic_tol))

    ca = np.asarray(control["alphas"], dtype=float)
    cci = [control["ci"][i] for i in np.argsort(ca)]
    sep_low = _ci_disjoint(ci[0], cci[0])
    sep_high = _ci_disjoint(ci[-1], cci[-1])
    separated = sep_low or sep_high

    worked = bool(monotone and separated and abs(overall) > 0)
    reason = (
        f"Δp_yes(min→max)={overall:+.3f}; монотонна={monotone}; "
        f"CI триггера vs контроля разделены: на min-alpha={sep_low}, "
        f"на max-alpha={sep_high} → {'СРАБОТАЛ' if worked else 'неотличим от контроля'}"
    )
    return worked, reason, {
        "overall_delta": overall, "monotone": monotone,
        "sep_low": bool(sep_low), "sep_high": bool(sep_high),
    }
