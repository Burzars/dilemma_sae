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
    temperature: float = 0.7,
    seed: int = 0,
    batch_size: int = 4,
) -> List[str]:
    """Батч-генерация ответов (chat-template + left-padding), seed на весь вызов.

    Возвращает список строк — только сгенерированный ассистентский текст
    (input-часть отрезается). Стиминг по батчам ради памяти на T4.
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
