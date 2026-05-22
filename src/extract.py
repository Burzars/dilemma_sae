"""Извлечение hidden states LLM на reasoning-ответах.

Ключевая идея (FAST §3.1, sequence-level processing): каждый prompt+answer
заворачивается в chat-template, проходит через модель за один forward,
а hidden states слоя `layer_idx` собираются на множестве позиций токенов
(stride=16) внутри области ответа ассистента.

Это эксплуатирует свойство каузальной маски: hidden state в позиции t
зависит только от токенов [0..t], так что один forward = все «префиксы»
бесплатно.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .data import ReasoningSample

log = logging.getLogger(__name__)


@dataclass
class ActivationConfig:
    """Параметры извлечения активаций.

    Внимание: stride=16 — это шаг по ПОЗИЦИЯМ ТОКЕНОВ внутри ответа,
    а не по примерам. min_position=8 пропускает первые токены (системный
    шаблон). only_assistant_positions=True означает, что собираются только
    позиции с user_len и далее (т.е. только сам reasoning ассистента).
    """
    layer_idx: int = -5
    max_length: int = 1024
    stride: int = 16
    min_position: int = 8
    batch_size: int = 4
    only_assistant_positions: bool = True


def _build_chat_text(prompt: str, answer: str, tokenizer):
    """prompt+answer в формате chat template + длина user-части в токенах.

    Возвращает (full_text, user_len), где user_len — сколько первых
    токенов в full_text относятся к user-prompt (включая generation
    prompt). Всё остальное — ответ ассистента.
    """
    user_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = user_text + answer
    user_len = len(tokenizer(user_text, add_special_tokens=False)["input_ids"])
    return full_text, user_len


@torch.no_grad()
def extract_activations(
    samples: Sequence[ReasoningSample],
    model,
    tokenizer,
    cfg: ActivationConfig,
    verbose: bool = True,
) -> dict:
    """Прогон reasoning-текстов через LLM и сбор hidden states.

    Returns
    -------
    dict:
      hidden     : np.ndarray (N_positions, d_model) float16
      sample_idx : np.ndarray (N_positions,)         int64  — индекс примера
      token_pos  : np.ndarray (N_positions,)         int64  — позиция токена в chat-tekste
      label      : np.ndarray (N_positions,)         str    — метка примера, продублированная
    """
    device = next(model.parameters()).device
    save_dtype = np.float16
    tokenizer.padding_side = "right"

    # Заранее соберём чат-тексты и границы user-части
    chat_texts, user_lens = [], []
    for s in samples:
        t, u_len = _build_chat_text(s.prompt, s.answer, tokenizer)
        chat_texts.append(t)
        user_lens.append(u_len)

    all_hidden, all_sample_idx, all_token_pos, all_labels = [], [], [], []

    n_batches = math.ceil(len(samples) / cfg.batch_size)
    for bi in range(n_batches):
        lo = bi * cfg.batch_size
        hi = min(lo + cfg.batch_size, len(samples))
        batch_texts = chat_texts[lo:hi]
        batch_user_lens = user_lens[lo:hi]

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        layer_h = outputs.hidden_states[cfg.layer_idx]   # (B, T, d)

        for k in range(layer_h.shape[0]):
            T_real = int(attn_mask[k].sum().item())
            user_len = batch_user_lens[k]
            start = max(
                cfg.min_position,
                user_len if cfg.only_assistant_positions else 0,
            )
            positions = np.arange(start, T_real, cfg.stride, dtype=np.int64)
            if positions.size == 0:
                continue
            h = layer_h[k, positions, :].float().cpu().numpy().astype(save_dtype)

            global_idx = lo + k
            all_hidden.append(h)
            all_sample_idx.append(np.full(positions.size, global_idx, dtype=np.int64))
            all_token_pos.append(positions)
            all_labels.extend([samples[global_idx].label] * positions.size)

        if verbose and (bi % 5 == 0 or bi == n_batches - 1):
            done = sum(a.shape[0] for a in all_hidden)
            log.info("  батч %d/%d, всего активаций: %d", bi + 1, n_batches, done)

        del outputs, layer_h, input_ids, attn_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pack = {
        "hidden":     np.concatenate(all_hidden, axis=0),
        "sample_idx": np.concatenate(all_sample_idx, axis=0),
        "token_pos":  np.concatenate(all_token_pos, axis=0),
        "label":      np.array(all_labels),
    }
    log.info(
        "Извлечение завершено: %d позиций, hidden.shape=%s, метки: %s",
        pack["hidden"].shape[0], pack["hidden"].shape, Counter(pack["label"].tolist()),
    )
    return pack
