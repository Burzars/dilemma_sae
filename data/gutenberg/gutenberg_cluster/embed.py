"""Эмбеддинги метаданных через sentence-transformers.

fake=True — детерминированные хеш-эмбеддинги без torch/скачивания модели: нужны
только для быстрой проверки пайплайна (load→cluster→name) на CPU/CI. На сервере
используется настоящая модель.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np

log = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    if device and device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 — torch может быть не установлен при fake
        return "cpu"


def _fake_embed(texts: list[str], dim: int = 256) -> np.ndarray:
    """Детерминированные хеш-эмбеддинги (bag-of-words по токенам), L2-нормированные."""
    import re

    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for tok in re.findall(r"[a-z]{2,}", str(text).lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            out[i, h % dim] += 1.0 if (h >> 8) & 1 else -1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(norms, 1e-9, None)


def embed_texts(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    device: str = "auto",
    fake: bool = False,
) -> np.ndarray:
    """Возвращает матрицу эмбеддингов (n_texts × d), float32, L2-нормированную."""
    if fake:
        log.warning("fake-embed: хеш-эмбеддинги (не для продакшена, только проверка пайплайна)")
        return _fake_embed(texts)

    from sentence_transformers import SentenceTransformer

    dev = _resolve_device(device)
    log.info("Загружаю модель %s на %s", model_name, dev)
    model = SentenceTransformer(model_name, device=dev)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,     # косинусная близость → евклид после нормировки
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return np.asarray(emb, dtype=np.float32)
