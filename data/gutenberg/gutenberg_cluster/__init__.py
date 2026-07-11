"""Кластеризация книг Project Gutenberg по метаданным.

Стадии пайплайна (см. cluster_books.py):
    load  → books.parquet        (src: load.py)
    embed → embeddings.npy        (src: embed.py)
    cluster → assignments.parquet (src: cluster.py)
    name  → clusters.json         (src: naming.py + features.py)
"""

from __future__ import annotations

__all__ = ["load", "features", "embed", "cluster", "naming"]
