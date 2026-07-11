"""Признаки для эмбеддинга и сводки кластеров для наименования.

- doc_text: текст на книгу для sentence-transformers (БЕЗ опаковых LoCC-кодов —
  они бессмысленны для энкодера; коды используем отдельно при наименовании).
- c-TF-IDF (как в BERTopic): характерные слова каждого кластера — на них и просим
  LLM придумать имя, они же служат офлайн-фолбэком.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd

try:
    from lc_classes import summarize_locc
except ImportError:  # pragma: no cover
    from ..lc_classes import summarize_locc  # type: ignore

log = logging.getLogger(__name__)


def build_doc_text(title: str, subjects: list[str], shelves: list[str]) -> str:
    """Текст книги для эмбеддинга: заголовок + LCSH-темы + полки (без LoCC-кодов)."""
    parts: list[str] = []
    title = (title or "").strip()
    if title:
        parts.append(title)
    if subjects:
        parts.append("; ".join(subjects))
    if shelves:
        parts.append("Shelves: " + "; ".join(shelves))
    return ". ".join(parts) if parts else (title or "")


def add_doc_text(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет колонку doc_text (in place-совместимо, возвращает тот же df)."""
    df["doc_text"] = [
        build_doc_text(t, s, sh)
        for t, s, sh in zip(df["title"], df["subjects"], df["shelves"])
    ]
    n_empty = int((df["doc_text"].str.len() == 0).sum())
    if n_empty:
        log.warning("Пустой doc_text у %d книг (нет ни заголовка, ни тем)", n_empty)
    return df


def _ctfidf_top_terms(
    cluster_docs: list[str], labels_order: list[int], top_n: int
) -> dict[int, list[str]]:
    """c-TF-IDF: топ-N характерных слов на кластер.

    cluster_docs[i] — конкатенированные темы+полки всех книг кластера labels_order[i].
    """
    from sklearn.feature_extraction.text import CountVectorizer

    non_empty = [d for d in cluster_docs if d.strip()]
    if not non_empty:
        return {c: [] for c in labels_order}

    vec = CountVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=20000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    try:
        X = vec.fit_transform(cluster_docs)          # (n_clusters, n_terms)
    except ValueError:                               # пустой словарь
        return {c: [] for c in labels_order}
    terms = np.array(vec.get_feature_names_out())

    counts = X.toarray().astype(float)               # кластеров немного (десятки–сотни)
    tf = counts / np.clip(counts.sum(axis=1, keepdims=True), 1.0, None)
    f_t = counts.sum(axis=0)                          # частота термина по всем кластерам
    avg_words = counts.sum() / max(len(cluster_docs), 1)
    idf = np.log(1.0 + avg_words / np.clip(f_t, 1.0, None))
    ctfidf = tf * idf

    out: dict[int, list[str]] = {}
    for i, cluster_id in enumerate(labels_order):
        order = np.argsort(ctfidf[i])[::-1][:top_n]
        out[cluster_id] = [terms[j] for j in order if ctfidf[i, j] > 0]
    return out


def cluster_summaries(
    df: pd.DataFrame,
    labels: np.ndarray,
    top_keywords: int = 10,
    examples: int = 5,
) -> list[dict]:
    """Сводка по каждому кластеру для наименования.

    Возвращает список dict (по возрастанию cluster id): cluster, size,
    top_keywords, top_shelves, top_subjects, top_locc, example_titles.
    """
    labels = np.asarray(labels)
    cluster_ids = sorted(int(c) for c in np.unique(labels))

    # корпус для c-TF-IDF: темы + полки каждой книги, склеенные по кластеру
    per_book_text = [
        " ; ".join(list(s) + list(sh))
        for s, sh in zip(df["subjects"], df["shelves"])
    ]
    per_book_text = np.array(per_book_text, dtype=object)
    cluster_docs = [
        " ; ".join(per_book_text[labels == c]) for c in cluster_ids
    ]
    keywords = _ctfidf_top_terms(cluster_docs, cluster_ids, top_keywords)

    summaries: list[dict] = []
    for c in cluster_ids:
        mask = labels == c
        sub = df[mask]

        shelf_counter: Counter[str] = Counter()
        subject_counter: Counter[str] = Counter()
        locc_all: list[str] = []
        for shelves, subjects, locc in zip(sub["shelves"], sub["subjects"], sub["locc"]):
            shelf_counter.update(shelves)
            subject_counter.update(subjects)
            locc_all.extend(locc)

        titles = [t for t in sub["title"] if isinstance(t, str) and t.strip()]

        summaries.append({
            "cluster": int(c),
            "size": int(mask.sum()),
            "top_keywords": keywords.get(c, []),
            "top_shelves": [s for s, _ in shelf_counter.most_common(5)],
            "top_subjects": [s for s, _ in subject_counter.most_common(5)],
            "top_locc": summarize_locc(locc_all, top=3),
            "example_titles": titles[:examples],
        })
    return summaries
