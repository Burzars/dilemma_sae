"""Кластеризация эмбеддингов в жёсткое разбиение (каждая книга ровно в 1 кластере).

kmeans (дефолт) — точное число кластеров k, все книги назначены.
hdbscan — авто-число; шум (-1) до-назначается ближайшему центроиду, чтобы
сохранить полное разбиение. Метки на выходе всегда 0..K-1 без дыр.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def reduce_dims(X: np.ndarray, method: str = "none", n_components: int = 10, seed: int = 0) -> np.ndarray:
    """Опциональное снижение размерности перед кластеризацией."""
    if method in (None, "none"):
        return X
    if method == "umap":
        import umap  # umap-learn
        log.info("UMAP: %d → %d измерений", X.shape[1], n_components)
        reducer = umap.UMAP(
            n_components=n_components, metric="cosine", random_state=seed, verbose=False,
        )
        return np.asarray(reducer.fit_transform(X), dtype=np.float32)
    raise ValueError(f"Неизвестный reduce: {method!r} (ожидалось none|umap)")


def _relabel_contiguous(labels: np.ndarray) -> np.ndarray:
    """Переименовывает метки в 0..K-1 без пропусков, сохраняя группировку."""
    uniq = sorted(set(int(v) for v in labels))
    remap = {old: new for new, old in enumerate(uniq)}
    return np.array([remap[int(v)] for v in labels], dtype=int)


def _assign_noise_to_nearest(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """До-назначает шумовые точки (-1) ближайшему центроиду непустых кластеров."""
    noise = labels == -1
    if not noise.any():
        return labels
    valid = np.unique(labels[labels >= 0])
    if len(valid) == 0:                       # всё шум — один кластер
        return np.zeros_like(labels)
    centroids = np.vstack([X[labels == c].mean(axis=0) for c in valid])
    # ближайший центроид по евклиду (эмбеддинги нормированы → ~косинус)
    d = np.linalg.norm(X[noise][:, None, :] - centroids[None, :, :], axis=2)
    labels = labels.copy()
    labels[noise] = valid[d.argmin(axis=1)]
    log.info("HDBSCAN: до-назначено %d шумовых точек ближайшему кластеру", int(noise.sum()))
    return labels


def cluster_embeddings(
    X: np.ndarray,
    algo: str = "kmeans",
    k: int = 60,
    min_cluster_size: int = 50,
    seed: int = 0,
) -> np.ndarray:
    """Возвращает метки кластеров (0..K-1), по строке на книгу."""
    n = X.shape[0]
    if algo == "kmeans":
        from sklearn.cluster import MiniBatchKMeans
        k_eff = min(k, n)
        if k_eff < k:
            log.warning("k=%d > числа книг %d, использую k=%d", k, n, k_eff)
        km = MiniBatchKMeans(
            n_clusters=k_eff, random_state=seed, n_init=3, batch_size=4096,
        )
        labels = km.fit_predict(X)
    elif algo == "hdbscan":
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
        labels = clusterer.fit_predict(X)
        n_clusters = len(set(int(v) for v in labels) - {-1})
        log.info("HDBSCAN: %d кластеров, %d шумовых точек", n_clusters, int((labels == -1).sum()))
        labels = _assign_noise_to_nearest(X, labels)
    else:
        raise ValueError(f"Неизвестный algo: {algo!r} (ожидалось kmeans|hdbscan)")

    labels = _relabel_contiguous(np.asarray(labels))
    log.info("Итог кластеризации: %d кластеров на %d книг", len(set(labels.tolist())), n)
    return labels
