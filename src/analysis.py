"""Анализ feature-векторов SAE.

Содержит:
  - encode_all                  : прогон всех активаций через encoder SAE
  - neuron_statistics           : fire_rate, mean_active, max_active, n_fires
  - select_top_neurons          : top-K «интересных» нейронов
  - top_contexts_for_neuron     : top-K контекстов с дедупликацией по sample_idx
  - label_contrast              : Yes/No контраст token-level (Cohen's d)
  - sample_level_contrast       : Yes/No контраст sample-level (одна точка на пример)

Контраст делается ДВАЖДЫ — token-level и sample-level — и берётся
пересечение топов по обоим. Token-level переоценивает значимость
(позиции внутри одного ответа скоррелированы), sample-level имеет
меньше статистики, но честнее.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import List, Sequence

import numpy as np
import torch

from .data import ReasoningSample
from .extract import _build_chat_text

log = logging.getLogger(__name__)


@torch.no_grad()
def encode_all(sae, hidden: np.ndarray, batch_size: int = 4096, device: str = "cuda") -> np.ndarray:
    """Прогоняет все hidden states через encoder SAE → разреженные features.

    Returns
    -------
    np.ndarray (N, d_hidden) float16
    """
    sae.eval()
    X = torch.from_numpy(np.ascontiguousarray(hidden, dtype=np.float32))
    n = X.shape[0]
    # Пишем сразу в преаллоцированный массив, без списка кусков + concatenate
    # (тот удваивал пиковую память — критично для широких SAE и больших корпусов).
    out = np.empty((n, int(sae.d_hidden)), dtype=np.float16)
    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size].to(device)
        f = sae.encode(xb)
        out[i:i + xb.shape[0]] = f.cpu().numpy().astype(np.float16)
    return out


@torch.no_grad()
def neuron_statistics_streaming(
    sae, hidden: np.ndarray, batch_size: int = 8192, device: str = "cuda",
) -> dict:
    """neuron_statistics БЕЗ материализации матрицы features (N, d_hidden).

    Делает один проход по батчам на GPU и аккумулирует только (d_hidden,)-векторы
    статистик. Для больших корпусов плотная (N, d_hidden) занимает десятки ГБ —
    этот вариант держит в памяти лишь сами статистики. Результат идентичен
    neuron_statistics(encode_all(...)).
    """
    sae.eval()
    X = torch.from_numpy(np.ascontiguousarray(hidden, dtype=np.float32))
    n = X.shape[0]
    dh = int(sae.d_hidden)
    n_fires = torch.zeros(dh, dtype=torch.float64, device=device)
    sum_active = torch.zeros(dh, dtype=torch.float64, device=device)
    max_active = torch.zeros(dh, dtype=torch.float32, device=device)
    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size].to(device)
        f = sae.encode(xb)
        active = f > 0
        n_fires += active.sum(dim=0).double()
        sum_active += (f * active).sum(dim=0).double()
        max_active = torch.maximum(max_active, f.max(dim=0).values)
    nf = n_fires.cpu().numpy()
    sa = sum_active.cpu().numpy()
    return {
        "fire_rate":   (nf / max(n, 1)).astype(np.float32),
        "mean_active": np.where(nf > 0, sa / np.clip(nf, 1, None), 0.0).astype(np.float32),
        "max_active":  max_active.cpu().numpy().astype(np.float32),
        "n_fires":     nf.astype(np.int64),
    }


@torch.no_grad()
def recon_ev(sae, hidden: np.ndarray, batch_size: int = 4096, device: str = "cuda") -> dict:
    """Explained variance (EV) реконструкции SAE на матрице активаций (N, d_in).

    EV = 1 − MSE / Var(X), где (как в ТЗ):
        MSE    = ((X − X̂) ** 2).mean()        # среднее по всем (точка, размерность)
        Var(X) = X.var(dim=0).mean()           # средняя по 2048 размерностям дисперсия

    Считается ПОТОКОВО по батчам — не материализует матрицу features
    (N, d_hidden), поэтому безопасно по памяти даже для широких SAE.
    Аккумуляторы в float64 ради устойчивости.

    Returns
    -------
    dict: {"ev": float, "mse": float, "var": float, "n": int}
    """
    sae.eval()
    X = torch.from_numpy(hidden.astype(np.float32))
    n, d = X.shape
    sse = torch.zeros((), dtype=torch.float64, device=device)
    sum_x = torch.zeros(d, dtype=torch.float64, device=device)
    sum_x2 = torch.zeros(d, dtype=torch.float64, device=device)
    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size].to(device)
        x_hat = sae.decode(sae.encode(xb))
        sse += (x_hat - xb).double().pow(2).sum()
        xbd = xb.double()
        sum_x += xbd.sum(0)
        sum_x2 += xbd.pow(2).sum(0)
    mse = (sse / (n * d)).item()
    mean = sum_x / n
    var_x = (sum_x2 / n - mean.pow(2)).mean().item()   # population variance (ddof=0)
    ev = (1.0 - mse / var_x) if var_x > 0 else float("nan")
    return {"ev": ev, "mse": mse, "var": var_x, "n": int(n)}


def neuron_statistics(features: np.ndarray) -> dict:
    """Per-neuron статистика по матрице features (N, d_hidden).

    Returns dict с массивами длины d_hidden:
      fire_rate    : доля позиций, где нейрон активен (>0)
      mean_active  : среднее значение нейрона в позициях, где он активен
      max_active   : максимальное значение
      n_fires      : сколько раз нейрон активировался
    """
    f = features.astype(np.float32)
    active = f > 0
    return {
        "fire_rate":   active.mean(axis=0),
        "mean_active": np.where(
            active.any(axis=0),
            (f * active).sum(axis=0) / active.sum(axis=0).clip(min=1),
            0.0,
        ),
        "max_active":  f.max(axis=0),
        "n_fires":     active.sum(axis=0).astype(np.int64),
    }


def select_top_neurons(
    stats: dict,
    k: int = 50,
    min_fires: int = 20,
    max_fire_rate: float = 0.5,
) -> np.ndarray:
    """Возвращает индексы top-K «интересных» нейронов.

    Фильтрация:
      - n_fires >= min_fires        (отсекаем шум: мало наблюдений)
      - fire_rate < max_fire_rate   (отсекаем BOS-подобные «всегда-активные»)

    Ранжирование: mean_active * log(1 + n_fires) — компромисс между
    силой активации и количеством наблюдений.
    """
    eligible_mask = (stats["n_fires"] >= min_fires) & (stats["fire_rate"] < max_fire_rate)
    eligible_idx = np.where(eligible_mask)[0]
    if eligible_idx.size == 0:
        return np.array([], dtype=np.int64)
    score = stats["mean_active"][eligible_idx] * np.log1p(stats["n_fires"][eligible_idx])
    order = np.argsort(-score)[:k]
    return eligible_idx[order]


def top_contexts_for_neuron(
    neuron_id: int,
    features: np.ndarray,
    sample_idx: np.ndarray,
    token_pos: np.ndarray,
    samples: Sequence[ReasoningSample],
    tokenizer,
    top_k: int = 10,
    window: int = 30,
) -> List[dict]:
    """Top-K контекстов для одного нейрона с дедупликацией по sample_idx.

    Внимание: ИЗВЕСТНАЯ ПРОБЛЕМА (см. HANDOVER.md, открытая проблема #1).
    Дедупликация по sample_idx показывает только ПЕРВЫЙ хит на каждый
    sample_idx, что может «усреднять» разные нейроны к одинаковому
    топу, если они активны на одних и тех же насыщенных примерах,
    но в разных токенных позициях. Это известно, но менять решено
    не сейчас (требуется фикс в visualize/notebook слое — показывать
    также token_pos).
    """
    activ = features[:, neuron_id].astype(np.float32)
    order = np.argsort(-activ)

    out: list[dict] = []
    seen: set[int] = set()
    for j in order:
        if activ[j] <= 0:
            break
        si = int(sample_idx[j])
        if si in seen:
            continue
        seen.add(si)

        s = samples[si]
        chat_text, _ = _build_chat_text(s.prompt, s.answer, tokenizer)
        ids = tokenizer(chat_text, add_special_tokens=False)["input_ids"]
        pos = int(token_pos[j])
        lo, hi = max(0, pos - window), min(len(ids), pos + 5)
        snippet = tokenizer.decode(ids[lo:hi], skip_special_tokens=True)

        out.append({
            "activation": float(activ[j]),
            "label":      s.label,
            "file_idx":   s.file_idx,
            "token_pos":  pos,
            "snippet":    snippet,
        })
        if len(out) >= top_k:
            break
    return out


def label_contrast(
    features: np.ndarray,
    labels: np.ndarray,
    pos_label: str = "Yes",
    neg_label: str = "No",
    min_fires: int = 20,
) -> dict:
    """Token-level контраст Yes vs No: Cohen's d на уровне отдельных позиций.

    Переоценивает значимость (позиции из одного ответа скоррелированы),
    но даёт богатую статистику. Используется вместе с sample_level_contrast.
    """
    f = features.astype(np.float32)
    pos_mask = labels == pos_label
    neg_mask = labels == neg_label
    fp, fn = f[pos_mask], f[neg_mask]

    mp = fp.mean(axis=0) if fp.size else np.zeros(f.shape[1])
    mn = fn.mean(axis=0) if fn.size else np.zeros(f.shape[1])
    sp = fp.std(axis=0) + 1e-6
    sn = fn.std(axis=0) + 1e-6
    pooled = np.sqrt(0.5 * (sp ** 2 + sn ** 2))
    cohens_d = (mp - mn) / (pooled + 1e-6)

    fires = (f > 0).sum(axis=0)
    eligible = fires >= min_fires
    cohens_d = np.where(eligible, cohens_d, 0.0)

    return {
        "mean_pos": mp,
        "mean_neg": mn,
        "diff":     mp - mn,
        "cohens_d": cohens_d,
        "fires":    fires,
    }


def sample_level_contrast(
    features: np.ndarray,
    sample_idx: np.ndarray,
    samples: Sequence[ReasoningSample],
    pos_label: str = "Yes",
    neg_label: str = "No",
) -> dict:
    """Sample-level контраст: одна точка на пример (усреднение внутри).

    Меньше статистики, чем token-level, но честнее (нет скоррелированных
    повторов внутри одного reasoning). Берётся пересечение топов
    обоих контрастов как защита от ложных срабатываний.
    """
    f = features.astype(np.float32)
    n_neurons = f.shape[1]
    by_sample = defaultdict(list)
    for i, si in enumerate(sample_idx):
        by_sample[int(si)].append(f[i])
    sample_ids = sorted(by_sample.keys())
    sample_feats = np.stack(
        [np.mean(np.stack(by_sample[si]), axis=0) for si in sample_ids]
    )
    sample_labels = np.array([samples[si].label for si in sample_ids])

    pos_mask = sample_labels == pos_label
    neg_mask = sample_labels == neg_label
    fp, fn = sample_feats[pos_mask], sample_feats[neg_mask]

    if fp.size == 0 or fn.size == 0:
        return {
            "diff":     np.zeros(n_neurons),
            "cohens_d": np.zeros(n_neurons),
            "n_pos":    int(pos_mask.sum()),
            "n_neg":    int(neg_mask.sum()),
        }

    mp, mn = fp.mean(0), fn.mean(0)
    sp, sn = fp.std(0) + 1e-6, fn.std(0) + 1e-6
    pooled = np.sqrt(0.5 * (sp ** 2 + sn ** 2))
    return {
        "diff":     mp - mn,
        "cohens_d": (mp - mn) / (pooled + 1e-6),
        "n_pos":    int(pos_mask.sum()),
        "n_neg":    int(neg_mask.sum()),
    }


def find_contrast_neurons(
    tok_contrast: dict,
    smp_contrast: dict,
    top_k: int = 10,
    pool_factor: int = 3,
) -> tuple[list[int], list[int]]:
    """Возвращает (yes_neurons, no_neurons): пересечения топов token / sample.

    Логика (из ноутбука):
      1. Берём top_k * pool_factor лучших по каждому контрасту.
      2. Пересекаем — берём только нейроны, попавшие в обоих.
      3. Сортируем по sample-level d, обрезаем до top_k.
    """
    pool = top_k * pool_factor

    # Yes: ищем нейроны с НАИБОЛЬШИМ положительным d
    top_yes_tok = set(np.argsort(-tok_contrast["cohens_d"])[:pool])
    top_yes_smp = set(np.argsort(-smp_contrast["cohens_d"])[:pool])
    yes_neurons = list(top_yes_tok & top_yes_smp)
    yes_neurons = sorted(
        yes_neurons,
        key=lambda n: -smp_contrast["cohens_d"][n],
    )[:top_k]

    # No: ищем нейроны с НАИБОЛЬШИМ отрицательным d (т.е. argsort без минуса)
    top_no_tok = set(np.argsort(tok_contrast["cohens_d"])[:pool])
    top_no_smp = set(np.argsort(smp_contrast["cohens_d"])[:pool])
    no_neurons = list(top_no_tok & top_no_smp)
    no_neurons = sorted(
        no_neurons,
        key=lambda n: smp_contrast["cohens_d"][n],
    )[:top_k]

    log.info(
        "Найдено: %d Yes-нейронов, %d No-нейронов (пересечение token & sample, pool=%d)",
        len(yes_neurons), len(no_neurons), pool,
    )
    return yes_neurons, no_neurons


def contrast_feature_slice(
    features: np.ndarray,
    labels: np.ndarray,
    sample_idx: np.ndarray,
    samples: Sequence[ReasoningSample],
    *,
    slice: str = "balanced",
    balanced_dir=None,
    balanced_seed: int = 0,
):
    """(features, labels, sample_idx) для Yes/No-контраста на нужном срезе.

    SAE/features/stats считаются на FULL, но сам КОНТРАСТ — на BALANCED, где
    метки используются и шум редкого класса вреден. Balanced-срез получаем
    маской по позициям того же full-pack (без повторной экстракции).

    slice="full" → данные как есть (старое поведение).
    slice="balanced" → маска из src/build_datasets.balanced_sample_mask.
    """
    if slice == "full":
        return features, labels, sample_idx
    if slice != "balanced":
        raise ValueError(f"contrast.slice должен быть 'full' или 'balanced', а не {slice!r}")

    from .build_datasets import balanced_sample_mask
    mask, info = balanced_sample_mask(
        samples, sample_idx, balanced_dir=balanced_dir, seed=balanced_seed,
    )
    n = int(mask.sum())
    log.info(
        "Контраст на BALANCED: позиций %d/%d, ответов %d (до баланса %s, источник=%s)",
        n, len(mask), info["n_answers_kept"], info["counts_before"], info["source"],
    )
    if n == 0:
        raise ValueError(
            "Balanced-срез пуст. Проверьте contrast.balanced_dir / редкий класс "
            "(нужны примеры всех классов Yes/No/?)."
        )
    return features[mask], np.asarray(labels)[mask], np.asarray(sample_idx)[mask]


def balanced_contrast_arrays(
    hidden: np.ndarray,
    labels: np.ndarray,
    sample_idx: np.ndarray,
    token_pos: np.ndarray,
    samples: Sequence[ReasoningSample],
    *,
    slice: str = "balanced",
    balanced_dir=None,
    balanced_seed: int = 0,
):
    """Маскирует АКТИВАЦИИ до balanced-среза ПЕРЕД кодированием.

    Возвращает (hidden, labels, sample_idx, token_pos) только для balanced-позиций
    — их потом кодируют в features. Так мы НЕ материализуем плотную матрицу
    (N_full, d_hidden) (десятки ГБ): кодируется лишь срез (~равные Yes/No/?).

    slice="full" → массивы как есть (внимание: кодирование всего корпуса в
    features может занять десятки ГБ ОЗУ).
    """
    if slice == "full":
        return hidden, labels, sample_idx, token_pos
    if slice != "balanced":
        raise ValueError(f"contrast.slice должен быть 'full'|'balanced', а не {slice!r}")

    from .build_datasets import balanced_sample_mask
    mask, info = balanced_sample_mask(
        samples, sample_idx, balanced_dir=balanced_dir, seed=balanced_seed,
    )
    n = int(mask.sum())
    log.info(
        "Контраст на BALANCED: позиций %d/%d, ответов %d (до баланса %s, источник=%s)",
        n, len(mask), info["n_answers_kept"], info["counts_before"], info["source"],
    )
    if n == 0:
        raise ValueError(
            "Balanced-срез пуст. Проверьте contrast.balanced_dir / наличие всех "
            "классов Yes/No/?."
        )
    return (hidden[mask], np.asarray(labels)[mask],
            np.asarray(sample_idx)[mask], np.asarray(token_pos)[mask])


def build_report(
    neuron_ids: Sequence[int],
    kind: str,
    features: np.ndarray,
    sample_idx: np.ndarray,
    token_pos: np.ndarray,
    samples: Sequence[ReasoningSample],
    tokenizer,
    stats: dict,
    tok_contrast: dict,
    smp_contrast: dict,
    top_k: int = 15,
    window: int = 40,
) -> List[dict]:
    """Собирает отчёт по списку нейронов (для скармливания LLM-судье)."""
    report = []
    for nid in neuron_ids:
        contexts = top_contexts_for_neuron(
            nid, features, sample_idx, token_pos, samples, tokenizer,
            top_k=top_k, window=window,
        )
        report.append({
            "neuron_id": int(nid),
            "kind":      kind,
            "stats": {
                "fire_rate":       float(stats["fire_rate"][nid]),
                "mean_active":     float(stats["mean_active"][nid]),
                "n_fires":         int(stats["n_fires"][nid]),
                "cohens_d_token":  float(tok_contrast["cohens_d"][nid]),
                "cohens_d_sample": float(smp_contrast["cohens_d"][nid]),
            },
            "top_contexts": contexts,
        })
    return report
