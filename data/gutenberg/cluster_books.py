#!/usr/bin/env python3
"""Кластеризация книг Project Gutenberg по метаданным → clusters.json.

Пайплайн из 4 стадий с кэшем артефактов (как scripts/01_extract.py: стадия
пропускается, если её артефакт уже на диске; --force / --from пересчитывают):

    load  → outputs/books.parquet
    embed → outputs/embeddings.npy (+ emb_ids.npy)
    cluster → outputs/assignments.parquet
    name  → outputs/clusters.json (+ clusters_detailed.json, clusters_summary.csv)

Запуск на сервере:
    export OPENROUTER_API_KEY=sk-...            # для LLM-имён (иначе --no-llm)
    python cluster_books.py                     # всё из config.yaml, GPU авто

Отладка/проверка без GPU и без датасета:
    python cluster_books.py --data-dir <папка с CSV> --smoke --fake-embed

Переопределение конфига:
    python cluster_books.py --set cluster.k=120 embed.model=BAAI/bge-base-en-v1.5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gutenberg_cluster import cluster as cluster_mod  # noqa: E402
from gutenberg_cluster import embed as embed_mod      # noqa: E402
from gutenberg_cluster import features, load, naming  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cluster_books")

STAGES = ["load", "embed", "cluster", "name"]


# ---------------------------------------------------------------------------
# Конфиг
# ---------------------------------------------------------------------------

def _get(cfg: dict, dotted: str, default=None):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _set(cfg: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def load_config(args) -> dict:
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for item in args.set or []:
        if "=" not in item:
            raise ValueError(f"--set ожидает key=value, получено {item!r}")
        key, raw = item.split("=", 1)
        _set(cfg, key.strip(), yaml.safe_load(raw))  # типизируем значение через yaml

    if args.data_dir is not None:
        _set(cfg, "data.data_dir", args.data_dir)
    if args.no_llm:
        _set(cfg, "name.use_llm", False)
    if args.smoke:
        if _get(cfg, "data.limit") is None:
            _set(cfg, "data.limit", 2000)
        _set(cfg, "cluster.k", min(_get(cfg, "cluster.k", 10), 10))
        _set(cfg, "name.use_llm", False)
        log.info("SMOKE: limit=%s, k=%s, use_llm=False",
                 _get(cfg, "data.limit"), _get(cfg, "cluster.k"))
    return cfg


# ---------------------------------------------------------------------------
# Стадии
# ---------------------------------------------------------------------------

def stage_load(cfg: dict, out: Path) -> None:
    data_dir = _get(cfg, "data.data_dir")
    if data_dir is None:
        import kagglehub
        kaggle_id = _get(cfg, "data.kaggle_id")
        log.info("Скачиваю датасет с Kaggle: %s", kaggle_id)
        data_dir = kagglehub.dataset_download(kaggle_id)
    df = load.load_books(
        Path(data_dir),
        languages=_get(cfg, "data.languages"),
        limit=_get(cfg, "data.limit"),
    )
    df = features.add_doc_text(df)
    df.to_pickle(out / "books.pkl")   # pickle: сохраняет list-колонки, без pyarrow
    log.info("→ %s (%d книг)", (out / "books.pkl").name, len(df))


def stage_embed(cfg: dict, out: Path, fake: bool) -> None:
    df = pd.read_pickle(out / "books.pkl")
    emb = embed_mod.embed_texts(
        df["doc_text"].tolist(),
        model_name=_get(cfg, "embed.model", "all-MiniLM-L6-v2"),
        batch_size=_get(cfg, "embed.batch_size", 256),
        device=_get(cfg, "embed.device", "auto"),
        fake=fake,
    )
    np.save(out / "embeddings.npy", emb)
    np.save(out / "emb_ids.npy", df["id"].to_numpy())
    log.info("→ embeddings.npy %s", emb.shape)


def stage_cluster(cfg: dict, out: Path) -> None:
    emb = np.load(out / "embeddings.npy")
    ids = np.load(out / "emb_ids.npy")
    X = cluster_mod.reduce_dims(
        emb, method=_get(cfg, "cluster.reduce", "none"),
        n_components=_get(cfg, "cluster.umap_components", 10),
        seed=_get(cfg, "cluster.seed", 0),
    )
    labels = cluster_mod.cluster_embeddings(
        X, algo=_get(cfg, "cluster.algo", "kmeans"),
        k=_get(cfg, "cluster.k", 60),
        min_cluster_size=_get(cfg, "cluster.min_cluster_size", 50),
        seed=_get(cfg, "cluster.seed", 0),
    )
    pd.DataFrame({"id": ids, "cluster": labels}).to_csv(out / "assignments.csv", index=False)
    log.info("→ assignments.csv")


def stage_name(cfg: dict, out: Path) -> None:
    df = pd.read_pickle(out / "books.pkl")
    asg = pd.read_csv(out / "assignments.csv")
    # выравниваем метки по порядку строк books.parquet через join по id
    df = df.merge(asg, on="id", how="inner")
    labels = df["cluster"].to_numpy()

    summaries = features.cluster_summaries(
        df, labels,
        top_keywords=_get(cfg, "name.top_keywords", 10),
        examples=_get(cfg, "name.examples", 5),
    )
    names = naming.name_clusters(
        summaries,
        use_llm=_get(cfg, "name.use_llm", True),
        judge_model=_get(cfg, "name.judge_model", "openai/gpt-4o-mini"),
        api_base=_get(cfg, "name.api_base", "https://openrouter.ai/api/v1"),
        concurrency=_get(cfg, "name.concurrency", 8),
    )

    # id по кластерам
    ids_by_cluster: dict[int, list[int]] = {}
    for cid, gid in zip(df["cluster"], df["id"]):
        ids_by_cluster.setdefault(int(cid), []).append(int(gid))

    _check_partition(ids_by_cluster, set(df["id"].astype(int)))

    # финальный clusters.json — по убыванию размера
    clusters = [
        {"cluster_name": names[c], "ids": sorted(ids_by_cluster[c])}
        for c in ids_by_cluster
    ]
    clusters.sort(key=lambda d: len(d["ids"]), reverse=True)
    with open(out / "clusters.json", "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)

    # подробный json + csv для глазами-проверки
    by_cluster = {s["cluster"]: s for s in summaries}
    detailed = [
        {"cluster_name": names[c], "size": by_cluster[c]["size"],
         "top_keywords": by_cluster[c]["top_keywords"],
         "top_shelves": by_cluster[c]["top_shelves"],
         "top_subjects": by_cluster[c]["top_subjects"],
         "top_locc": by_cluster[c]["top_locc"],
         "example_titles": by_cluster[c]["example_titles"]}
        for c in sorted(by_cluster, key=lambda x: by_cluster[x]["size"], reverse=True)
    ]
    with open(out / "clusters_detailed.json", "w", encoding="utf-8") as f:
        json.dump(detailed, f, ensure_ascii=False, indent=2)

    pd.DataFrame([
        {"cluster_name": d["cluster_name"], "size": d["size"],
         "top_shelves": "; ".join(d["top_shelves"]),
         "top_subjects": "; ".join(d["top_subjects"]),
         "top_keywords": ", ".join(d["top_keywords"])}
        for d in detailed
    ]).to_csv(out / "clusters_summary.csv", index=False)

    log.info("→ clusters.json (%d кластеров), clusters_detailed.json, clusters_summary.csv",
             len(clusters))
    log.info("Топ-10 кластеров по размеру:")
    for d in detailed[:10]:
        log.info("  %5d  %s", d["size"], d["cluster_name"])


def _check_partition(ids_by_cluster: dict[int, list[int]], all_ids: set[int]) -> None:
    """Инвариант жёсткого разбиения: каждый id ровно в одном непустом кластере."""
    flat: list[int] = [i for ids in ids_by_cluster.values() for i in ids]
    assert all(ids for ids in ids_by_cluster.values()), "есть пустой кластер"
    assert len(flat) == len(all_ids), f"сумма размеров {len(flat)} != числа книг {len(all_ids)}"
    assert set(flat) == all_ids, "множество id в кластерах != входному множеству"
    assert len(flat) == len(set(flat)), "id повторяется в нескольких кластерах"
    log.info("Инвариант разбиения ок: %d книг, %d кластеров, без пересечений",
             len(all_ids), len(ids_by_cluster))


# ---------------------------------------------------------------------------
# Оркестрация стадий с кэшем
# ---------------------------------------------------------------------------

_STAGE_OUTPUT = {
    "load": "books.pkl",
    "embed": "embeddings.npy",
    "cluster": "assignments.csv",
    "name": "clusters.json",
}


def run(args) -> None:
    cfg = load_config(args)
    out = (SCRIPT_DIR / _get(cfg, "paths.outputs_dir", "outputs")).resolve()
    out.mkdir(parents=True, exist_ok=True)

    from_idx = STAGES.index(args.from_stage) if args.from_stage else None
    must_run = False
    for idx, stage in enumerate(STAGES):
        output = out / _STAGE_OUTPUT[stage]
        forced = args.force or (from_idx is not None and idx >= from_idx)
        run_it = must_run or forced or not output.exists()
        if not run_it:
            log.info("[%s] пропуск — %s уже есть (--force/--from пересчитают)",
                     stage, output.name)
            continue
        must_run = True  # если пересчитали стадию — все ниже тоже пересчитываем
        log.info("[%s] запуск", stage)
        if stage == "load":
            stage_load(cfg, out)
        elif stage == "embed":
            stage_embed(cfg, out, fake=args.fake_embed)
        elif stage == "cluster":
            stage_cluster(cfg, out)
        elif stage == "name":
            stage_name(cfg, out)

    log.info("ГОТОВО. Результат: %s", out / "clusters.json")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", "-c", default=str(SCRIPT_DIR / "config.yaml"))
    p.add_argument("--data-dir", default=None,
                   help="папка/CSV датасета; по умолчанию из config (null → Kaggle)")
    p.add_argument("--from", dest="from_stage", choices=STAGES, default=None,
                   help="пересчитать с этой стадии и ниже")
    p.add_argument("--force", "-f", action="store_true", help="пересчитать все стадии")
    p.add_argument("--set", nargs="*", default=[],
                   help="переопределения конфига: dotted.key=value (через yaml-типизацию)")
    p.add_argument("--smoke", action="store_true",
                   help="быстрый прогон: limit=2000, k=10, без LLM")
    p.add_argument("--no-llm", action="store_true", help="имена из ключевых слов, без LLM")
    p.add_argument("--fake-embed", action="store_true",
                   help="хеш-эмбеддинги без torch/модели (только для проверки пайплайна)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
