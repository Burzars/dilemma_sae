"""Наименование кластеров.

Офлайн: имя из самых характерных полок/c-TF-IDF слов.
LLM: короткое английское имя от gpt-4o-mini через OpenRouter — зеркалит паттерн
src/interpret.py (тот же OPENROUTER_API_KEY, base_url, async + Semaphore, «сбой
одного запроса не валит батч»). Реализовано локально, чтобы data/gutenberg был
самодостаточным и не тянул весь src/.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You name a thematic cluster of Project Gutenberg books.

You are given the cluster's most representative Library-of-Congress subjects,
Gutenberg bookshelves, distinctive keywords, and a few example titles. Produce a
concise English category name for the cluster: 2-4 words, Title Case, like
"Detective Fiction", "Cookbooks", "Marine Biology", "Ancient Greek Drama",
"Children's Picture Books", "Christian Theology".

Output ONLY the category name — no quotes, no explanation, no trailing period.
"""


def _get_async_client(api_base: str):
    """AsyncOpenAI c OpenRouter base_url и ключом из OPENROUTER_API_KEY."""
    from openai import AsyncOpenAI

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY не задан. export OPENROUTER_API_KEY=sk-... "
            "(или запускайте с name.use_llm=false / --no-llm)"
        )
    return AsyncOpenAI(base_url=api_base, api_key=key)


def _titlecase(text: str) -> str:
    small = {"of", "and", "the", "in", "for", "a", "an", "to", "on"}
    words = re.split(r"\s+", text.strip())
    out = [
        w if (i and w.lower() in small) else (w[:1].upper() + w[1:])
        for i, w in enumerate(words) if w
    ]
    return " ".join(out)


def keyword_name(summary: dict) -> str:
    """Офлайн-имя кластера: лучшая полка → иначе c-TF-IDF слова → иначе LC-класс."""
    if summary.get("top_shelves"):
        return _titlecase(summary["top_shelves"][0])
    if summary.get("top_keywords"):
        return _titlecase(" ".join(summary["top_keywords"][:3]))
    if summary.get("top_locc"):
        return summary["top_locc"][0]
    return f"Cluster {summary['cluster']}"


def _format_summary(summary: dict) -> str:
    def _fmt(key: str) -> str:
        vals = summary.get(key) or []
        return ", ".join(vals) if vals else "—"

    return (
        f"Subjects: {_fmt('top_subjects')}\n"
        f"Bookshelves: {_fmt('top_shelves')}\n"
        f"LC classes: {_fmt('top_locc')}\n"
        f"Keywords: {_fmt('top_keywords')}\n"
        f"Example titles: {_fmt('example_titles')}\n"
        f"({summary['size']} books)\n\n"
        f"Category name:"
    )


async def _name_one(client, summary: dict, model: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _format_summary(summary)},
                ],
                temperature=0.0,
                max_tokens=16,
            )
            name = (resp.choices[0].message.content or "").strip().strip('"').rstrip(".")
            return name or keyword_name(summary)
        except Exception as e:  # noqa: BLE001 — сбой одного не валит остальные
            log.warning("LLM naming: ошибка на кластере #%d: %s", summary["cluster"], e)
            return keyword_name(summary)


async def _name_all_async(summaries, model, api_base, concurrency) -> dict[int, str]:
    client = _get_async_client(api_base)
    sem = asyncio.Semaphore(max(1, concurrency))
    try:
        names = await asyncio.gather(*(_name_one(client, s, model, sem) for s in summaries))
    finally:
        if hasattr(client, "close"):
            try:
                await client.close()
            except Exception:  # noqa: BLE001
                pass
    return {s["cluster"]: n for s, n in zip(summaries, names)}


def _dedupe(names_by_cluster: dict[int, str], summaries: list[dict]) -> dict[int, str]:
    """Гарантирует уникальность имён: коллизии различаем ключевым словом/id."""
    by_cluster = {s["cluster"]: s for s in summaries}
    used: set[str] = set()
    out: dict[int, str] = {}
    for cluster in sorted(names_by_cluster):
        name = names_by_cluster[cluster]
        if name not in used:
            out[cluster] = name
            used.add(name)
            continue
        # пробуем добавить различающее ключевое слово, затем id кластера
        for kw in by_cluster[cluster].get("top_keywords", []):
            cand = f"{name} ({_titlecase(kw)})"
            if cand not in used:
                name = cand
                break
        else:
            name = f"{name} (#{cluster})"
        out[cluster] = name
        used.add(name)
    return out


def name_clusters(
    summaries: list[dict],
    use_llm: bool = True,
    judge_model: str = "openai/gpt-4o-mini",
    api_base: str = "https://openrouter.ai/api/v1",
    concurrency: int = 8,
) -> dict[int, str]:
    """Имена всех кластеров {cluster_id: name}. use_llm=False → офлайн ключевые слова.

    Если LLM недоступен (нет ключа/пакета) — мягкий фолбэк на офлайн-имена.
    """
    if use_llm:
        try:
            raw = asyncio.run(_name_all_async(summaries, judge_model, api_base, concurrency))
        except Exception as e:  # noqa: BLE001 — не роняем пайплайн из-за наименования
            log.warning("LLM naming недоступен (%s) — офлайн-имена по ключевым словам", e)
            raw = {s["cluster"]: keyword_name(s) for s in summaries}
    else:
        raw = {s["cluster"]: keyword_name(s) for s in summaries}
    return _dedupe(raw, summaries)
