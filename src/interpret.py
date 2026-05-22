"""LLM-судья для авто-интерпретации SAE-нейронов.

Принцип: для каждого нейрона у нас есть top-K сниппетов reasoning-текста,
где нейрон активировался сильнее всего. Скармливаем их LLM через
OpenRouter и просим вернуть одну общую тему (или "NO CLEAR THEME"
для шумных нейронов).

Системный промпт настроен на возврат "NO CLEAR THEME" — без него
LLM начинает галлюцинировать темы из пунктуационного мусора.

Токен:
  - переменная окружения OPENROUTER_API_KEY (для сервера);
  - в Colab можно сначала загнать в env через google.colab.userdata.
"""

from __future__ import annotations

import logging
import os
from typing import List

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """You analyze hidden-state features in a language model that
was reasoning about a Volunteer's Dilemma (a social dilemma in which the
question is whether to volunteer effort for a public good).

You are given the top-activating contexts of a single SAE feature
(neuron). Each context is a short snippet of the model's reasoning at the
moment the feature was firing most strongly.

Your job: identify the common theme of these contexts in ONE short
English phrase (max ~10 words). The phrase should describe what unifies
the contexts — e.g. "expressions of moral duty", "weighing personal
cost vs group benefit", "explicit refusal to act".

If the contexts are too noisy or share no clear semantic theme (e.g.
only common punctuation, generic connectives, etc.), respond with exactly:
  NO CLEAR THEME

Do not output explanations, only the phrase or NO CLEAR THEME.
"""


def _get_client(api_base: str):
    """Создаёт OpenAI-совместимый клиент с OpenRouter base_url.

    Ищет ключ в OPENROUTER_API_KEY.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "openai package required. pip install 'openai>=1.0,<2.0'"
        ) from e

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Export it before running: "
            "  export OPENROUTER_API_KEY=sk-..."
        )
    return OpenAI(base_url=api_base, api_key=key)


def _format_user_message(contexts: List[dict], max_snippets: int) -> str:
    """Форматируем top контексты в один user-message."""
    lines = ["Top activating contexts for this feature:", ""]
    for i, c in enumerate(contexts[:max_snippets], 1):
        snippet = c["snippet"].replace("\n", " ").strip()
        # ограничим длину одного сниппета чтобы не выжирать контекст
        if len(snippet) > 300:
            snippet = snippet[:300] + "…"
        label = c.get("label", "?")
        lines.append(f"  {i:>2}. [label={label}] ...{snippet}")
    lines.append("")
    lines.append("What is the common theme? (one short phrase or NO CLEAR THEME)")
    return "\n".join(lines)


def interpret_neuron(
    contexts: List[dict],
    judge_model: str = "openai/gpt-4o-mini",
    api_base: str = "https://openrouter.ai/api/v1",
    max_snippets: int = 15,
    client=None,
) -> str:
    """Один вызов LLM-судьи для одного нейрона."""
    if client is None:
        client = _get_client(api_base)
    user_msg = _format_user_message(contexts, max_snippets)
    resp = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=64,
    )
    return resp.choices[0].message.content.strip()


def interpret_report(
    report: List[dict],
    judge_model: str = "openai/gpt-4o-mini",
    api_base: str = "https://openrouter.ai/api/v1",
    max_snippets: int = 15,
) -> List[dict]:
    """Прогоняет весь report (список словарей с top_contexts) через LLM-судью.

    Возвращает копию report с добавленным полем "theme" у каждого
    нейрона. В случае ошибки API записывает строку "ERROR: <msg>".
    """
    client = _get_client(api_base)
    out: list[dict] = []
    for i, item in enumerate(report):
        nid = item["neuron_id"]
        try:
            theme = interpret_neuron(
                item["top_contexts"],
                judge_model=judge_model,
                api_base=api_base,
                max_snippets=max_snippets,
                client=client,
            )
        except Exception as e:
            log.warning("Ошибка LLM для нейрона #%d: %s", nid, e)
            theme = f"ERROR: {e}"
        log.info("[%d/%d] neuron #%d (%s) → %s",
                 i + 1, len(report), nid, item.get("kind", "?"), theme)
        new_item = dict(item)
        new_item["theme"] = theme
        out.append(new_item)
    return out
