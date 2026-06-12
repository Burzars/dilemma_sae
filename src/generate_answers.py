"""Генерация reasoning-ответов на сценарии Volunteer's Dilemma через OpenRouter.

Пайплайн:
  1. Для каждого из 10 курированных сценариев (src/scenarios.py:selected_scenarios)
     генерируем фиксированное число ответов (по умолчанию 800) ОДНОЙ моделью —
     Qwen2.5-7B-Instruct — с нейтральным промптом и temperature=1.0. Промпт НЕ
     толкает к Yes/No: модель свободно рассуждает, как в исходных данных.
  2. Ответы классифицируем LLM-судьёй (gpt-4o-mini) в Yes/No/? — переиспользуем
     classify_decisions_async из src/interpret.py (тот же judge, что в
     steering-эксперименте, ради консистентности разметки). Генерим и судим
     ПАКЕТАМИ (--batch-size, по умолчанию 50), чтобы логировать накопительное
     распределение и не держать всё в памяти разом.
  3. Пишем по файлу на сценарий в формате, который читает src/data.py:
       { "prompt": "...", "answers": [ {"answer": "...", "type": "Yes|No|?"}, ... ] }
     Плюс служебное поле "scenario_meta" (data.py его игнорирует, но оно нужно
     для отбора промптов по податливости в steering — Задача 3).
     Имя файла — под glob в data.py: volunteer_data_llama_reasoning_<N>_res.json,
     нумерация с 9 (исходные 1..8 не трогаем).

КРИТИЧНО — одна и та же модель. Тексты генерим через OpenRouter моделью
`qwen/qwen-2.5-7b-instruct`, а активации потом снимаем ЛОКАЛЬНО тем же
чекпойнтом `Qwen/Qwen2.5-7B-Instruct` (см. configs/default.yaml: model.name).
Это должна быть ОДНА И ТА ЖЕ версия с ОДНИМ chat-template — иначе тексты и
их активации рассогласуются. Поэтому id генератора зафиксирован ниже как
GEN_MODEL и не должен подменяться на другую модель.

Дисбаланс классов: цель проекта — побольше редкого класса "No". Здесь мы
просто генерим фиксированное N на каждый сценарий; баланс достигается
КОНСТРУКЦИЕЙ сценариев (оси risk/benefit/others) + temperature=1.0, а уже
ПОСЛЕ генерации делаем сбалансированный срез утилитой src/build_datasets.py.

Запуск:
    export OPENROUTER_API_KEY=sk-...
    python -m src.generate_answers \
        --out-dir data/results_dilemma_v2 \
        --n-per-prompt 800 \
        --concurrency 8
    # дешёвая проверка пайплайна без большого бюджета:
    python -m src.generate_answers --smoke

Зависимости: openai>=1.0 (как в requirements.txt проекта).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

# Делаем корень проекта импортируемым независимо от способа запуска
# (python src/generate_answers.py | python -m src.generate_answers | import).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.interpret import classify_decisions_async, _get_async_client  # noqa: E402
from src.scenarios import Scenario, selected_scenarios  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("generate")


# ---------------------------------------------------------------------------
# Зафиксированные модели (см. шапку: генератор и локальный extract — одно и то же).
# ---------------------------------------------------------------------------
GEN_MODEL = "qwen/qwen-2.5-7b-instruct"      # OpenRouter id генератора
LOCAL_MODEL = "Qwen/Qwen2.5-7B-Instruct"     # ДОЛЖЕН совпадать (локальный extract)
JUDGE_MODEL = "openai/gpt-4o-mini"           # судья Yes/No/? (как в interpret.py)


# Системный промпт намеренно НЕ требует "answer yes or no". Просим
# рассуждать от первого лица, как участник ситуации, — это воспроизводит
# стиль исходных данных, где ответы развёрнутые.
GEN_SYSTEM_PROMPT = (
    "You are a thoughtful person reasoning out loud about a real situation "
    "you find yourself in. Think through the considerations honestly and in "
    "the first person, as the participant described. Do not give a checklist "
    "lecture; reason naturally about what you would actually do and why."
)


# ---------------------------------------------------------------------------
# Генерация ответов (нейтральный промпт, свободное рассуждение)
# ---------------------------------------------------------------------------

async def _generate_one(client, idx, scenario_prompt, model, temperature, max_tokens, sem) -> str:
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": GEN_SYSTEM_PROMPT},
                    {"role": "user", "content": scenario_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:  # noqa: BLE001 — одиночный сбой не валит батч
            log.warning("Ошибка генерации #%d: %s", idx, e)
            return ""


async def _generate_batch(
    client, scenario_prompt, *, model, n, temperature, max_tokens, sem,
) -> List[str]:
    """n ответов на один промпт, параллельно. Пустые (сбои API) выкидываются.

    Каждый вызов OpenRouter сэмплирует независимо, поэтому при temperature=1.0
    отдельный seed на запрос не нужен — стохастичность даёт сам провайдер.
    """
    tasks = [
        _generate_one(client, i, scenario_prompt, model, temperature, max_tokens, sem)
        for i in range(n)
    ]
    answers = await asyncio.gather(*tasks)
    return [a for a in answers if a]


async def generate_and_label_for_scenario(
    client,
    scenario: Scenario,
    *,
    gen_model: str,
    judge_model: str,
    api_base: str,
    n: int,
    batch_size: int,
    temperature: float,
    max_tokens: int,
    sem: asyncio.Semaphore,
    concurrency: int,
) -> Tuple[List[str], List[str], Counter]:
    """Набирает n ответов на сценарий ПАКЕТАМИ: генерация → разметка судьёй.

    Возвращает (answers, labels, dist). Никакой адаптивной остановки — просто
    фиксированное n (по договорённости: по 800 на промпт). Лог накопительного
    распределения после каждого пакета.
    """
    answers: List[str] = []
    labels: List[str] = []
    dist: Counter = Counter()
    batch_no = 0

    while len(answers) < n:
        remaining = n - len(answers)
        want = min(batch_size, remaining)
        batch = await _generate_batch(
            client, scenario.prompt,
            model=gen_model, n=want, temperature=temperature,
            max_tokens=max_tokens, sem=sem,
        )
        if not batch:
            # Полностью пустой пакет = устойчивый сбой API; не зацикливаемся.
            log.warning("  [%s] пустой пакет (API?), останавливаю сценарий на %d/%d",
                        scenario.sid, len(answers), n)
            break

        batch_labels = await classify_decisions_async(
            batch, judge_model=judge_model, api_base=api_base,
            concurrency=concurrency, client=client,
        )
        answers.extend(batch)
        labels.extend(batch_labels)
        dist.update(batch_labels)
        batch_no += 1
        log.info("  [%s] пакет %d: +%d (всего %d/%d) накоплено=%s",
                 scenario.sid, batch_no, len(batch), len(answers), n, dict(dist))

    return answers, labels, dist


# ---------------------------------------------------------------------------
# Основной цикл
# ---------------------------------------------------------------------------

async def run(args) -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY не задан. export OPENROUTER_API_KEY=sk-...")

    if args.gen_model != GEN_MODEL:
        # Не запрещаем жёстко (вдруг нужен другой OpenRouter-алиас той же модели),
        # но громко предупреждаем: генератор обязан совпадать с локальным extract.
        log.warning(
            "gen-model=%s != зафиксированного %s. Активации снимаются моделью %s — "
            "генератор и extract ДОЛЖНЫ быть одной версией с одним chat-template!",
            args.gen_model, GEN_MODEL, LOCAL_MODEL,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = selected_scenarios(use_all=args.all_scenarios)
    if args.max_scenarios is not None:
        scenarios = scenarios[: args.max_scenarios]
    target_total = len(scenarios) * args.n_per_prompt
    log.info("Сценариев к обработке: %d × %d ответов ≈ %d на корпус (генератор=%s, судья=%s)",
             len(scenarios), args.n_per_prompt, target_total, args.gen_model, args.judge_model)

    client = _get_async_client(args.api_base)          # один клиент на gen + judge
    sem = asyncio.Semaphore(max(1, args.concurrency))
    overall = Counter()

    try:
        for offset, scenario in enumerate(scenarios):
            file_idx = args.start_idx + offset
            log.info(
                "[%d/%d] сценарий %s (r=%s b=%s o=%s) → набираю %d ответов",
                offset + 1, len(scenarios), scenario.sid,
                scenario.risk, scenario.benefit, scenario.others, args.n_per_prompt,
            )

            answers, labels, dist = await generate_and_label_for_scenario(
                client, scenario,
                gen_model=args.gen_model,
                judge_model=args.judge_model,
                api_base=args.api_base,
                n=args.n_per_prompt,
                batch_size=args.batch_size,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                sem=sem,
                concurrency=args.concurrency,
            )
            if not answers:
                log.warning("  сценарий %s: 0 ответов, пропускаю файл", scenario.sid)
                continue

            overall.update(dist)
            log.info("  сценарий %s ИТОГ: %s", scenario.sid, dict(dist))

            payload = {
                "prompt": scenario.prompt,
                "scenario_meta": {            # доп. поле — data.py его игнорит
                    "sid": scenario.sid,
                    "risk": scenario.risk,
                    "benefit": scenario.benefit,
                    "others": scenario.others,
                    "gen_model": args.gen_model,
                    "judge_model": args.judge_model,
                    "temperature": args.temperature,
                },
                "answers": [
                    {"answer": a, "type": t} for a, t in zip(answers, labels)
                ],
            }
            out_path = out_dir / f"volunteer_data_llama_reasoning_{file_idx}_res.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            log.info("  → сохранено %s (%d ответов)", out_path.name, len(answers))
    finally:
        if hasattr(client, "close"):
            try:
                await client.close()
            except Exception:  # noqa: BLE001
                pass

    log.info("ГОТОВО. Итоговое распределение по всему корпусу: %s", dict(overall))
    tot = sum(overall.values())
    if tot:
        for k in ("Yes", "No", "?"):
            log.info("  %s: %d (%.1f%%)", k, overall[k], 100 * overall[k] / tot)
    log.info("Дальше: 1) снять активации на этом FULL-срезе локально (extract); "
             "2) python -m src.build_datasets для full/balanced срезов.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="data/results_dilemma_v2")
    p.add_argument("--gen-model", default=GEN_MODEL,
                   help=f"генератор (OpenRouter id). Зафиксирован {GEN_MODEL}; "
                        f"ДОЛЖЕН совпадать с локальным extract ({LOCAL_MODEL}).")
    p.add_argument("--judge-model", default=JUDGE_MODEL,
                   help="судья Yes/No/? (тот же, что в проекте)")
    p.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    p.add_argument("--n-per-prompt", type=int, default=800,
                   help="фиксированное число ответов на каждый сценарий")
    p.add_argument("--batch-size", type=int, default=50,
                   help="размер пакета генерация→разметка (для лога и устойчивости)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="высокая T → больше разнообразия исходов")
    p.add_argument("--max-tokens", type=int, default=600)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--start-idx", type=int, default=9,
                   help="с какого номера нумеровать файлы (исходные занимают 1..8)")
    p.add_argument("--all-scenarios", action="store_true",
                   help="взять все 16 сценариев вместо курированных 10")
    p.add_argument("--max-scenarios", type=int, default=None,
                   help="ограничить число сценариев (для отладки)")
    p.add_argument("--smoke", action="store_true",
                   help="дешёвый прогон: 2 сценария × 8 ответов, batch 4")
    args = p.parse_args()
    if args.smoke:
        args.n_per_prompt = 8
        args.batch_size = 4
        if args.max_scenarios is None:
            args.max_scenarios = 2
    return args


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
