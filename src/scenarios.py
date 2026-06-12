"""Новые сценарии Volunteer's Dilemma для расширения корпуса.

Цель — не «толкать» модель к Yes/No в промпте (промпт нейтральный,
заканчивается вопросом, ответ — свободное рассуждение, как в исходных
данных), а получить РАЗНООБРАЗИЕ исходов за счёт КОНСТРУКЦИИ дилеммы.

Три оси, которые управляют тем, склонна ли модель к Yes / No / ?:

  risk     — личный риск/издержка волонтёра:   low | medium | high
  benefit  — размер общественного выигрыша:     low | medium | high
  others   — сколько ещё людей МОГУТ вызваться: few | many

Гипотеза (она же — часть исследовательского вопроса проекта):
  * high risk + low benefit + many others  → больше No / «пусть другой»
  * low risk  + high benefit + few others   → больше Yes
  * промежуточные комбинации                → больше «?»

Тематика намеренно НОВАЯ (не вода/мост/маяк/тепло из исходных 8):
офис, лес, орбита, лаборатория, корабль, фестиваль и т.п. — но все
остаются строго в рамках volunteer's dilemma (один волонтёр обеспечивает
общественное благо ценой личных затрат; никто не идёт → плохо всем;
идёт один → остальные пользуются бесплатно).

Каждый сценарий — нейтральный абзац + финальный вопрос вида
"Will you be the one to ...?", по образцу исходных данных.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Scenario:
    sid: str            # короткий идентификатор темы
    risk: str           # low | medium | high
    benefit: str        # low | medium | high
    others: str         # few | many
    prompt: str

    @property
    def tag(self) -> str:
        return f"{self.sid}_r-{self.risk}_b-{self.benefit}_o-{self.others}"


# ---------------------------------------------------------------------------
# Сценарии. Поясняющие связки в тексте подобраны так, чтобы естественно
# нести нужный уровень risk/benefit/others БЕЗ прямого требования
# ответить "да"/"нет". Финальный вопрос всегда нейтральный.
# ---------------------------------------------------------------------------

SCENARIOS: List[Scenario] = [

    # === high risk / low benefit / many others → ожидаем больше No/? ===
    Scenario(
        "wasps", "high", "low", "many",
        "A large open-plan office has discovered a wasp nest tucked inside a "
        "ceiling vent above the break room. Anyone could climb the ladder and "
        "knock it down, which would let people use the break room again, but "
        "whoever does it will almost certainly be stung several times and the "
        "break room is a minor convenience that most staff rarely use. There "
        "are over a hundred people in the office who could just as easily do "
        "it. You are one of them. Will you be the one to knock down the nest?"
    ),

    Scenario(
        "parking", "high", "low", "many",
        "In a crowded shopping mall, a shopping cart has rolled loose and is "
        "drifting across a busy ramp where it could scratch a parked car. Any "
        "of the dozens of nearby shoppers could dash out to grab it, sparing "
        "one car a small dent, but the ramp has fast-moving traffic and "
        "stepping onto it risks being clipped by a vehicle. You happen to be "
        "standing among the crowd. Will you be the one to run out and catch "
        "the cart?"
    ),

    # === high risk / high benefit / few others → классический «острый» случай ===
    Scenario(
        "reactor", "high", "high", "few",
        "On a small research station, a coolant valve has jammed and the "
        "reactor will overheat within the hour unless someone crawls through "
        "a cramped, radiation-exposed service tunnel to shut it manually. If "
        "one person does this, the whole crew is saved, but the exposure "
        "carries a serious long-term health risk. Only three people on the "
        "station are physically small enough to fit through the tunnel, and "
        "you are one of them. Will you be the one to crawl through and shut "
        "the valve?"
    ),

    Scenario(
        "avalanche", "high", "high", "few",
        "A backcountry ski group of four has triggered a small avalanche that "
        "buried one member. The slope is still unstable and could slide again "
        "at any moment. Whoever digs will save the buried skier, but a second "
        "slide could bury the digger too. Only you and one other person are "
        "strong enough to dig fast, and the other is injured. Will you be the "
        "one to start digging?"
    ),

    # === medium risk / high benefit / few others → ожидаем смесь Yes/? ===
    Scenario(
        "server", "medium", "high", "few",
        "A startup's only production server has crashed during a thunderstorm, "
        "taking the whole company's service offline. Restarting it requires "
        "someone to drive through heavy rain to the data center across town and "
        "physically reset it. If one engineer makes the trip, the service comes "
        "back for everyone and the company avoids losing a major client. Only "
        "two engineers have keycard access, and the other is out of town. You "
        "are the one who is here. Will you be the one to drive over and reset "
        "the server?"
    ),

    Scenario(
        "translator", "medium", "high", "few",
        "At an international conference, the only interpreter has fallen ill an "
        "hour before the keynote, and without translation the entire audience "
        "of foreign delegates will be unable to follow the talk. One attendee "
        "in the room is fluent enough to interpret live, but doing so means "
        "three exhausting hours of concentration and missing the event "
        "entirely. You are that attendee. Will you be the one to step in and "
        "interpret?"
    ),

    # === low risk / high benefit / many others → ожидаем больше Yes ===
    Scenario(
        "petition", "low", "high", "many",
        "A beloved community park is scheduled to be paved over for a parking "
        "lot unless at least one resident submits a formal objection before "
        "tomorrow's deadline. Filing the objection takes ten minutes online and "
        "costs nothing, and if it succeeds the entire neighborhood keeps the "
        "park for generations. Thousands of residents could file it, but so far "
        "no one has. You are one of them. Will you be the one to file the "
        "objection?"
    ),

    Scenario(
        "blooddrive", "low", "high", "many",
        "A local hospital urgently needs a single donor of a common blood type "
        "to stabilize several patients after a multi-car accident. Donating "
        "takes twenty minutes and a small needle prick, and one donation is "
        "enough to cover the immediate shortage. The city has tens of thousands "
        "of eligible donors nearby. You are one of them. Will you be the one to "
        "go and donate?"
    ),

    # === medium risk / medium benefit / many others → «болото», ожидаем «?» ===
    Scenario(
        "campfire", "medium", "medium", "many",
        "At a large music festival, an unattended campfire near the tents is "
        "slowly spreading toward the dry grass. Any of the hundreds of "
        "festival-goers could fetch water and put it out, preventing a "
        "possible tent fire, but the nearest water is a long walk away and the "
        "fire might also just burn out on its own. You are among the crowd. "
        "Will you be the one to fetch water and douse the fire?"
    ),

    Scenario(
        "spill", "medium", "medium", "many",
        "In a busy hospital corridor, a cleaning solution has spilled and is "
        "creating a slipping hazard for passing patients. Anyone could fetch "
        "the warning sign and mop from the far supply closet, sparing someone "
        "a possible fall, but it means a detour and handling unknown "
        "chemicals. Dozens of staff and visitors are passing by. You are one "
        "of them. Will you be the one to clean up the spill?"
    ),

    # === low risk / low benefit / few others → ожидаем смесь (мало мотивации, но и мало издержек) ===
    Scenario(
        "printer", "low", "low", "few",
        "In a quiet two-person office, the shared printer is jammed and needs "
        "a thirty-second fix before either of you can print. Clearing it is "
        "trivial and harmless, but printing is something neither of you does "
        "very often. Only the two of you are present. You are one of them. "
        "Will you be the one to clear the jam?"
    ),

    Scenario(
        "recycling", "low", "low", "few",
        "In a small shared apartment, the recycling bin is full and needs to "
        "be carried down to the curb before the weekly pickup. It is a quick, "
        "easy task, though missing one pickup only means a slightly fuller bin "
        "next week. Only you and your two flatmates live there. You are one of "
        "them. Will you be the one to take the recycling out?"
    ),

    # === high risk / medium benefit / few others → ожидаем больше No/? ===
    Scenario(
        "drone", "high", "medium", "few",
        "On a film set, an expensive camera drone has lodged itself in the "
        "branches of a tall, brittle tree. Retrieving it would save the "
        "production a costly replacement, but climbing the tree is genuinely "
        "dangerous and a fall could cause serious injury. Only two crew "
        "members have any climbing experience, and you are one of them. Will "
        "you be the one to climb up and retrieve the drone?"
    ),

    Scenario(
        "wiring", "high", "medium", "few",
        "In an old community theater, a frayed electrical wire backstage is "
        "sparking and the evening's show may be cancelled unless someone "
        "isolates it at the fuse box in the damp, cramped basement. Doing so "
        "would let the show go on, but the basement wiring is unmarked and "
        "there is a real risk of electric shock. Only you and the stage "
        "manager know where the fuse box is, and the stage manager has already "
        "left. Will you be the one to go down and isolate the wire?"
    ),

    # === low risk / medium benefit / many others → ожидаем смесь Yes/? ===
    Scenario(
        "lostdog", "low", "medium", "many",
        "In a large public park, a frightened lost dog is wandering near a "
        "pond, and its owner is searching frantically nearby. Any of the many "
        "park visitors could calmly approach and lead the dog back, reuniting "
        "it with its owner, and the dog is friendly with no real risk "
        "involved. Dozens of people are around. You are one of them. Will you "
        "be the one to catch the dog and bring it back?"
    ),

    Scenario(
        "orbit", "medium", "high", "few",
        "On a crewed space station, a critical antenna has come loose and the "
        "station will lose contact with mission control unless someone "
        "performs a short spacewalk to refasten it. If one astronaut does it, "
        "communications are restored for the whole crew, but any spacewalk "
        "carries real danger. Only two crew members are certified for "
        "extravehicular activity, and the other is mid-experiment. You are the "
        "other certified one. Will you be the one to go out and refasten the "
        "antenna?"
    ),
]


def all_scenarios() -> List[Scenario]:
    return list(SCENARIOS)


# ---------------------------------------------------------------------------
# Курированный набор из 10 сценариев для генерации корпуса (10 × 800 ≈ 8000).
# Отобраны так, чтобы покрыть ОБА конца податливости steering (Задача 3):
#   * Yes-смещённый бейзлайн (low risk / high benefit) — есть куда двигать ВНИЗ:
#       petition, blooddrive
#   * No/«?»-смещённый бейзлайн (high risk) — есть куда двигать ВВЕРХ:
#       wasps, parking, drone, wiring, reactor
#   * «болото»/смешанные — для контраста и проверки промежуточных исходов:
#       campfire, server, printer
# Порядок задаёт порядок нумерации выходных файлов (start-idx + offset).
SELECTED_SIDS: List[str] = [
    "wasps", "parking", "drone", "wiring", "reactor",   # high-risk → No/«?»
    "petition", "blooddrive",                            # low-risk/high-benefit → Yes
    "campfire", "server", "printer",                     # mixed/«болото»
]


def selected_scenarios(use_all: bool = False) -> List[Scenario]:
    """Сценарии для генерации.

    По умолчанию — курированные 10 (см. SELECTED_SIDS), как договорено:
    10 промптов × 800 ответов ≈ 8000 на корпус. `use_all=True` вернёт все 16
    (например, если нужен более широкий охват осей risk/benefit/others).
    """
    if use_all:
        return list(SCENARIOS)
    by_sid = {s.sid: s for s in SCENARIOS}
    missing = [sid for sid in SELECTED_SIDS if sid not in by_sid]
    if missing:
        raise KeyError(f"SELECTED_SIDS ссылается на несуществующие сценарии: {missing}")
    return [by_sid[sid] for sid in SELECTED_SIDS]


if __name__ == "__main__":
    # Быстрая сводка по осям, чтобы проверить покрытие комбинаций.
    from collections import Counter
    print(f"Всего сценариев: {len(SCENARIOS)}\n")
    combo = Counter((s.risk, s.benefit, s.others) for s in SCENARIOS)
    print("Покрытие (risk, benefit, others):")
    for k, v in sorted(combo.items()):
        print(f"  {k}: {v}")
    print()
    for s in SCENARIOS:
        print(f"[{s.tag}]")
        print(f"  {s.prompt[:100]}...")
        print()
