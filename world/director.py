"""Turning what someone asked for into changes to the world.

Everything a model may touch is listed in :data:`EDITABLE` with the range each one
is held inside. A value outside its range is pulled back to the nearest end and the
correction is reported, because a silent correction leaves the reader believing the
world ended up somewhere nobody asked for.

The seed is deliberately not editable: rewriting it only destroys the ability to
tell the same story twice.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from world.goods import ALL_GOODS, Goods
from world.llm import LLMClient, ModelTimeout, ModelUnreachable, NoJSONAnswer
from world.village import Village
from world.world import World

STOCK_LIMIT = 1_000_000.0
PRODUCTION_LIMIT = 1_000.0
CONSUMPTION_LIMIT = 1_000.0
POPULATION_LIMIT = 1_000


@dataclass(frozen=True)
class Field:
    """One thing a model is allowed to change, and the range it is held inside.

    Attributes:
        name: How the model writes it, for example ``production.Minerals``.
        read: How to read it off a target, which is a village or the world.
        write: How to put it back.
        low: Smallest value that makes sense.
        high: Largest value that makes sense.
        whole: Whether the value must be a whole number.
        unit: What the number counts, since a number without one cannot be scaled.
    """

    name: str
    read: Callable[[Village | World], float]
    write: Callable[[Village | World, float], None]
    low: float
    high: float
    whole: bool = False
    unit: str = ""


def _world_field(name: str, attribute: str, low: float, high: float, whole: bool = False):
    return Field(
        name=name,
        read=lambda world: getattr(world, attribute),
        write=lambda world, value: setattr(world, attribute, int(value) if whole else value),
        low=low,
        high=high,
        whole=whole,
    )


def _village_field(name: str, attribute: str, low: float, high: float, whole: bool = False):
    return Field(
        name=name,
        read=lambda village: getattr(village, attribute),
        write=lambda village, value: setattr(
            village, attribute, int(value) if whole else value
        ),
        low=low,
        high=high,
        whole=whole,
    )


def _stock_field(goods: Goods) -> Field:
    def read(village: Village | World) -> float:
        return village.stock[goods]  # type: ignore[union-attr]

    def write(village: Village | World, value: float) -> None:
        village.stock[goods] = value  # type: ignore[union-attr]

    return Field(
        name=f"stock.{goods.label}",
        read=read,
        write=write,
        low=0.0,
        high=STOCK_LIMIT,
    )


def _consumption_field(goods: Goods) -> Field:
    def read(village: Village | World) -> float:
        return village.consumption[goods]  # type: ignore[union-attr]

    def write(village: Village | World, value: float) -> None:
        village.consumption[goods] = value  # type: ignore[union-attr]

    return Field(
        name=f"consumption.{goods.label}",
        read=read,
        write=write,
        low=0.0,
        high=CONSUMPTION_LIMIT,
        unit="units per day for the whole village",
    )


def _production_field(goods: Goods) -> Field:
    def read(village: Village | World) -> float:
        return village.production[goods]  # type: ignore[union-attr]

    def write(village: Village | World, value: float) -> None:
        village.production[goods] = value  # type: ignore[union-attr]

    return Field(
        name=f"production.{goods.label}",
        read=read,
        write=write,
        low=0.0,
        high=PRODUCTION_LIMIT,
        unit="units per day for the whole village",
    )


WORLD_FIELDS: dict[str, Field] = {
    field.name: field
    for field in (
        _world_field("time_per_day", "time_per_day", 0.0, 24.0, whole=True),
        _world_field("trade_chance", "trade_chance", 0.0, 1.0),
        _world_field("max_rounds", "max_rounds", 1.0, 10.0, whole=True),
    )
}

VILLAGE_FIELDS: dict[str, Field] = {
    field.name: field
    for field in (
        _village_field("population", "population", 1.0, float(POPULATION_LIMIT), whole=True),
        _village_field("reserve_days", "reserve_days", 0.0, 10.0),
        _village_field("production_noise", "production_noise", 0.0, 1.0),
        *(_stock_field(goods) for goods in ALL_GOODS),
        *(_production_field(goods) for goods in ALL_GOODS),
        *(_consumption_field(goods) for goods in ALL_GOODS),
    )
}

NEVER_EDITABLE = ("seed",)


@dataclass
class Change:
    """One value a model wants set, and what was done about it.

    ``previous`` is read when the change is carried out, so two changes to one
    setting each report what the one before it left behind.

    Attributes:
        target: Name of the village, or ``None`` for the world itself.
        field: The field the model named.
        asked: The value it asked for.
        previous: What the value was before this change, filled in on the way in.
        applied: The value that was actually set.
        note: Why it differs from the request, or ``None`` when it does not.
    """

    target: str | None
    field: str
    asked: float
    previous: float = 0.0
    applied: float = 0.0
    note: str | None = None

    def line(self) -> str:
        """One line saying what changed, for the screen.

        The old value is always shown, so a reader can see what a request actually
        did rather than only where it left things.
        """
        where = f"{self.target} " if self.target else ""
        head = f"{where}{self.field}: {_shown(self.previous)} -> {_shown(self.applied)}"
        if self.note is None:
            return head
        return f"{head}  ({self.note})"

    def log_line(self) -> str:
        """The same line with the value asked for, for the written record."""
        if self.asked == self.applied:
            return self.line()
        return f"{self.line()}  [asked for {_shown(self.asked)}]"


def _said_once(lines: Sequence[str]) -> list[str]:
    """Drop repeats, keeping the order they first arrived in."""
    seen: set[str] = set()
    out = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out


def _shown(value: float) -> str:
    """A value as it is written on screen, without a pointless decimal tail."""
    return f"{value:g}"


@dataclass
class Reply:
    """What a request turned into, in a shape a person can read.

    Attributes:
        said: What the model said it was doing.
        applied: The changes that went through, each with what it replaced.
        held: What was set aside for a later day.
        refused: Why anything did not.
        unmapped: Parts of the request the model said it had nowhere to put.
        prompt: The instruction as given, so a bad answer can be traced back.
        broke: Set when the answer could not be read at all, asking for a retry.
    """

    said: str = ""
    applied: list[Change] = field(default_factory=list)
    held: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    unmapped: list[str] = field(default_factory=list)
    prompt: str = ""
    broke: str | None = None

    def line(self) -> str:
        """The whole reply as the lines a chat shows.

        A refusal is shown as a refusal rather than as one more line of narration,
        because a request that did not do what it said has to read as a request
        that failed. Saying the same thing once is enough.
        """
        out = [self.said] if self.said else []
        out += [change.line() for change in self.applied]
        out += self.held
        problems = _said_once(self.refused)
        if self.unmapped:
            problems = [
                f"I could not make sense of {part!r} as anything I can change."
                for part in _said_once(self.unmapped)
            ] + problems
        if problems:
            out.append("")
            out.append("Not done:")
            out += problems
        if self.broke is not None:
            out.append(self.broke)
        return "\n".join(out) if out else "Nothing to change."

    def worked(self) -> bool:
        """Whether anything at all came of the request."""
        return bool(self.applied or self.held or not self.refused)


@dataclass(frozen=True)
class ScheduledChange:
    """Changes held back for a later day.

    Attributes:
        day: The day they land on.
        label: What the day is called in the log.
        changes: What to set when it arrives.
    """

    day: int
    label: str
    changes: tuple[Change, ...]


@dataclass
class Plan:
    """A read instruction: what the model wants done, and when.

    Attributes:
        said: What the model said it was doing.
        now: Changes to apply straight away.
        later: Changes held back for a named day.
    """

    said: str = ""
    now: list[Change] = field(default_factory=list)
    later: list[ScheduledChange] = field(default_factory=list)
    unmapped: list[str] = field(default_factory=list)


class Director:
    """Reads an instruction and carries out what it can safely."""

    SYSTEM_PROMPT = (
        "You run a simulation of villages that barter with each other. "
        "The user describes a change they want. Answer with JSON only.\n"
        'Shape: {"reply": "what you did, in one or two plain sentences",\n'
        '        "changes": [{"village": "Farmers", "field": "population", "value": 20}],\n'
        '        "schedule": [{"day": 12, "label": "the drought",\n'
        '                     "changes": [{"village": "Farmers",\n'
        '                                  "field": "production.Agriculture", "value": 2}]}],\n'
        '        "unmapped": ["the part of the request you could not express"]}\n'
        "Rules:\n"
        "- Values are the new total, never an amount to add or subtract. "
        "To halve something, work out the current value and give the new one.\n"
        "- A setting of the whole world, such as trade_chance, needs "
        '"village": null. A setting of one village, such as population, needs that '
        "village's name. Getting this backwards loses the change.\n"
        "- Only name the fields from the settings list. Never invent a field, and "
        "never rename a village.\n"
        "- NEVER approximate. A request has no setting for it until you have found "
        "one whose name says exactly that. A tax rate is not a harvest, a morale "
        "is not a population, and hunger is not a shortage of stores. If no field "
        "matches, change nothing for it and name it in unmapped instead. Guessing "
        "here does worse than refusing: the world moves somewhere nobody asked for "
        "and nobody can tell.\n"
        "- The seed is not changeable and must never appear in a change.\n"
        "- Use schedule only when the user says when. Otherwise use changes.\n"
        "- If nothing should change, return an empty changes list and say why.\n"
        "Worked example. The user says: make it harsher, the mines run dry and "
        "nobody meets anyone.\n"
        '{"reply": "I emptied every mineral store and stopped meetings.",\n'
        ' "changes": [{"village": null, "field": "trade_chance", "value": 0.0},\n'
        '              {"village": "Miners", "field": "production.Minerals", "value": 0.0},\n'
        '              {"village": "Farmers", "field": "stock.Minerals", "value": 0.0}],\n'
        ' "schedule": [], "unmapped": []}\n'
        'A request for a tax rate, a morale, or a market price has no field. The '
        'answer is to leave the changes empty and say so:\n'
        '{"reply": "There is no tax rate in this world, so I changed nothing.",\n'
        ' "changes": [], "schedule": [], "unmapped": ["a tax rate for the Miners"]}\n'
    )

    def __init__(self, client: LLMClient, world: World) -> None:
        self.client = client
        self.world = world

    def follow(self, instruction: str) -> Reply:
        """Read one instruction and do what it asks, as far as it can safely."""
        prompt = self._brief(instruction)
        reply = Reply(prompt=prompt)
        try:
            answer = self.client.ask_json(self.SYSTEM_PROMPT, prompt)
        except NoJSONAnswer as broken:
            reply.broke = (
                "That could not be read as a set of changes.\n"
                f"You asked: {instruction}\n"
                f"The model answered: {broken.answer}\n"
                "Say it again."
            )
            return reply
        except ModelTimeout as broken:
            reply.broke = f"The model took too long. {broken}."
            return reply
        except ModelUnreachable as broken:
            reply.broke = f"The model could not be reached: {broken}"
            return reply

        plan = self._read(answer, reply)
        reply.said = plan.said
        reply.unmapped = list(plan.unmapped)
        self._carry_out(plan, reply)
        return reply

    def settings(self) -> dict[str, object]:
        """What a model is allowed to touch, and what it may not."""
        return {
            "world": sorted(WORLD_FIELDS),
            "village": sorted(VILLAGE_FIELDS),
            "villages": [village.name for village in self.world.villages],
            "commodities": [goods.label for goods in ALL_GOODS],
            "ranges": {
                name: [field.low, field.high]
                for name, field in {**WORLD_FIELDS, **VILLAGE_FIELDS}.items()
            },
            "units": {
                name: field.unit
                for name, field in {**WORLD_FIELDS, **VILLAGE_FIELDS}.items()
                if field.unit
            },
            "not_changeable": list(NEVER_EDITABLE),
        }

    def _brief(self, instruction: str) -> str:
        return json.dumps(
            {"today": self.world.day + 1, "state": self._state(), **self.settings()},
            indent=2,
        ) + f"\n\nThe user says: {instruction}"

    def _state(self) -> dict[str, object]:
        villages = {}
        for village in self.world.villages:
            villages[village.name] = {
                "population": village.population,
                "reserve_days": village.reserve_days,
                "production_noise": village.production_noise,
                "production": {
                    goods.label: round(village.production[goods], 2) for goods in ALL_GOODS
                },
                "stock": {
                    goods.label: round(village.stock[goods], 2) for goods in ALL_GOODS
                },
            }
        return {
            "day_simulated": self.world.day,
            "time_per_day": self.world.time_per_day,
            "trade_chance": self.world.trade_chance,
            "max_rounds": self.world.max_rounds,
            "villages": villages,
        }

    def _read(self, answer: object, reply: Reply) -> Plan:
        """Turn a model's answer into a plan, refusing anything it should not touch."""
        if not isinstance(answer, dict):
            reply.broke = "That was not a set of changes. Say it again."
            return Plan()
        said = answer.get("reply")
        plan = Plan(said=said if isinstance(said, str) else "")
        for raw in _entries(answer, "changes"):
            change = self._read_change(raw, reply)
            if change is not None:
                plan.now.append(change)
        for raw in _entries(answer, "schedule"):
            self._read_scheduled(raw, reply, plan)
        unmapped = answer.get("unmapped")
        if isinstance(unmapped, list):
            plan.unmapped = [str(part) for part in unmapped if str(part).strip()]
        return plan

    def _read_scheduled(self, raw: object, reply: Reply, plan: Plan) -> None:
        if not isinstance(raw, dict):
            reply.refused.append("A scheduled item was not an object, so it was dropped.")
            return
        day = raw.get("day")
        if not isinstance(day, int) or isinstance(day, bool) or day < 1:
            reply.refused.append(
                f"A scheduled item named no usable day ({day!r}), so it was dropped."
            )
            return
        label = raw.get("label")
        changes = []
        for entry in _entries(raw, "changes"):
            change = self._read_change(entry, reply)
            if change is not None:
                changes.append(change)
        if not changes:
            return
        plan.later.append(
            ScheduledChange(
                day=day,
                label=label if isinstance(label, str) and label else f"day {day}",
                changes=tuple(changes),
            )
        )

    def _read_change(self, raw: object, reply: Reply) -> Change | None:
        if not isinstance(raw, dict):
            reply.refused.append("A change was not an object, so it was dropped.")
            return None
        target = raw.get("village")
        if target is not None and not isinstance(target, str):
            reply.refused.append(f"A change named no village ({target!r}), so it was dropped.")
            return None
        name = raw.get("field")
        if not isinstance(name, str):
            reply.refused.append(f"A change named no field ({name!r}), so it was dropped.")
            return None
        table = VILLAGE_FIELDS if target else WORLD_FIELDS
        spec = table.get(name)
        if spec is None:
            elsewhere = (WORLD_FIELDS if target else VILLAGE_FIELDS).get(name)
            if elsewhere is not None and target is None:
                reply.refused.append(
                    f"{name} is a setting of one village, not of the world, so it was dropped."
                )
            elif elsewhere is not None:
                reply.refused.append(
                    f"{name} is a setting of the world, so it must not name a village. "
                    "It was dropped."
                )
            else:
                reply.refused.append(
                    f"There is no setting called {name!r}, so it was dropped. "
                    f"Changeable here: {', '.join(sorted(table))}."
                )
            return None
        if target and not self._known(target):
            reply.refused.append(
                f"There is no village called {target!r}, so it was dropped. "
                f"The villages are: {', '.join(v.name for v in self.world.villages)}."
            )
            return None
        asked = raw.get("value")
        if not isinstance(asked, (int, float)) or isinstance(asked, bool):
            reply.refused.append(f"{name} was given {asked!r} rather than a number, so it was dropped.")
            return None
        applied, note = _hold(spec, float(asked))
        return Change(
            target=target, field=name, asked=float(asked), applied=applied, note=note
        )

    def _carry_out(self, plan: Plan, reply: Reply) -> None:
        """Apply what was asked, now and later."""
        for change in plan.now:
            self._apply(change)
            if change.previous == change.applied:
                where = change.target or "the world"
                reply.refused.append(
                    f"{where} {change.field} was already "
                    f"{_shown(change.applied)}, so nothing changed."
                )
            else:
                reply.applied.append(change)
        for scheduled in plan.later:
            self.world.schedule(
                _event_for(self.world, scheduled, lambda change: self._apply(change))
            )
            reply.held.append(
                f"held for day {scheduled.day}, {scheduled.label}: "
                + ", ".join(change.field for change in scheduled.changes)
            )

    def _known(self, name: str) -> bool:
        """Whether a village of that name is in the world right now."""
        return any(village.name == name for village in self.world.villages)

    def _apply(self, change: Change) -> None:
        """Put one change into the world, remembering what was there first."""
        spec = (VILLAGE_FIELDS if change.target else WORLD_FIELDS)[change.field]
        target = self.world.village(change.target) if change.target else self.world
        change.previous = float(spec.read(target))
        spec.write(target, change.applied)


def _hold(spec: Field, value: float) -> tuple[float, str | None]:
    """Pull a value inside its range, saying how far it had to move."""
    if spec.whole:
        value = float(round(value))
    held = min(spec.high, max(spec.low, value))
    if held == value:
        return held, None
    if value > spec.high:
        return held, f"held down to {held:g}, the most allowed"
    return held, f"held up to {held:g}, the least allowed"


def _entries(answer: dict, key: str) -> Iterable[object]:
    raw = answer.get(key)
    return raw if isinstance(raw, list) else ()


def _event_for(world: World, scheduled: ScheduledChange, apply: Callable[[Change], None]):
    """Turn held-back changes into something the world can run on its day."""
    from world.world import ScheduledEvent

    def run(_world: World) -> None:
        for change in scheduled.changes:
            apply(change)

    return ScheduledEvent(day=scheduled.day, label=scheduled.label, apply=run)


__all__ = [
    "NEVER_EDITABLE",
    "VILLAGE_FIELDS",
    "WORLD_FIELDS",
    "Change",
    "Director",
    "Field",
    "Plan",
    "Reply",
    "ScheduledChange",
]