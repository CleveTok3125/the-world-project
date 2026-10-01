"""The world itself: a set of villages and the tick that moves them forward."""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from world.agent import NegotiationBrain
from world.goods import ALL_GOODS, Goods
from world.negotiation import MAX_ROUNDS, NegotiationResult, negotiate
from world.village import DayRecord, Village

DEFAULT_POPULATION = 10
SPECIALISATION_OUTPUT = 20.0
SIDE_OUTPUT = 5.0
OPENING_STOCK = 20.0
RESERVE_DAYS = 1.0
DEFAULT_TIME_PER_DAY = 3
DEFAULT_TRADE_CHANCE = 0.6


@dataclass(frozen=True)
class TradeEvent:
    """One barter, seen from outside.

    Attributes:
        proposer: Name of the village that opened the conversation.
        responder: Name of the village that answered it.
        result: The full negotiation outcome, transcript included.
    """

    proposer: str
    responder: str
    result: NegotiationResult


@dataclass(frozen=True)
class DayReport:
    """Everything that happened during one simulated day.

    Attributes:
        day: The day number that has just been completed.
        production: Units produced per village and commodity.
        consumption: Units actually consumed per village and commodity.
        unmet: Units each village wanted but could not obtain.
        trades: The barters that were attempted, in the order they took place.
        hours_left: Barters the day had room for that were still unused when it
            closed.
        hours_total: Barters the day had room for when it opened.
    """

    day: int
    production: dict[str, dict[Goods, float]]
    consumption: dict[str, dict[Goods, float]]
    unmet: dict[str, dict[Goods, float]]
    trades: tuple[TradeEvent, ...] = ()
    hours_left: int = 0
    hours_total: int = 0

    @property
    def hours_spent(self) -> int:
        """How many of the day's barters were taken up."""
        return self.hours_total - self.hours_left

    @property
    def fully_supplied(self) -> bool:
        """Whether every village met its own needs that day.

        This is not the same as an even spread: three villages holding nothing at
        all are level with one another and still starving.
        """
        return not any(
            amount > 0 for unmet in self.unmet.values() for amount in unmet.values()
        )


@dataclass(frozen=True)
class ScheduledEvent:
    """Something waiting to happen to the world on a given day.

    Attributes:
        day: The day it fires on, counted the way ``advance_day`` counts.
        label: What it is, in a few words, for the run log and the narrator.
        apply: What it does to the world when its day arrives.
    """

    day: int
    label: str
    apply: Callable[[World], None]

    def fires_today(self, day: int) -> bool:
        """Whether this is the day it was waiting for."""
        return self.day == day


@dataclass
class World:
    """A closed world of villages that trade with each other day after day.

    Attributes:
        villages: The villages taking part, in a fixed order.
        day: How many days have been simulated so far.
        max_rounds: Negotiation rounds allowed per barter.
        time_per_day: Barters the world has room for in one day, in total.
        trade_chance: Likelihood that a pair of villages meets on any given day.
        seed: Makes who meets whom reproducible from run to run.
        scheduled: What is waiting to happen, in the order it was arranged.
        fired: What has already happened, oldest first.
    """

    villages: Sequence[Village]
    day: int = 0
    max_rounds: int = MAX_ROUNDS
    time_per_day: int = DEFAULT_TIME_PER_DAY
    trade_chance: float = DEFAULT_TRADE_CHANCE
    seed: int = 0
    scheduled: list[ScheduledEvent] = field(default_factory=list)
    fired: list[ScheduledEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.villages = list(self.villages)
        if not self.villages:
            raise ValueError("a world needs at least one village")
        names = [village.name for village in self.villages]
        duplicates = [name for name in names if names.count(name) > 1]
        if duplicates:
            raise ValueError(f"village names must be unique, duplicated: {sorted(set(duplicates))}")
        self._rng = random.Random(self.seed)

    def schedule(self, event: ScheduledEvent) -> ScheduledEvent:
        """Put something in the calendar and hand it back for convenience."""
        self.scheduled.append(event)
        return event

    def village(self, name: str) -> Village:
        """Find a village by name, saying so plainly when there is no such thing.

        Raises:
            KeyError: When no village carries that name.
        """
        for village in self.villages:
            if village.name == name:
                return village
        raise KeyError(f"no village called {name!r}")

    def add_village(self, village: Village) -> None:
        """Bring a village into the world.

        Raises:
            ValueError: When a village of that name is already here.
        """
        if any(known.name == village.name for known in self.villages):
            raise ValueError(f"a village called {village.name!r} is already in the world")
        self.villages.append(village)

    def remove_village(self, name: str) -> Village:
        """Send a village away and hand it back.

        Raises:
            KeyError: When no village carries that name.
            ValueError: When it would leave the world with nobody in it.
        """
        leaving = self.village(name)
        if len(self.villages) == 1:
            raise ValueError("the last village cannot leave, there would be nobody")
        self.villages.remove(leaving)
        return leaving

    def pairs(self) -> list[tuple[Village, Village]]:
        """Every unordered pair of villages, in a stable order."""
        return list(itertools.combinations(self.villages, 2))

    def advance_day(self) -> DayReport:
        """Simulate one day: whatever was due, production, consumption, learning, barter.

        Anything due today lands before anybody produces, so an event changes the day
        it was promised for. ``time_per_day`` caps the barters, so not every pair meets.
        """
        self._check_rules()
        self.day += 1
        self._run_due_events()
        production: dict[str, dict[Goods, float]] = {}
        consumption: dict[str, dict[Goods, float]] = {}
        unmet: dict[str, dict[Goods, float]] = {}
        for village in self.villages:
            record: DayRecord = village.produce_and_consume()
            production[village.name] = record.produced
            consumption[village.name] = record.consumed
            unmet[village.name] = record.unmet
        for village in self.villages:
            village.observe_day()

        trades = []
        hours_left = self.time_per_day
        for first, second in self._day_order():
            if hours_left == 0:
                break
            if not self._will_meet(first, second):
                continue
            proposer, responder = self._openers(first, second)
            result = negotiate(proposer, responder, max_rounds=self.max_rounds)
            trades.append(TradeEvent(proposer.name, responder.name, result))
            hours_left -= 1

        return DayReport(
            self.day,
            production,
            consumption,
            unmet,
            tuple(trades),
            hours_left,
            self.time_per_day,
        )

    def _run_due_events(self) -> None:
        """Fire everything waiting for today, in the order it was arranged.

        Nothing here draws from the world random stream, so a day with an event on
        it still hands out the same meetings as a day without one.
        """
        due = [event for event in self.scheduled if event.fires_today(self.day)]
        for event in due:
            event.apply(self)
            self.scheduled.remove(event)
            self.fired.append(event)

    def _check_rules(self) -> None:
        if not 0.0 <= self.trade_chance <= 1.0:
            raise ValueError(
                f"the meeting chance must sit between 0 and 1, got {self.trade_chance}"
            )
        if self.time_per_day < 0:
            raise ValueError(
                f"a day cannot have negative barters, got {self.time_per_day}"
            )

    def _day_order(self) -> list[tuple[Village, Village]]:
        """The pairs of the day, the ones in need of each other first.

        A pair where somebody runs short is served before a pair that merely has
        something spare, so a village that needs help is served first. The rest are
        shuffled, which keeps the calendar from settling into a fixed pattern.
        """
        urgent = []
        casual = []
        for pair in self.pairs():
            (urgent if self._in_need(*pair) else casual).append(pair)
        self._rng.shuffle(casual)
        return urgent + casual

    def _in_need(self, first: Village, second: Village) -> bool:
        """Whether either of the two is running short of something."""
        return any(
            village.shortfall(goods) > 0
            for village in (first, second)
            for goods in village.stock
        )

    def _will_meet(self, first: Village, second: Village) -> bool:
        """Whether this pair actually gets together today."""
        if self._in_need(first, second):
            return True
        return self._rng.random() < self.trade_chance

    def _openers(self, first: Village, second: Village) -> tuple[Village, Village]:
        """Choose at random which of the two starts the conversation."""
        if self._rng.random() < 0.5:
            return first, second
        return second, first

    def stock_of(self, goods: Goods) -> dict[str, float]:
        """Stock of one commodity per village, keyed by village name."""
        return {village.name: village.stock[goods] for village in self.villages}

    def stock_spread(self) -> dict[Goods, float]:
        """Difference between the richest and the poorest village per commodity."""
        spread = {}
        for goods in ALL_GOODS:
            amounts = list(self.stock_of(goods).values())
            spread[goods] = max(amounts) - min(amounts) if amounts else 0.0
        return spread

    def satisfaction(self) -> dict[str, float]:
        """Share of needs currently covered per village."""
        return {village.name: village.satisfaction() for village in self.villages}

    def satisfaction_spread(self) -> float:
        """Difference between the best fed and the worst fed village."""
        values = list(self.satisfaction().values())
        if not values:
            return 0.0
        return max(values) - min(values)


def build_default_world(seed: int = 0, production_noise: float = 0.0) -> World:
    """Three equally sized villages, each specialised in a different commodity.

    Every village produces exactly what the three of them consume, so the world
    runs without external input and the barters only move goods around.
    """
    names = {
        Goods.FARM: "Farmers",
        Goods.MINERAL: "Miners",
        Goods.CRAFT: "Artisans",
    }
    villages = []
    for index, specialty in enumerate(ALL_GOODS):
        production = {goods: SIDE_OUTPUT for goods in ALL_GOODS}
        production[specialty] = SPECIALISATION_OUTPUT
        villages.append(
            Village(
                name=names[specialty],
                population=DEFAULT_POPULATION,
                stock={goods: OPENING_STOCK for goods in ALL_GOODS},
                production=production,
                brain=NegotiationBrain(seed=seed + index),
                reserve_days=RESERVE_DAYS,
                production_noise=production_noise,
                seed=seed + index,
            )
        )
    return World(villages, seed=seed)