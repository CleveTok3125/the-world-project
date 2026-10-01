"""A single village: its economy, its adaptive brain and the common trade method."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from world.agent import NegotiationBrain, Observation
from world.goods import ALL_GOODS, Goods

TRADE_EPSILON = 1e-9


@dataclass(frozen=True)
class TradeOffer:
    """One side of a barter, expressed from the point of view of the proposer.

    Attributes:
        offer_goods: The commodity the proposer gives away.
        offer_quantity: How many units of it the proposer gives away.
        want_goods: A different commodity the proposer asks for.
        want_quantity: How many units of that commodity the proposer wants.
    """

    offer_goods: Goods
    offer_quantity: float
    want_goods: Goods
    want_quantity: float

    def swapped(self) -> TradeOffer:
        """The same deal rewritten from the counterpart's point of view."""
        return TradeOffer(
            offer_goods=self.want_goods,
            offer_quantity=self.want_quantity,
            want_goods=self.offer_goods,
            want_quantity=self.offer_quantity,
        )


@dataclass(frozen=True)
class TradeResult:
    """Outcome of a trade attempt, rejected offers included.

    Attributes:
        accepted: True when the goods were actually exchanged.
        reason: A short machine readable explanation, ``"ok"`` when accepted.
        offer: The offer that was attempted, even when it was refused.
    """

    accepted: bool
    reason: str
    offer: TradeOffer | None

    def __bool__(self) -> bool:
        return self.accepted


@dataclass(frozen=True)
class DayRecord:
    """Production and consumption of one village over one day.

    Attributes:
        produced: Units of each commodity that actually entered the stock.
        consumed: Units of each commodity that were actually used.
        unmet: Units of each commodity the village wanted but could not obtain.
    """

    produced: dict[Goods, float]
    consumed: dict[Goods, float]
    unmet: dict[Goods, float]


@dataclass
class Village:
    """An economy that produces, consumes, learns and trades.

    Attributes:
        name: Display name of the village.
        population: Number of inhabitants, and nothing else it is counted for.
        stock: Units of each commodity currently held.
        production: Units of each commodity produced per day in ideal conditions.
        consumption: Units of each commodity one person uses per day. Every
            commodity is an input to work, so none of them is food to any one
            village.
        brain: The adaptive pricing model tuned to this village's own experience.
        reserve_days: Days of use kept aside before anything is traded.
        production_noise: Relative amplitude of the day to day production variation.
        seed: Makes the production variation reproducible.
    """

    name: str
    population: int
    stock: dict[Goods, float]
    production: dict[Goods, float]
    consumption: dict[Goods, float] = field(default_factory=dict)
    brain: NegotiationBrain = field(default_factory=NegotiationBrain)
    reserve_days: float = 2.0
    production_noise: float = 0.0
    seed: int = 0
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.consumption:
            self.consumption = {
                goods: self.population * goods.daily_consumption for goods in ALL_GOODS
            }
        self._rng = random.Random(self.seed)

    def daily_consumption(self, goods: Goods) -> float:
        """Units of ``goods`` the village needs per day."""
        return self.consumption[goods]

    def target_stock(self, goods: Goods) -> float:
        """Stock the village aims for, expressed in days of consumption."""
        return self.daily_consumption(goods) * self.reserve_days

    def observation(self, goods: Goods) -> Observation:
        """Snapshot of the village economy for one commodity."""
        return Observation(
            stock=self.stock[goods],
            target_stock=self.target_stock(goods),
            production=self.production[goods],
            consumption=self.daily_consumption(goods),
        )

    def excess(self, goods: Goods) -> float:
        """Units above the target stock, the amount safe to trade away."""
        return max(0.0, self.stock[goods] - self.target_stock(goods))

    def empty_stores(self) -> None:
        """Take every commodity down to nothing, granaries and all."""
        self.stock = {goods: 0.0 for goods in self.stock}

    def shortfall(self, goods: Goods) -> float:
        """Units missing to reach the target stock."""
        return max(0.0, self.target_stock(goods) - self.stock[goods])

    def most_surplus_good(self) -> Goods | None:
        """The commodity with the largest tradable surplus, or ``None``."""
        best: Goods | None = None
        best_excess = TRADE_EPSILON
        for goods in ALL_GOODS:
            excess = self.excess(goods)
            if excess > best_excess:
                best, best_excess = goods, excess
        return best

    def scarcest_good(self) -> Goods | None:
        """The commodity with the largest shortfall, or ``None`` when fully stocked."""
        best: Goods | None = None
        best_deficit = TRADE_EPSILON
        for goods in ALL_GOODS:
            deficit = self.shortfall(goods)
            if deficit > best_deficit:
                best, best_deficit = goods, deficit
        return best

    def total_value(self) -> float:
        """Worth of the whole stock at reference prices."""
        return sum(self.stock[goods] * goods.base_value for goods in ALL_GOODS)

    def satisfaction(self) -> float:
        """Weighted share of the needs that are currently covered, from 0 upward."""
        total = 0.0
        for goods in ALL_GOODS:
            target = self.target_stock(goods)
            if target <= 0:
                continue
            total += goods.base_value * min(1.0, self.stock[goods] / target)
        return total

    def produce(self) -> DayRecord:
        """Add one day of production, respecting storage limits and noise."""
        produced = {}
        for goods in ALL_GOODS:
            planned = self.production[goods]
            if planned and self.production_noise:
                planned *= 1.0 + self._rng.uniform(-self.production_noise, self.production_noise)
            planned = min(planned, goods.storage_limit - self.stock[goods])
            added = max(0.0, planned)
            self.stock[goods] += added
            produced[goods] = added
        return DayRecord(
            produced=produced,
            consumed={goods: 0.0 for goods in ALL_GOODS},
            unmet={goods: 0.0 for goods in ALL_GOODS},
        )

    def consume(self) -> DayRecord:
        """Use one day of consumption, limited to what is actually in stock."""
        consumed = {}
        unmet = {}
        for goods in ALL_GOODS:
            required = self.daily_consumption(goods)
            taken = min(required, self.stock[goods])
            self.stock[goods] -= taken
            consumed[goods] = taken
            unmet[goods] = required - taken
        return DayRecord(
            produced={goods: 0.0 for goods in ALL_GOODS},
            consumed=consumed,
            unmet=unmet,
        )

    def produce_and_consume(self) -> DayRecord:
        """Run one full production and consumption cycle."""
        produced = self.produce().produced
        consumption = self.consume()
        return DayRecord(
            produced=produced,
            consumed=consumption.consumed,
            unmet=consumption.unmet,
        )

    def observe_day(self) -> dict[Goods, float]:
        """Feed one observation per commodity to the brain and return the predictions."""
        return {goods: self.brain.observe(self.observation(goods)) for goods in ALL_GOODS}

    def trade(self, counterpart: Village, offer: TradeOffer) -> TradeResult:
        """Exchange exactly one commodity for a different one.

        The exchange is atomic: when any precondition fails nothing is moved.

        Args:
            counterpart: The village on the other side of the deal.
            offer: What this village gives and what it wants in return.

        Returns:
            A :class:`TradeResult` describing whether the goods changed hands.
        """
        if counterpart is self:
            return TradeResult(False, "same village", offer)
        if offer.offer_goods is offer.want_goods:
            return TradeResult(False, "same commodity both ways", offer)
        for quantity in (offer.offer_quantity, offer.want_quantity):
            if not math.isfinite(quantity) or quantity <= 0:
                return TradeResult(False, "quantity unusable", offer)
        if self.stock[offer.offer_goods] + TRADE_EPSILON < offer.offer_quantity:
            return TradeResult(False, f"short of {offer.offer_goods.label}", offer)
        if counterpart.stock[offer.want_goods] + TRADE_EPSILON < offer.want_quantity:
            return TradeResult(False, "the other side is short", offer)

        self.stock[offer.offer_goods] -= offer.offer_quantity
        self.stock[offer.want_goods] += offer.want_quantity
        counterpart.stock[offer.want_goods] -= offer.want_quantity
        counterpart.stock[offer.offer_goods] += offer.offer_quantity
        return TradeResult(True, "ok", offer)