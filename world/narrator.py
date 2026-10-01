"""The narrator: turns the state of the world into a short story.

The narrator only ever sees a :class:`WorldSummary`, which holds what a story
actually needs and nothing more. Writing it is done by a language model, so a day
with no model gets no story rather than a badly written one.

Changing the world is :mod:`world.director`'s job, not this module's.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

from world.goods import ALL_GOODS, Goods
from world.llm import LLMClient
from world.village import Village
from world.world import DayReport, World

NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


@dataclass(frozen=True)
class VillageDigest:
    """What a story needs to know about one village.

    Attributes:
        name: Name of the village.
        satisfied: Share of needs currently covered, the maximum being the sum of
            the commodity values.
        missing: Units of every commodity the village fell short on that day.
        deprived: Share of the village's daily needs that went unmet, from 0 upward.
        shortest: The commodity it fell furthest short on, ``None`` when it ate
            everything it needed.
        richest: The commodity it holds furthest above its target, ``None`` when it
            holds nothing beyond what it needs.
        surplus: Units of ``richest`` it holds above its target.
    """

    name: str
    satisfied: float
    missing: float
    deprived: float = 0.0
    shortest: Goods | None = None
    richest: Goods | None = None
    surplus: float = 0.0


@dataclass(frozen=True)
class TradeDigest:
    """One barter, reduced to the numbers a story can quote.

    Attributes:
        proposer: Name of the village that opened the conversation.
        responder: Name of the village that answered it.
        agreed: Whether the goods actually changed hands.
        given: Units of ``given_goods`` handed over, zero when refused.
        given_goods: The commodity handed over, ``None`` when refused.
        taken: Units of ``taken_goods`` received, zero when refused.
        taken_goods: The commodity received, ``None`` when refused.
        rate: Units of the received commodity per unit of the given one.
        reason: Why no deal was reached, empty when the barter succeeded.
    """

    proposer: str
    responder: str
    agreed: bool
    given: float
    given_goods: Goods | None
    taken: float
    taken_goods: Goods | None
    rate: float
    reason: str = ""

    def sentence(self) -> str:
        """The barter as one unambiguous sentence, directions included."""
        if not self.agreed or self.given_goods is None or self.taken_goods is None:
            reason = self.reason or "no agreement"
            return (
                f"{self.proposer} and {self.responder} could not agree, "
                f"because {reason}."
            )
        return (
            f"{self.proposer} gave {self.responder} {self.given:.1f} "
            f"{self.given_goods.label} and received {self.taken:.1f} "
            f"{self.taken_goods.label}."
        )


@dataclass(frozen=True)
class ShortageDigest:
    """A village running short of specific commodities.

    Attributes:
        name: Name of the village.
        goods: The commodities it could not obtain, mapped to how many units.
    """

    name: str
    goods: dict[Goods, float]


@dataclass(frozen=True)
class Situation:
    """How the day went, in one line.

    Several villages short of one commodity is a shortage of the world rather than of
    one village, and is reported as a group.

    Attributes:
        key: One of ``calm``, ``shortage``, ``severe``, ``shared`` or ``surplus``.
        villages: Names of the villages the verdict is about, worst first.
        commodity: The commodity at the heart of the verdict, if any.
        severity: How badly it went, from 0 upward; the deprivation share for a
            shortage and the pile-up share for a surplus.
    """

    key: str
    villages: tuple[str, ...]
    commodity: Goods | None
    severity: float

    @property
    def label(self) -> str:
        """A short phrase naming the situation, ready to drop into a sentence."""
        if self.key == "shared":
            assert self.commodity is not None
            return f"a shortage of {self.commodity.label} shared by {_names(self.villages)}"
        if self.key == "severe":
            assert self.commodity is not None
            return f"a severe shortage of {self.commodity.label}"
        if self.key == "shortage":
            assert self.commodity is not None
            return f"a shortage of {self.commodity.label}"
        if self.key == "surplus":
            assert self.commodity is not None
            if len(self.villages) == 1:
                return f"a surplus of {self.commodity.label} held by {self.villages[0]}"
            return (
                f"a surplus of {self.commodity.label} "
                f"held by {_names(self.villages)}"
            )
        return "the world in balance"


SEVERE_DEPRIVATION = 0.25
SURPLUS_PILE_UP = 4.0


def _names(names: tuple[str, ...]) -> str:
    """Two names joined with 'and', more joined with commas."""
    if len(names) < 2:
        return names[0]
    return f"{', '.join(names[:-1])} and {names[-1]}"


def situation(summary: WorldSummary) -> Situation:
    """Classify the day so a narrator can close on it in one line.

    One village running out is its own bad luck; two short of the same commodity is
    a shortage the world cannot trade its way out of, so the verdict names them
    together. A pile-up only gets a turn when nobody is short, since goods already
    in the stores are a problem the world has solved by other means.
    """
    hungry = [village for village in summary.villages if village.missing > 0]
    if hungry:
        worst = max(hungry, key=lambda village: village.missing)
        shared = tuple(
            village.name
            for village in hungry
            if village.shortest is not None and village.shortest is worst.shortest
        )
        if len(shared) > 1:
            return Situation(
                key="shared",
                villages=shared,
                commodity=worst.shortest,
                severity=worst.deprived,
            )
        return Situation(
            key="severe" if worst.deprived >= SEVERE_DEPRIVATION else "shortage",
            villages=(worst.name,),
            commodity=worst.shortest,
            severity=worst.deprived,
        )
    if summary.pile_up is not None:
        return Situation(
            key="surplus",
            villages=summary.pile_up.holders,
            commodity=summary.pile_up.goods,
            severity=summary.pile_up.days,
        )
    return Situation(key="calm", villages=(), commodity=None, severity=0.0)


@dataclass(frozen=True)
class WorldSummary:
    """Everything a narrator is told about the world, and nothing more.

    Attributes:
        day: The day number that has just been completed.
        spread: How far the best fed village is ahead of the worst fed one.
        villages: One digest per village.
        trades: The barters that took place, in the order they happened.
        shortages: The villages that could not meet their own needs.
        pile_up: The commodity that has piled up past usefulness, if any has.
    """

    day: int
    spread: float
    villages: tuple[VillageDigest, ...]
    trades: tuple[TradeDigest, ...]
    shortages: tuple[ShortageDigest, ...] = ()
    pile_up: PileUpDigest | None = None

    def worst_off(self) -> VillageDigest | None:
        """The village with the largest shortfall, if any village has one."""
        hungry = [village for village in self.villages if village.missing > 0]
        return max(hungry, key=lambda village: village.missing, default=None)

    def biggest_trade(self) -> TradeDigest | None:
        """The agreed barter that moved the most goods, by total volume."""
        agreed = [trade for trade in self.trades if trade.agreed]
        return max(agreed, key=lambda trade: trade.given + trade.taken, default=None)

    def refusals(self) -> tuple[TradeDigest, ...]:
        """The barters that never reached an agreement."""
        return tuple(trade for trade in self.trades if not trade.agreed)

    def to_payload(self) -> dict[str, object]:
        """A plain dictionary, ready to be handed to a language model.

        The trades are written out as whole sentences rather than as field names,
        because a small model reads a finished sentence far more reliably than it
        reads a pair of quantities it has to reassemble.
        """
        verdict = situation(self)
        return {
            "day": self.day,
            "situation": verdict.label,
            "severity": round(verdict.severity, 2),
            "spread_between_villages": round(self.spread, 2),
            "villages": [
                f"{village.name} has {village.satisfied:.1f} of 5.0 needs met"
                + (f", missing {village.missing:.1f} units." if village.missing else ".")
                for village in self.villages
            ],
            "trades": [trade.sentence() for trade in self.trades],
            "shortages": [
                f"{shortage.name} went without "
                + ", ".join(
                    f"{amount:.1f} {goods.label}"
                    for goods, amount in shortage.goods.items()
                )
                for shortage in self.shortages
            ],
        }


def _deprived_of(village: Village, unmet: dict[Goods, float]) -> float:
    """Share of what the village needed today that it did not get, by value.

    Counting what is left in the stores instead of what went missing would call a
    village that fell half a unit short at the very end of its reserves as badly
    deprived as one that starved all day.
    """
    need = sum(
        goods.base_value * village.daily_consumption(goods) for goods in ALL_GOODS
    )
    if need <= 0:
        return 0.0
    lost = sum(goods.base_value * unmet[goods] for goods in unmet)
    return lost / need


@dataclass(frozen=True)
class PileUpDigest:
    """More of a commodity sitting in stores than the world can spend.

    The measure spans every village, because the same units spread thinly are a pile
    the world cannot move, while the same units in one granary are just a village
    that made more than it needed.

    Attributes:
        goods: The commodity that has piled up.
        days: Days of world consumption it adds up to.
        holders: Names of the villages holding it above their own target.
    """

    goods: Goods
    days: float
    holders: tuple[str, ...]


def _richest_of(village: Village) -> Goods | None:
    """The commodity a village holds furthest above its target, dearest first on a tie."""
    worst = max(
        ALL_GOODS,
        key=lambda goods: (village.excess(goods), goods.base_value),
    )
    return worst if village.excess(worst) > 0 else None


def _pile_up_of(world: World) -> PileUpDigest | None:
    """Find a commodity sitting far beyond what the world can spend, if one does.

    Only units above a village's own target count, so a stack that merely replaces
    what the world has already used is not a pile.
    """
    worst: PileUpDigest | None = None
    for goods in ALL_GOODS:
        eaten = sum(village.daily_consumption(goods) for village in world.villages)
        if eaten <= 0:
            continue
        spare = sum(village.excess(goods) for village in world.villages)
        days = spare / eaten
        if days < SURPLUS_PILE_UP:
            continue
        holders = tuple(
            village.name for village in world.villages if village.excess(goods) > 0
        )
        if worst is None or days > worst.days:
            worst = PileUpDigest(goods=goods, days=days, holders=holders)
    return worst


def _shortest_of(unmet: dict[Goods, float]) -> Goods | None:
    """The commodity a village fell furthest short on, dearest first on a tie."""
    worst = max(
        unmet,
        key=lambda goods: (unmet[goods], goods.base_value),
        default=None,
    )
    if worst is None or unmet[worst] <= 0:
        return None
    return worst


def summarize(world: World, report: DayReport) -> WorldSummary:
    """Reduce a simulated day to the handful of facts a story needs.

    Shortage is the one recorded when a village drew on its stores, not whatever was
    still missing at the end of the day, so the closing verdict agrees with the lines
    the narrator already wrote about the same shortage.
    """
    digests = []
    for village in world.villages:
        richest = _richest_of(village)
        digests.append(
            VillageDigest(
                name=village.name,
                satisfied=village.satisfaction(),
                missing=sum(report.unmet[village.name].values()),
                deprived=_deprived_of(village, report.unmet[village.name]),
                shortest=_shortest_of(report.unmet[village.name]),
                richest=richest,
                surplus=village.excess(richest) if richest is not None else 0.0,
            )
        )
    villages = tuple(digests)

    trades = []
    for event in report.trades:
        trade = event.result.trade
        offer = trade.offer if trade is not None else None
        if event.result.agreed and offer is not None:
            trades.append(
                TradeDigest(
                    proposer=event.proposer,
                    responder=event.responder,
                    agreed=True,
                    given=offer.offer_quantity,
                    given_goods=offer.offer_goods,
                    taken=offer.want_quantity,
                    taken_goods=offer.want_goods,
                    rate=event.result.rate,
                )
            )
        else:
            trades.append(
                TradeDigest(
                    proposer=event.proposer,
                    responder=event.responder,
                    agreed=False,
                    given=0.0,
                    given_goods=None,
                    taken=0.0,
                    taken_goods=None,
                    rate=event.result.rate,
                    reason=event.result.reason,
                )
            )
    shortages = tuple(
        ShortageDigest(
            name=name,
            goods={goods: amount for goods, amount in unmet.items() if amount > 0},
        )
        for name, unmet in report.unmet.items()
        if any(amount > 0 for amount in unmet.values())
    )
    return WorldSummary(
        day=report.day,
        spread=world.satisfaction_spread(),
        villages=villages,
        trades=tuple(trades),
        shortages=shortages,
        pile_up=_pile_up_of(world),
    )


class Narrator(Protocol):
    """Anything able to turn a summary into prose."""

    def describe(self, summary: WorldSummary) -> list[str]:
        """Write the lines that tell what happened on the summarized day."""
        ...


@dataclass
class LLMNarrator:
    """A narrator backed by a large language model.

    There is no template writer behind this any more. A model that loses a quantity
    or renames a commodity has written something unusable, so the reply is refused
    rather than quietly replaced with something that was never going to be good.

    Attributes:
        client: The model to write with.
    """

    client: LLMClient

    SYSTEM_PROMPT = (
        "You are the narrator of a small barter economy with three villages.\n"
        "Rewrite the day given to you as flowing prose of two to four sentences.\n"
        "Rules:\n"
        "- Never repeat the input sentences word for word, put them in your own words.\n"
        "- Keep every direction of every trade exactly as given: whoever gave a "
        "good is the one who received the other good.\n"
        "- Write every quantity as the digits you are given, so 20.0 stays 20.0. "
        "Never spell a number out as a word.\n"
        "- Name the goods exactly as written, so Minerals stays Minerals and "
        "never becomes ores, rock or metal.\n"
        "- Use only the villages, goods and numbers you are given, invent nothing.\n"
        "- A village that went without something did not receive it. Never turn a "
        "shortage into a delivery or a store into a gain.\n"
        "- Close with one final short sentence giving your verdict on the day, "
        "based on the situation line and in the same wording where you can.\n"
        "- The severity number under the situation is the share of what the world "
        "needed that went unmet, or the days of use held in stores when the line "
        "names a surplus. Weigh your closing sentence by that number yourself: 0.05 "
        "is a bad morning, 0.5 is a famine, 0.9 is a catastrophe. Never call a "
        "crisis a shortage or a catastrophe merely a shortage.\n"
        "- Reply with prose only. No bullet points, no headings, no quotation marks."
    )

    def describe(self, summary: WorldSummary) -> list[str]:
        """Ask the model to write the story of the summarized day."""
        answer = self.client.ask(
            self.SYSTEM_PROMPT, json.dumps(summary.to_payload(), indent=2)
        )
        lines = _as_lines(answer)
        if not keeps_the_figures(lines, summary):
            raise LostTheFacts(answer.strip()[:200])
        return lines


class LostTheFacts(RuntimeError):
    """The reply dropped a quantity or a commodity name, so it cannot be used."""


def build_narrator(base_url: str = "") -> Narrator:
    """The narrator every run uses: a language model, with no template behind it."""
    return LLMNarrator(client=LLMClient.from_environment(base_url))


def _as_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def keeps_the_figures(lines: Sequence[str], summary: WorldSummary) -> bool:
    """Whether a reply still carries every quantity and commodity name it was given.

    A language model likes to turn ``20.0`` into ``twenty`` and ``Minerals`` into
    ``ores``. The numbers are facts, so a reply that lost them is not usable.
    """
    text = " ".join(lines)
    numbers = {round(value, 1) for value in _written_numbers(text)}
    for trade in summary.trades:
        if not trade.agreed or trade.given_goods is None or trade.taken_goods is None:
            continue
        for quantity, goods in (
            (trade.given, trade.given_goods),
            (trade.taken, trade.taken_goods),
        ):
            if round(quantity, 1) not in numbers:
                return False
            if goods.label not in text:
                return False
    return True


def _written_numbers(text: str) -> Iterator[float]:
    for token in NUMBER_PATTERN.findall(text):
        try:
            yield float(token)
        except ValueError:
            continue


def _reply(body: object) -> str:
    """Pull the answer out of an OpenAI chat completion response.

    A reasoning model fills ``reasoning_content`` and may leave ``content`` empty,
    so the two are stitched together when the answer is missing.
    """
    if not isinstance(body, dict):
        return ""
    choices = body.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    return content or message.get("reasoning_content") or ""


__all__ = [
    "LLMNarrator",
    "LostTheFacts",
    "Narrator",
    "PileUpDigest",
    "ShortageDigest",
    "Situation",
    "TradeDigest",
    "VillageDigest",
    "WorldSummary",
    "build_narrator",
    "keeps_the_figures",
    "situation",
    "summarize",
]
