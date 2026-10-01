"""Two villages bargaining over one commodity for another.

Each side knows only its own reservation price. The proposer opens at the reference
rate both sides can compute from the base values, the responder takes it or answers
with the lowest rate it can live with, and whatever is still contested settles in
the middle of the two.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from world.goods import ALL_GOODS, Goods
from world.village import TradeOffer, TradeResult, Village

MAX_ROUNDS = 3
MIN_TRADE_QUANTITY = 1e-6


@dataclass(frozen=True)
class NegotiationRound:
    """One offer put on the table and the answer it received.

    Attributes:
        round_index: Position of the offer in the conversation, starting at one.
        offer: The quantities proposed for that round.
        response: ``"accepted"`` or ``"counter-offer"``.
    """

    round_index: int
    offer: TradeOffer
    response: str


@dataclass(frozen=True)
class NegotiationResult:
    """Outcome of a negotiation, whether a deal was reached or not.

    Attributes:
        agreed: True when the goods were actually exchanged.
        reason: A short machine readable explanation.
        rounds: The conversation transcript, empty when no offer was ever made.
        trade: The executed trade, present as soon as an offer was agreed.
        rate: Units of the wanted commodity settled per unit of the offered one.
        proposer_limit: Highest rate the proposer can pay.
        responder_limit: Lowest rate the responder can accept.
    """

    agreed: bool
    reason: str
    rounds: tuple[NegotiationRound, ...]
    trade: TradeResult | None = None
    rate: float = 0.0
    proposer_limit: float = 0.0
    responder_limit: float = 0.0

    def __bool__(self) -> bool:
        return self.agreed


def surplus_goods(village: Village) -> list[Goods]:
    """Commodities the village holds above its target stock, richest surplus first."""
    ranked = [(goods, village.excess(goods)) for goods in ALL_GOODS]
    return [goods for goods, excess in sorted(ranked, key=lambda item: -item[1]) if excess > 0]


def intended_goods(
    proposer: Village, responder: Village, avoid: Sequence[Goods] = ()
) -> tuple[Goods, Goods] | None:
    """Which commodity each side would offer, before anyone negotiates.

    Args:
        proposer: The village that would open the conversation.
        responder: The village that would answer it.
        avoid: Commodities the caller refuses to trade this time round.

    Returns:
        The commodity the proposer would give and the one it would ask for, or
        ``None`` when the two have no acceptable pair of surplases between them.
    """
    banned = set(avoid)
    for offered in surplus_goods(proposer):
        if offered in banned:
            continue
        for wanted in surplus_goods(responder):
            if offered is not wanted:
                return offered, wanted
    return None


def negotiate(
    proposer: Village,
    responder: Village,
    max_rounds: int = MAX_ROUNDS,
    execute: Callable[[Village, Village, TradeOffer], TradeResult] | None = None,
) -> NegotiationResult:
    """Bargain a single barter between two villages.

    The traded quantity is capped by the surplus on both sides, so no village ever
    gives away what it needs to survive the coming days.

    Args:
        proposer: The village that opens the conversation.
        responder: The village that answers it.
        max_rounds: How many offers may be exchanged before giving up.
        execute: Performs the agreed exchange, the village method by default.

    Returns:
        A :class:`NegotiationResult` holding the transcript and the executed trade.
    """
    if proposer is responder:
        return NegotiationResult(False, "same village", ())

    intent = intended_goods(proposer, responder)
    if intent is None:
        return NegotiationResult(False, _no_deal_reason(proposer, responder), ())
    offered, wanted = intent

    proposer_limit = proposer.brain.fair_rate(
        offered, wanted, proposer.observation(offered), proposer.observation(wanted)
    )
    responder_limit = responder.brain.fair_rate(
        offered, wanted, responder.observation(offered), responder.observation(wanted)
    )
    if responder_limit > proposer_limit:
        return NegotiationResult(
            False, "prices too far apart", ()
        )
    if max_rounds < 1:
        return NegotiationResult(False, "no rounds left", ())

    settlement_rate = _settlement_rate(proposer_limit, responder_limit)
    opening_rate = wanted.base_value / offered.base_value
    opening_offer, opening_failure = _build_offer(
        proposer, responder, offered, wanted, opening_rate
    )
    settlement_offer, settlement_failure = _build_offer(
        proposer, responder, offered, wanted, settlement_rate
    )
    rounds: list[NegotiationRound] = []
    if opening_rate >= responder_limit:
        if settlement_offer is None:
            return NegotiationResult(
                False, settlement_failure, (), rate=settlement_rate
            )
        rounds.append(NegotiationRound(1, settlement_offer, "accepted"))
    else:
        if opening_offer is None:
            return NegotiationResult(False, opening_failure, (), rate=settlement_rate)
        rounds.append(NegotiationRound(1, opening_offer, "counter-offer"))
        if max_rounds < 2:
            return NegotiationResult(
                False,
                "no agreement reached",
                tuple(rounds),
                rate=settlement_rate,
            )
        if settlement_offer is None:
            return NegotiationResult(
                False, settlement_failure, tuple(rounds), rate=settlement_rate
            )
        rounds.append(NegotiationRound(2, settlement_offer, "accepted"))

    agreed_offer = rounds[-1].offer
    trade = (execute or _trade_villages)(proposer, responder, agreed_offer)
    return NegotiationResult(
        trade.accepted,
        trade.reason,
        tuple(rounds),
        trade=trade,
        rate=settlement_rate,
        proposer_limit=proposer_limit,
        responder_limit=responder_limit,
    )


def _trade_villages(proposer: Village, responder: Village, offer: TradeOffer) -> TradeResult:
    return proposer.trade(responder, offer)


def _settlement_rate(proposer_limit: float, responder_limit: float) -> float:
    """Split the contested rate in the geometric middle, the Nash bargaining point.

    The geometric mean of two positive rates is never worse for either side than
    the arithmetic mean of the same pair, and it collapses to the common rate when
    the two villages agree from the start.
    """
    return math.sqrt(proposer_limit * responder_limit)


def _no_deal_reason(proposer: Village, responder: Village) -> str:
    """Say why these two ran out of anything worth offering each other."""
    mine = surplus_goods(proposer)
    theirs = surplus_goods(responder)
    if not mine or not theirs:
        return "nothing to spare"
    if all(goods in set(theirs) for goods in mine):
        return "only the same commodity to spare"
    return "nothing worth trading"


def _build_offer(
    proposer: Village,
    responder: Village,
    offered: Goods,
    wanted: Goods,
    rate: float,
) -> tuple[TradeOffer | None, str]:
    """Size an offer at ``rate`` units of ``wanted`` per unit of ``offered``."""
    if rate <= 0:
        return None, "exchange rate unusable"
    give = min(proposer.excess(offered), responder.excess(wanted) / rate)
    if give < MIN_TRADE_QUANTITY:
        return None, "quantity too small"
    return TradeOffer(offered, give, wanted, give * rate), "ok"