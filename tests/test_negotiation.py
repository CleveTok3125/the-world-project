from __future__ import annotations

import math
from dataclasses import dataclass, field

import pytest

from world.agent import NegotiationBrain, Observation
from world.goods import ALL_GOODS, Goods, zero_stock
from world.negotiation import MIN_TRADE_QUANTITY, negotiate
from world.village import TradeResult, Village


@dataclass
class ScriptedBrain(NegotiationBrain):
    """A brain whose reservation prices are dictated by the test."""

    prices: dict[Goods, float] = field(default_factory=dict)

    def reservation_price(self, goods: Goods, observation: Observation) -> float:
        return self.prices.get(goods, goods.base_value)


def make_village(
    name: str,
    stock: dict[Goods, float],
    brain: NegotiationBrain | None = None,
    population: int = 10,
) -> Village:
    return Village(
        name=name,
        population=population,
        stock=stock,
        production=zero_stock(),
        brain=brain if brain is not None else NegotiationBrain(seed=0),
    )


def stocked(name: str, good: Goods, amount: float) -> Village:
    """A village holding a surplus of a single commodity."""
    return make_village(name, {**zero_stock(), good: amount})


class TestNegotiationWithoutASurplus:
    def test_two_empty_villages_cannot_reach_an_agreement(self) -> None:
        first = make_village("A", zero_stock())
        second = make_village("B", zero_stock())

        result = negotiate(first, second)

        assert not result
        assert not result.agreed
        assert result.trade is None
        assert result.rounds == ()
        assert "nothing to spare" == result.reason

    def test_a_village_is_never_its_own_counterpart(self) -> None:
        lonely = stocked("A", Goods.FARM, 100.0)

        result = negotiate(lonely, lonely)

        assert not result.agreed
        assert result.reason == "same village"

    def test_a_single_commodity_cannot_be_exchanged_for_itself(self) -> None:
        first = stocked("A", Goods.FARM, 100.0)
        second = stocked("B", Goods.FARM, 100.0)

        result = negotiate(first, second)

        assert not result.agreed
        assert result.reason == "only the same commodity to spare"

    def test_a_second_commodity_saves_a_village_that_only_holds_one(self) -> None:
        first = make_village(
            "A", {**zero_stock(), Goods.FARM: 100.0, Goods.CRAFT: 40.0}
        )
        second = stocked("B", Goods.CRAFT, 100.0)

        result = negotiate(first, second)

        assert result.agreed
        assert result.rounds[-1].offer.offer_goods is Goods.FARM


class TestNegotiationOutcome:
    def setup_method(self) -> None:
        self.proposer = stocked("A", Goods.FARM, 100.0)
        self.responder = stocked("B", Goods.CRAFT, 100.0)

    def test_a_typical_pair_agrees_on_the_first_round(self) -> None:
        result = negotiate(self.proposer, self.responder)

        assert result.agreed
        assert result.trade is not None and result.trade.accepted
        assert result.rounds[0].round_index == 1
        assert result.rounds[0].response == "accepted"

    def test_the_transcript_shows_a_counter_offer_before_the_settlement(self) -> None:
        self.proposer.brain = ScriptedBrain(prices={Goods.FARM: 1.0, Goods.CRAFT: 5.0})
        self.responder.brain = ScriptedBrain(prices={Goods.CRAFT: 2.5, Goods.FARM: 0.6})

        result = negotiate(self.proposer, self.responder)

        assert result.agreed
        assert [item.response for item in result.rounds] == ["counter-offer", "accepted"]

    def test_the_settled_rate_matches_the_reference_value_of_the_two_commodities(
        self,
    ) -> None:
        result = negotiate(self.proposer, self.responder)

        assert result.agreed
        assert result.rate == pytest.approx(2.5)
        assert result.responder_limit <= result.rate <= result.proposer_limit

    def test_the_settled_rate_is_the_geometric_middle_of_the_two_limits(self) -> None:
        self.proposer.brain = ScriptedBrain(prices={Goods.FARM: 1.0, Goods.CRAFT: 5.0})
        self.responder.brain = ScriptedBrain(prices={Goods.CRAFT: 2.5, Goods.FARM: 0.6})

        result = negotiate(self.proposer, self.responder)

        assert result.agreed
        assert result.proposer_limit == pytest.approx(5.0)
        assert result.responder_limit == pytest.approx(25.0 / 6.0)
        assert result.rate == pytest.approx(math.sqrt(5.0 * 25.0 / 6.0))
        assert result.responder_limit <= result.rate <= result.proposer_limit

    def test_the_trade_moves_goods_from_both_sides(self) -> None:
        negotiate(self.proposer, self.responder)

        assert self.proposer.stock[Goods.FARM] == pytest.approx(68.0)
        assert self.proposer.stock[Goods.CRAFT] == pytest.approx(80.0)
        assert self.responder.stock[Goods.CRAFT] == pytest.approx(20.0)
        assert self.responder.stock[Goods.FARM] == pytest.approx(32.0)

    def test_the_accepted_offer_is_recorded_from_the_proposer_point_of_view(self) -> None:
        result = negotiate(self.proposer, self.responder)

        assert result.agreed
        final = result.rounds[-1]
        assert final.offer.offer_goods is Goods.FARM
        assert final.offer.want_goods is Goods.CRAFT
        assert final.offer.offer_quantity == pytest.approx(32.0)
        assert final.offer.want_quantity == pytest.approx(final.offer.offer_quantity * 2.5)

    def test_the_executed_trade_holds_the_offers_both_sides_wrote(self) -> None:
        result = negotiate(self.proposer, self.responder)

        assert result.trade is not None
        assert result.trade.offer == result.rounds[-1].offer

    def test_no_village_is_left_with_a_negative_stock(self) -> None:
        negotiate(self.proposer, self.responder)

        for village in (self.proposer, self.responder):
            for goods in ALL_GOODS:
                assert village.stock[goods] >= 0.0

    def test_a_single_round_is_enough_when_the_opening_rate_is_acceptable(self) -> None:
        self.responder.brain = ScriptedBrain(prices={Goods.CRAFT: 1.0})

        result = negotiate(self.proposer, self.responder)

        assert result.agreed
        assert len(result.rounds) == 1

    def test_a_counter_offer_needs_a_second_round(self) -> None:
        self.responder.brain = ScriptedBrain(prices={Goods.CRAFT: 5.0, Goods.FARM: 1.0})

        result = negotiate(self.proposer, self.responder, max_rounds=1)

        assert not result.agreed
        assert result.trade is None
        assert result.rounds[0].response == "counter-offer"
        assert result.reason == "no agreement reached"

    def test_the_counterparty_can_start_the_negotiation_instead(self) -> None:
        result = negotiate(self.responder, self.proposer)

        assert result.agreed
        assert result.rounds[-1].offer.offer_goods is Goods.CRAFT
        assert result.rounds[-1].offer.offer_quantity == pytest.approx(80.0)
        assert result.rate == pytest.approx(0.4)


class TestNegotiationLimits:
    def test_a_reservation_gap_that_cannot_be_bridged_ends_the_talk(self) -> None:
        proposer = stocked("A", Goods.FARM, 100.0)
        responder = stocked("B", Goods.CRAFT, 100.0)
        responder.brain = ScriptedBrain(prices={Goods.CRAFT: 100.0, Goods.FARM: 0.25})

        result = negotiate(proposer, responder)

        assert not result.agreed
        assert result.reason == "prices too far apart"
        assert result.trade is None
        assert result.rounds == ()

    def test_a_quantity_below_the_minimum_trade_size_is_declined(self) -> None:
        proposer = stocked("A", Goods.FARM, 20.0 + MIN_TRADE_QUANTITY / 2)
        responder = stocked("B", Goods.CRAFT, 100.0)
        responder.brain = ScriptedBrain(prices={Goods.CRAFT: 1.0})

        result = negotiate(proposer, responder)

        assert not result.agreed
        assert result.reason == "quantity too small"

    def test_an_accepted_offer_refused_by_the_village_ends_the_talk(self) -> None:
        proposer = stocked("A", Goods.FARM, 100.0)
        responder = stocked("B", Goods.CRAFT, 100.0)

        def refuse(*_: object) -> TradeResult:
            return TradeResult(False, "not enough Agriculture in stock", None)

        result = negotiate(proposer, responder, execute=refuse)

        assert not result.agreed
        assert result.trade is not None
        assert not result.trade.accepted
        assert "stock" in result.reason

    def test_the_trade_is_limited_by_the_smaller_surplus(self) -> None:
        proposer = stocked("A", Goods.FARM, 1_000.0)
        responder = stocked("B", Goods.CRAFT, 50.0)
        responder.brain = ScriptedBrain(prices={Goods.CRAFT: 2.5})

        result = negotiate(proposer, responder)

        assert result.agreed
        assert result.rounds[-1].offer.want_quantity == pytest.approx(30.0)
        assert result.rounds[-1].offer.offer_quantity == pytest.approx(30.0 / result.rate)
        assert responder.stock[Goods.CRAFT] == pytest.approx(20.0)

    def test_the_trade_is_limited_by_the_smaller_surplus_after_the_sides_are_swapped(
        self,
    ) -> None:
        proposer = stocked("A", Goods.FARM, 30.0)
        responder = stocked("B", Goods.CRAFT, 1_000.0)

        result = negotiate(proposer, responder)

        assert result.agreed
        assert result.rounds[-1].offer.offer_quantity == pytest.approx(10.0)
        assert result.rounds[-1].offer.want_quantity == pytest.approx(25.0)

    def test_nothing_is_traded_beyond_the_target_stock(self) -> None:
        proposer = stocked("A", Goods.FARM, 20.0)
        responder = stocked("B", Goods.CRAFT, 1_000.0)

        negotiate(proposer, responder)

        assert proposer.stock[Goods.FARM] == pytest.approx(20.0)