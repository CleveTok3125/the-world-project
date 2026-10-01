from __future__ import annotations

import math

import pytest

from world.agent import NegotiationBrain
from world.goods import ALL_GOODS, Goods, zero_stock
from world.village import TradeOffer, TradeResult, Village

TRADE_EPSILON = 1e-9


def make_village(
    name: str = "Village A",
    population: int = 10,
    stock: dict[Goods, float] | None = None,
    production: dict[Goods, float] | None = None,
    **kwargs: object,
) -> Village:
    return Village(
        name=name,
        population=population,
        stock=stock if stock is not None else zero_stock(),
        production=production if production is not None else zero_stock(),
        **kwargs,  # type: ignore[arg-type]
    )


class TestTradeOffer:
    def test_swapped_exchanges_both_sides(self) -> None:
        offer = TradeOffer(Goods.FARM, 4.0, Goods.CRAFT, 10.0)

        mirrored = offer.swapped()

        assert mirrored.offer_goods is Goods.CRAFT
        assert mirrored.offer_quantity == 10.0
        assert mirrored.want_goods is Goods.FARM
        assert mirrored.want_quantity == 4.0

    def test_swapping_twice_returns_the_original(self) -> None:
        offer = TradeOffer(Goods.MINERAL, 2.5, Goods.CRAFT, 1.5)

        assert offer.swapped().swapped() == offer

    def test_a_trade_result_without_a_deal_is_falsy(self) -> None:
        assert not TradeResult(accepted=False, reason="rejected", offer=None)
        assert TradeResult(accepted=True, reason="ok", offer=None)


class TestVillageBasics:
    def test_a_new_village_is_given_a_share_for_each_of_its_people(self) -> None:
        assert make_village(population=12).daily_consumption(Goods.FARM) == 12.0

    def test_growing_a_village_does_not_silently_move_its_daily_use(self) -> None:
        """Consumption is a setting in its own right, not a shadow of population.

        Tying the two together would mean a director raising the population also
        raised what the village needs, which is a change nobody asked for. It is
        left to be set on its own.
        """
        village = make_village(population=10)
        before = village.daily_consumption(Goods.FARM)

        village.population = 20

        assert village.daily_consumption(Goods.FARM) == before

    def test_target_stock_reserves_several_days(self) -> None:
        village = make_village(population=10, reserve_days=2.0)

        assert village.target_stock(Goods.CRAFT) == 20.0

    def test_observation_reports_the_state_of_one_commodity(self) -> None:
        village = make_village(population=10, stock={**zero_stock(), Goods.FARM: 5.0})

        observation = village.observation(Goods.FARM)

        assert observation.stock == 5.0
        assert observation.target_stock == 20.0
        assert observation.stock_ratio == pytest.approx(0.25)
        assert observation.scarcity_label() == pytest.approx(0.75)

    def test_a_village_without_goods_can_offer_nothing_but_needs_something(self) -> None:
        village = make_village()

        assert village.most_surplus_good() is None
        assert village.scarcest_good() is not None

    def test_a_fully_stocked_village_needs_nothing(self) -> None:
        village = make_village(
            population=10,
            stock={goods: 10_000.0 for goods in ALL_GOODS},
        )

        assert village.most_surplus_good() is Goods.FARM
        assert village.scarcest_good() is None

    def test_most_surplus_good_returns_the_largest_excess(self) -> None:
        village = make_village(
            population=10,
            stock={**zero_stock(), Goods.FARM: 40.0, Goods.CRAFT: 25.0},
        )

        assert village.most_surplus_good() is Goods.FARM

    def test_scarcest_good_returns_the_largest_shortfall(self) -> None:
        village = make_village(
            population=10,
            stock={**zero_stock(), Goods.FARM: 5.0, Goods.MINERAL: 20.0, Goods.CRAFT: 10.0},
        )

        assert village.scarcest_good() is Goods.FARM

    def test_excess_and_shortfall_are_measured_against_the_target(self) -> None:
        village = make_village(
            population=10,
            stock={**zero_stock(), Goods.FARM: 30.0, Goods.CRAFT: 5.0},
        )

        assert village.excess(Goods.FARM) == pytest.approx(10.0)
        assert village.excess(Goods.CRAFT) == 0.0
        assert village.shortfall(Goods.FARM) == 0.0
        assert village.shortfall(Goods.CRAFT) == pytest.approx(15.0)

    def test_total_value_uses_base_prices(self) -> None:
        village = make_village(stock={**zero_stock(), Goods.FARM: 2.0, Goods.CRAFT: 1.0})

        assert village.total_value() == pytest.approx(2.0 + 2.5)

    def test_satisfaction_is_capped_at_one_per_commodity(self) -> None:
        full = make_village(
            population=10,
            stock={goods: 10_000.0 for goods in ALL_GOODS},
        )
        empty = make_village(population=10)

        assert full.satisfaction() == pytest.approx(sum(g.base_value for g in ALL_GOODS))
        assert empty.satisfaction() == 0.0

    def test_a_village_with_no_population_consumes_nothing(self) -> None:
        village = make_village(population=0, stock=zero_stock())

        assert village.target_stock(Goods.FARM) == 0.0
        assert village.satisfaction() == 0.0


class TestDailyProductionAndConsumption:
    def test_production_is_deterministic_without_noise(self) -> None:
        village = make_village(production={**zero_stock(), Goods.FARM: 7.0})

        village.produce()
        village.produce()

        assert village.stock[Goods.FARM] == pytest.approx(14.0)

    def test_noise_stays_inside_the_configured_band(self) -> None:
        village = make_village(
            production={**zero_stock(), Goods.FARM: 100.0},
            production_noise=0.2,
            seed=3,
        )

        village.produce()

        assert 80.0 <= village.stock[Goods.FARM] <= 120.0

    def test_the_same_seed_reproduces_the_same_production(self) -> None:
        production = {**zero_stock(), Goods.MINERAL: 50.0}
        first = make_village(production=production, production_noise=0.3, seed=11)
        second = make_village(production=production, production_noise=0.3, seed=11)

        first.produce()
        second.produce()

        assert first.stock == second.stock

    def test_production_never_exceeds_the_storage_limit(self) -> None:
        village = make_village(production={**zero_stock(), Goods.FARM: 100.0})

        village.stock[Goods.FARM] = Goods.FARM.storage_limit
        record = village.produce()

        assert village.stock[Goods.FARM] == pytest.approx(Goods.FARM.storage_limit)
        assert record.produced[Goods.FARM] == 0.0

    def test_consumption_stops_at_the_available_stock(self) -> None:
        village = make_village(population=10, stock={**zero_stock(), Goods.FARM: 3.0})

        record = village.consume()

        assert village.stock[Goods.FARM] == 0.0
        assert record.consumed[Goods.FARM] == pytest.approx(3.0)
        assert record.unmet[Goods.FARM] == pytest.approx(7.0)

    def test_using_more_of_a_commodity_makes_the_shortfall_worse(self) -> None:
        """The setting has to reach the day, not just sit in the record."""
        village = make_village(population=10, stock={**zero_stock(), Goods.FARM: 3.0})
        village.consumption[Goods.FARM] = 1.0
        before = village.consume().unmet[Goods.FARM]

        village.stock[Goods.FARM] = 3.0
        village.consumption[Goods.FARM] = 4.0
        after = village.consume().unmet[Goods.FARM]

        assert after > before

    def test_a_village_asked_to_use_more_reserves_more_stock(self) -> None:
        village = make_village(population=10, reserve_days=3.0)

        village.consumption[Goods.FARM] = 5.0

        assert village.target_stock(Goods.FARM) == 15.0

    def test_produce_and_consume_reports_both_halves_of_the_day(self) -> None:
        village = make_village(
            population=10,
            stock={**zero_stock(), Goods.FARM: 20.0},
            production={**zero_stock(), Goods.FARM: 5.0},
        )

        record = village.produce_and_consume()

        assert record.produced[Goods.FARM] == pytest.approx(5.0)
        assert record.consumed[Goods.FARM] == pytest.approx(10.0)
        assert record.unmet[Goods.CRAFT] == pytest.approx(10.0)
        assert village.stock[Goods.FARM] == pytest.approx(15.0)

    def test_brain_sees_every_commodity_every_day(self) -> None:
        brain = NegotiationBrain(seed=1)
        village = make_village(
            population=10,
            stock={**zero_stock(), Goods.FARM: 4.0},
            brain=brain,
        )

        predictions = village.observe_day()

        assert set(predictions) == set(ALL_GOODS)
        assert all(math.isfinite(value) for value in predictions.values())


class TestTrade:
    def setup_method(self) -> None:
        self.giver = make_village(
            name="Giver",
            population=10,
            stock={**zero_stock(), Goods.FARM: 30.0},
        )
        self.taker = make_village(
            name="Taker",
            population=10,
            stock={**zero_stock(), Goods.CRAFT: 20.0},
        )

    def test_a_accepted_deal_moves_the_goods(self) -> None:
        result = self.giver.trade(
            self.taker, TradeOffer(Goods.FARM, 6.0, Goods.CRAFT, 3.0)
        )

        assert result.accepted
        assert result.reason == "ok"
        assert result.offer is not None
        assert self.giver.stock[Goods.FARM] == pytest.approx(24.0)
        assert self.giver.stock[Goods.CRAFT] == pytest.approx(3.0)
        assert self.taker.stock[Goods.CRAFT] == pytest.approx(17.0)
        assert self.taker.stock[Goods.FARM] == pytest.approx(6.0)

    def test_trading_starts_from_either_side(self) -> None:
        self.taker.trade(self.giver, TradeOffer(Goods.CRAFT, 4.0, Goods.FARM, 8.0))

        assert self.taker.stock[Goods.CRAFT] == pytest.approx(16.0)
        assert self.giver.stock[Goods.FARM] == pytest.approx(22.0)

    def test_the_offer_is_exactly_the_whole_stock(self) -> None:
        result = self.giver.trade(
            self.taker, TradeOffer(Goods.FARM, 30.0, Goods.CRAFT, 12.0)
        )

        assert result.accepted
        assert self.giver.stock[Goods.FARM] == pytest.approx(0.0)

    def test_trading_a_larger_share_than_the_stock_is_refused(self) -> None:
        result = self.giver.trade(
            self.taker, TradeOffer(Goods.FARM, 30.5, Goods.CRAFT, 3.0)
        )

        assert not result.accepted
        assert result.reason.startswith("short of")
        assert self.giver.stock[Goods.FARM] == pytest.approx(30.0)

    def test_a_counterpart_without_the_commodity_blocks_the_whole_deal(self) -> None:
        offer = TradeOffer(Goods.FARM, 6.0, Goods.CRAFT, 25.0)

        result = self.giver.trade(self.taker, offer)

        assert not result.accepted
        assert result.reason == "the other side is short"
        assert self.giver.stock[Goods.FARM] == pytest.approx(30.0)
        assert self.taker.stock[Goods.CRAFT] == pytest.approx(20.0)

    def test_one_commodity_cannot_be_traded_for_itself(self) -> None:
        result = self.giver.trade(
            self.taker, TradeOffer(Goods.FARM, 5.0, Goods.FARM, 5.0)
        )

        assert not result.accepted
        assert result.reason == "same commodity both ways"

    @pytest.mark.parametrize("quantity", [0.0, -1.0, float("nan"), float("inf")])
    def test_a_quantity_that_is_not_a_positive_finite_number_is_refused(
        self, quantity: float
    ) -> None:
        result = self.giver.trade(
            self.taker, TradeOffer(Goods.FARM, quantity, Goods.CRAFT, 3.0)
        )

        assert not result.accepted
        assert result.reason == "quantity unusable"
        assert self.giver.stock[Goods.FARM] == pytest.approx(30.0)

    def test_a_village_cannot_trade_with_itself(self) -> None:
        result = self.giver.trade(
            self.giver, TradeOffer(Goods.FARM, 5.0, Goods.CRAFT, 1.0)
        )

        assert not result.accepted
        assert result.reason == "same village"

    def test_a_rejected_offer_still_reports_what_was_asked(self) -> None:
        offer = TradeOffer(Goods.FARM, 6.0, Goods.CRAFT, 99.0)

        result = self.giver.trade(self.taker, offer)

        assert not result.accepted
        assert result.offer == offer

    def test_a_village_with_an_empty_stock_cannot_trade(self) -> None:
        empty = make_village(name="Empty")

        result = empty.trade(self.taker, TradeOffer(Goods.FARM, 1.0, Goods.CRAFT, 1.0))

        assert not result.accepted
        assert self.taker.stock[Goods.FARM] == pytest.approx(0.0)
        assert result.offer is not None

    def test_a_floating_point_residue_is_still_tradable(self) -> None:
        self.giver.stock[Goods.FARM] = 0.1 + 0.2
        self.taker.stock[Goods.CRAFT] = 0.3

        result = self.giver.trade(
            self.taker, TradeOffer(Goods.FARM, 0.1 + 0.2, Goods.CRAFT, 0.3)
        )

        assert result.accepted
        assert self.giver.stock[Goods.FARM] == pytest.approx(0.0)
        assert self.taker.stock[Goods.CRAFT] == pytest.approx(0.0)
        assert self.giver.stock[Goods.FARM] >= -TRADE_EPSILON