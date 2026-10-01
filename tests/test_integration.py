"""End to end checks that the whole world reaches an even distribution."""

from __future__ import annotations

import pytest

from world.goods import ALL_GOODS, Goods
from world.village import Village
from world.world import World, build_default_world

MAX_DAYS = 40
LONG_DAYS = 80
TAIL_DAYS = 20
BALANCE_TOLERANCE = 1e-9
TRANSIENT_TOLERANCE = 2.6
NOISY_TOLERANCE = 3.5
PERFECT_DAY_RATIO = 0.55
TAIL_MEAN_FLOOR = 4.5
WORST_STRETCH_FLOOR = 2.0


def run_days(days: int, seed: int = 0, noise: float = 0.0) -> World:
    world = build_default_world(seed=seed, production_noise=noise)
    for _ in range(days):
        world.advance_day()
    return world


class TestTheDefaultWorldBalances:
    def test_a_village_is_left_short_on_only_a_few_days(self) -> None:
        world = build_default_world()
        hungry = 0

        for _ in range(MAX_DAYS):
            report = world.advance_day()
            if any(amount > 0 for unmet in report.unmet.values() for amount in unmet.values()):
                hungry += 1

        assert hungry <= MAX_DAYS // 10

    def test_the_three_villages_stay_close_to_one_another_over_a_long_run(self) -> None:
        world = build_default_world()
        tail = []

        for day in range(LONG_DAYS):
            world.advance_day()
            if day >= LONG_DAYS - TAIL_DAYS:
                tail += list(world.satisfaction().values())

        assert sum(tail) / len(tail) >= TAIL_MEAN_FLOOR

    def test_the_spread_stays_zero_on_most_days_of_the_run(self) -> None:
        world = build_default_world()
        spreads = []

        for _ in range(MAX_DAYS):
            world.advance_day()
            spreads.append(world.satisfaction_spread())

        perfect = sum(1 for spread in spreads if spread <= BALANCE_TOLERANCE)
        assert perfect / len(spreads) >= PERFECT_DAY_RATIO

    def test_a_transient_spread_never_grows_beyond_one_commodity_price(self) -> None:
        world = build_default_world()
        spreads = []

        for _ in range(MAX_DAYS):
            world.advance_day()
            spreads.append(world.satisfaction_spread())

        assert max(spreads) <= TRANSIENT_TOLERANCE

    def test_no_village_ever_loses_more_than_half_its_needs(self) -> None:
        """Meetings are random, so a village can have a bad stretch now and then."""
        world = build_default_world()
        worst = 5.0

        for _ in range(LONG_DAYS):
            world.advance_day()
            worst = min(worst, min(world.satisfaction().values()))

        assert worst >= WORST_STRETCH_FLOOR

    def test_every_village_keeps_enough_to_survive_the_next_days(self) -> None:
        world = run_days(MAX_DAYS)

        for village in world.villages:
            for goods in ALL_GOODS:
                assert village.stock[goods] >= 0.0

    def test_the_run_is_reproducible(self) -> None:
        first = run_days(MAX_DAYS, seed=9)
        second = run_days(MAX_DAYS, seed=9)

        assert first.stock_spread() == pytest.approx(second.stock_spread())
        assert first.satisfaction_spread() == pytest.approx(second.satisfaction_spread())


class TestBalanceUnderNoise:
    def test_the_villages_still_balance_when_production_wavers(self) -> None:
        world = run_days(MAX_DAYS, seed=3, noise=0.2)

        assert world.satisfaction_spread() <= NOISY_TOLERANCE

    def test_production_noise_never_pushes_a_village_into_debt(self) -> None:
        world = run_days(MAX_DAYS, seed=3, noise=0.4)

        for village in world.villages:
            assert all(village.stock[goods] >= 0.0 for goods in ALL_GOODS)

    def test_no_village_goes_hungry_under_light_noise(self) -> None:
        world = run_days(MAX_DAYS, seed=9, noise=0.1)

        for village in world.villages:
            assert village.satisfaction() >= 4.0


class TestTotalResourcesAreConserved:
    def test_the_world_total_moves_only_by_production_and_consumption(self) -> None:
        world = build_default_world()

        for _ in range(MAX_DAYS):
            before = {
                goods: sum(village.stock[goods] for village in world.villages)
                for goods in ALL_GOODS
            }
            report = world.advance_day()
            produced = {
                goods: sum(day[goods] for day in report.production.values())
                for goods in ALL_GOODS
            }
            consumed = {
                goods: sum(day[goods] for day in report.consumption.values())
                for goods in ALL_GOODS
            }
            after = {
                goods: sum(village.stock[goods] for village in world.villages)
                for goods in ALL_GOODS
            }
            for goods in ALL_GOODS:
                assert after[goods] == pytest.approx(
                    before[goods] + produced[goods] - consumed[goods]
                )


class TestBartersHappenEveryDay:
    def test_at_least_one_barter_is_agreed_on_every_day(self) -> None:
        world = build_default_world()

        for _ in range(MAX_DAYS):
            report = world.advance_day()
            assert any(event.result.agreed for event in report.trades)

    def test_a_settled_rate_stays_inside_the_bounds_of_the_two_villages(self) -> None:
        world = build_default_world()

        for _ in range(MAX_DAYS):
            for event in world.advance_day().trades:
                result = event.result
                if not result.agreed:
                    continue
                assert result.responder_limit <= result.rate <= result.proposer_limit

    def test_every_agreed_barter_never_moves_a_negative_amount(self) -> None:
        world = build_default_world()

        for _ in range(MAX_DAYS):
            for event in world.advance_day().trades:
                trade = event.result.trade
                if trade is None or trade.offer is None or not trade.accepted:
                    continue
                assert trade.offer.offer_quantity > 0
                assert trade.offer.want_quantity > 0


class TestTheReserveProtectsSurvival:
    def test_a_barter_never_takes_a_village_below_its_reserve(self) -> None:
        poor = Village(
            "Poor",
            10,
            {goods: 60.0 for goods in ALL_GOODS},
            {goods: 0.0 for goods in ALL_GOODS},
        )
        rich = Village(
            "Rich",
            10,
            {goods: 500.0 for goods in ALL_GOODS},
            {goods: 0.0 for goods in ALL_GOODS},
        )
        world = World((poor, rich))
        reserve = {goods: poor.target_stock(goods) for goods in ALL_GOODS}

        report = world.advance_day()

        assert report.unmet["Poor"] == pytest.approx(
            {goods: 0.0 for goods in ALL_GOODS}
        )
        assert all(event.result.agreed for event in report.trades)
        for goods in ALL_GOODS:
            assert world.villages[0].stock[goods] >= reserve[goods] - 1e-9

    def test_a_village_without_surplus_cannot_start_a_barter(self) -> None:
        needy = Village(
            "Needy",
            10,
            {goods: 20.0 for goods in ALL_GOODS},
            {goods: 0.0 for goods in ALL_GOODS},
        )
        rich = Village(
            "Rich",
            10,
            {goods: 500.0 for goods in ALL_GOODS},
            {goods: 0.0 for goods in ALL_GOODS},
        )
        world = World((needy, rich))

        report = world.advance_day()

        assert all(not event.result.agreed for event in report.trades)
        assert report.trades[0].result.reason == "nothing to spare"

    def test_a_village_alone_produces_and_consumes_on_its_own(self) -> None:
        world = build_default_world()
        solo = World((world.villages[0],))

        report = solo.advance_day()

        assert report.trades == ()
        assert report.production["Farmers"][Goods.FARM] == pytest.approx(20.0)
        assert report.unmet["Farmers"][Goods.FARM] == pytest.approx(0.0)