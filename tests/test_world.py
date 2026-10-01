from __future__ import annotations

import pytest

from world.agent import NegotiationBrain
from world.goods import ALL_GOODS, Goods, zero_stock
from world.village import Village
from world.world import World, build_default_world

OPENING_STOCK = 30.0


def make_world(
    count: int = 3, noise: float = 0.0, seed: int = 0, **kwargs
) -> World:
    """A world where every village produces every commodity at the same rate."""
    villages = tuple(
        Village(
            name=f"Village {index}",
            population=10,
            stock={goods: OPENING_STOCK for goods in ALL_GOODS},
            production={goods: 10.0 for goods in ALL_GOODS},
            brain=NegotiationBrain(seed=seed + index),
            production_noise=noise,
            seed=seed + index,
        )
        for index in range(count)
    )
    return World(villages, seed=seed, **kwargs)


class TestWorldBasics:
    def test_a_new_world_has_not_advanced_yet(self) -> None:
        assert make_world().day == 0

    def test_a_world_needs_at_least_one_village(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            World(())

    def test_villages_must_have_distinct_names(self) -> None:
        first, second = make_world(2).villages

        second.name = first.name

        with pytest.raises(ValueError, match="unique"):
            World((first, second))

    def test_the_pairs_are_every_combination_of_villages(self) -> None:
        assert len(make_world(4).pairs()) == 6
        assert len(make_world(1).pairs()) == 0

    def test_a_pair_is_listed_only_once(self) -> None:
        world = build_default_world()

        for first, second in world.pairs():
            assert (second, first) not in world.pairs()

    def test_advancing_a_day_increments_the_counter(self) -> None:
        world = make_world()

        world.advance_day()
        world.advance_day()

        assert world.day == 2

    def test_every_village_reports_its_own_day_record(self) -> None:
        world = make_world()

        report = world.advance_day()

        assert set(report.production) == {"Village 0", "Village 1", "Village 2"}
        assert report.production["Village 0"][Goods.FARM] == pytest.approx(10.0)
        assert report.consumption["Village 0"][Goods.FARM] == pytest.approx(10.0)

    def test_needs_that_cannot_be_met_are_reported(self) -> None:
        world = make_world(1)
        world.villages[0].stock[Goods.CRAFT] = 2.0
        world.villages[0].production[Goods.CRAFT] = 0.0

        report = world.advance_day()

        assert report.unmet["Village 0"][Goods.CRAFT] == pytest.approx(8.0)
        assert report.consumption["Village 0"][Goods.CRAFT] == pytest.approx(2.0)
        assert report.unmet["Village 0"][Goods.FARM] == pytest.approx(0.0)

    def test_a_starved_village_reports_every_need_as_unmet(self) -> None:
        world = make_world(1)
        world.villages[0].stock = zero_stock()
        world.villages[0].production = zero_stock()

        report = world.advance_day()

        assert all(report.unmet["Village 0"][goods] == 10.0 for goods in ALL_GOODS)

    def test_the_brain_of_every_village_learns_every_day(self) -> None:
        world = make_world()

        before = [tuple(village.brain.weights) for village in world.villages]
        world.advance_day()

        for village, previous in zip(world.villages, before, strict=True):
            assert tuple(village.brain.weights) != previous

    def test_a_village_still_trades_after_being_stripped_of_everything(self) -> None:
        world = make_world(3)
        world.villages[0].stock = zero_stock()
        world.villages[0].production = zero_stock()

        report = world.advance_day()

        assert len(report.trades) == 3
        assert report.trades[0].result.agreed is False
        assert world.villages[0].stock == pytest.approx(zero_stock())


class TestWorldTrading:
    def test_no_pair_meets_twice_in_one_day(self) -> None:
        world = build_default_world()

        for _ in range(20):
            report = world.advance_day()

            assert len({(e.proposer, e.responder) for e in report.trades}) == len(report.trades)

    def test_the_days_hour_budget_bounds_the_barters(self) -> None:
        world = build_default_world()

        for _ in range(20):
            report = world.advance_day()

            assert len(report.trades) <= world.time_per_day

    def test_a_pair_never_appears_twice_in_the_same_day(self) -> None:
        report = build_default_world().advance_day()

        assert all(event.proposer != event.responder for event in report.trades)

    def test_both_villages_get_a_turn_to_open_over_a_run(self) -> None:
        world = build_default_world()
        openers = set()

        for _ in range(30):
            openers |= {e.proposer for e in world.advance_day().trades}

        assert openers == {village.name for village in world.villages}

    def test_two_villages_holding_the_same_surplus_exchange_it(self) -> None:
        world = make_world(2, trade_chance=1.0)
        report = world.advance_day()

        event = report.trades[0]
        assert event.result.agreed
        assert event.result.trade is not None
        assert event.result.trade.offer is not None
        assert event.result.trade.offer.offer_goods is not event.result.trade.offer.want_goods

    def test_a_single_village_is_never_negotiated(self) -> None:
        report = make_world(1).advance_day()

        assert report.trades == ()

    def test_the_barters_leave_the_world_total_untouched(self) -> None:
        world = build_default_world()
        for village in world.villages:
            village.production = zero_stock()
        before = {
            goods: sum(village.stock[goods] for village in world.villages)
            for goods in ALL_GOODS
        }

        report = world.advance_day()

        after = {
            goods: sum(village.stock[goods] for village in world.villages)
            for goods in ALL_GOODS
        }
        consumed = {
            goods: sum(record[goods] for record in report.consumption.values())
            for goods in ALL_GOODS
        }
        for goods in ALL_GOODS:
            assert after[goods] == pytest.approx(before[goods] - consumed[goods])

    def test_each_agreed_barter_moves_exactly_its_quantities(self) -> None:
        world = build_default_world()
        for village in world.villages:
            village.production = zero_stock()
        snapshot = {village.name: dict(village.stock) for village in world.villages}

        report = world.advance_day()

        expected = {name: {goods: 0.0 for goods in ALL_GOODS} for name in snapshot}
        for name, record in report.consumption.items():
            for goods in ALL_GOODS:
                expected[name][goods] -= record[goods]
        for event in report.trades:
            if event.result.trade is None or not event.result.trade.accepted:
                continue
            offer = event.result.trade.offer
            assert offer is not None
            expected[event.proposer][offer.offer_goods] -= offer.offer_quantity
            expected[event.proposer][offer.want_goods] += offer.want_quantity
            expected[event.responder][offer.offer_goods] += offer.offer_quantity
            expected[event.responder][offer.want_goods] -= offer.want_quantity
        for village in world.villages:
            for goods in ALL_GOODS:
                moved = village.stock[goods] - snapshot[village.name][goods]
                assert moved == pytest.approx(expected[village.name][goods])

    def test_no_village_ever_goes_into_debt(self) -> None:
        world = build_default_world()

        for _ in range(10):
            world.advance_day()
            for village in world.villages:
                assert all(village.stock[goods] >= 0.0 for goods in ALL_GOODS)


class TestWorldDeterminism:
    def test_the_same_seed_replays_the_same_run(self) -> None:
        first = build_default_world(seed=5, production_noise=0.3)
        second = build_default_world(seed=5, production_noise=0.3)

        for _ in range(4):
            first.advance_day()
            second.advance_day()

        assert [v.stock for v in first.villages] == [v.stock for v in second.villages]

    def test_a_different_seed_changes_the_perturbed_run(self) -> None:
        first = build_default_world(seed=5, production_noise=0.3)
        second = build_default_world(seed=6, production_noise=0.3)

        for _ in range(4):
            first.advance_day()
            second.advance_day()

        assert [v.stock for v in first.villages] != [v.stock for v in second.villages]

    def test_a_run_without_noise_is_fully_reproducible(self) -> None:
        first = build_default_world(seed=0)
        second = build_default_world(seed=0)

        for _ in range(5):
            first.advance_day()
            second.advance_day()

        assert [v.stock for v in first.villages] == [v.stock for v in second.villages]


class TestDefaultWorldBalance:
    def test_the_three_villages_are_specialised_in_three_different_commodities(self) -> None:
        world = build_default_world()

        specialists = [
            max(ALL_GOODS, key=lambda goods: village.production[goods])
            for village in world.villages
        ]

        assert set(specialists) == set(ALL_GOODS)

    def test_the_three_villages_have_the_same_population(self) -> None:
        world = build_default_world()

        assert len({village.population for village in world.villages}) == 1

    def test_the_world_produces_exactly_what_it_consumes(self) -> None:
        world = build_default_world()
        population = sum(village.population for village in world.villages)

        for goods in ALL_GOODS:
            produced = sum(village.production[goods] for village in world.villages)
            assert produced == pytest.approx(population * goods.daily_consumption)

    def test_every_village_can_feed_itself_on_the_first_day(self) -> None:
        world = build_default_world()

        report = world.advance_day()

        for unmet in report.unmet.values():
            assert unmet == pytest.approx(zero_stock())

    def test_the_report_groups_villages_by_name(self) -> None:
        world = build_default_world()

        assert set(world.stock_of(Goods.FARM)) == {v.name for v in world.villages}


class TestFullySupplied:
    def test_a_productive_day_supplies_everyone(self) -> None:
        report = build_default_world().advance_day()

        assert report.fully_supplied is True

    @staticmethod
    def _starving_world() -> World:
        """A world that stops producing, once the stored food has run out."""
        world = build_default_world()
        world.trade_chance = 0.0
        for village in world.villages:
            village.production = zero_stock()
        for _ in range(2):
            world.advance_day()
        return world

    def test_a_day_nobody_produced_on_leaves_everyone_short(self) -> None:
        world = self._starving_world()

        assert world.advance_day().fully_supplied is False

    def test_even_stocks_are_not_balance_when_everyone_is_short(self) -> None:
        """Three villages holding nothing at all are level with one another and starving."""
        world = self._starving_world()

        world.advance_day()

        assert world.satisfaction_spread() == pytest.approx(0.0)