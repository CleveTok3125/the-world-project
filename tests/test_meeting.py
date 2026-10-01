from __future__ import annotations

import pytest

from world.goods import ALL_GOODS, Goods
from world.world import World, build_default_world


def make_world(
    count: int = 3, trade_chance: float | None = None, time_per_day: int | None = None, seed: int = 0
) -> World:
    """A world of the given size, optionally with meeting rules of its own."""
    villages = tuple(build_default_world(seed=seed).villages[index] for index in range(count))
    extra = {}
    if trade_chance is not None:
        extra["trade_chance"] = trade_chance
    if time_per_day is not None:
        extra["time_per_day"] = time_per_day
    return World(villages, seed=seed, **extra)


class TestTimeIsTheBudget:
    def test_every_barter_costs_one_hour(self) -> None:
        world = make_world(trade_chance=1.0, time_per_day=3)

        report = world.advance_day()

        assert report.hours_total == 3
        assert report.hours_spent == len(report.trades)
        assert report.hours_left == 3 - len(report.trades)

    def test_a_refused_barter_costs_an_hour_too(self) -> None:
        world = make_world(trade_chance=1.0, time_per_day=3)
        for village in world.villages:
            village.production = {goods: 0.0 for goods in ALL_GOODS}

        report = world.advance_day()

        assert len(report.trades) == 3
        assert all(not event.result.agreed for event in report.trades)
        assert report.hours_left == 0

    def test_a_day_with_no_hours_never_meets_anyone(self) -> None:
        world = make_world(trade_chance=1.0, time_per_day=0)

        for _ in range(10):
            report = world.advance_day()

            assert report.trades == ()
            assert report.hours_left == 0

    def test_a_day_with_one_hour_meets_at_most_one_pair(self) -> None:
        world = make_world(trade_chance=1.0, time_per_day=1)

        for _ in range(10):
            report = world.advance_day()

            assert len(report.trades) == 1

    def test_the_hours_are_reported_even_when_nobody_trades(self) -> None:
        world = make_world(trade_chance=0.0, time_per_day=4)

        report = world.advance_day()

        assert report.hours_total == 4
        assert report.hours_spent == 0
        assert report.hours_left == 4


class TestMeetingIsNotFixed:
    def test_the_number_of_barters_varies_from_day_to_day(self) -> None:
        world = make_world(trade_chance=0.55, time_per_day=3)
        counts = []

        for _ in range(60):
            counts.append(len(world.advance_day().trades))

        assert min(counts) < max(counts)

    def test_the_same_pairs_do_not_meet_every_single_day(self) -> None:
        world = make_world(trade_chance=0.55, time_per_day=3)
        seen = []

        for _ in range(30):
            report = world.advance_day()
            seen.append({frozenset((e.proposer, e.responder)) for e in report.trades})

        assert any(pairs != seen[0] for pairs in seen)

    def test_a_certain_chance_with_enough_hours_meets_every_pair(self) -> None:
        world = make_world(trade_chance=1.0, time_per_day=3)

        for _ in range(10):
            assert len(world.advance_day().trades) == 3

    def test_a_chance_of_zero_keeps_idle_villages_apart(self) -> None:
        world = make_world(trade_chance=0.0)

        report = world.advance_day()

        assert report.trades == ()

    def test_a_village_in_need_is_served_even_at_a_chance_of_zero(self) -> None:
        world = make_world(trade_chance=0.0, time_per_day=3)
        world.villages[0].stock[Goods.CRAFT] = 0.0
        world.villages[0].production[Goods.CRAFT] = 0.0

        report = world.advance_day()

        assert any(event.result.agreed for event in report.trades)

    def test_the_same_seed_replays_the_same_meetings(self) -> None:
        first = make_world(trade_chance=0.5, seed=5)
        second = make_world(trade_chance=0.5, seed=5)
        one = two = []

        for _ in range(15):
            one = [(e.proposer, e.responder) for e in first.advance_day().trades]
            two = [(e.proposer, e.responder) for e in second.advance_day().trades]

        assert one == two

    def test_a_different_seed_changes_the_meetings(self) -> None:
        first = make_world(trade_chance=0.5, seed=5)
        second = make_world(trade_chance=0.5, seed=6)
        one = two = []

        for _ in range(15):
            one = [(e.proposer, e.responder) for e in first.advance_day().trades]
            two = [(e.proposer, e.responder) for e in second.advance_day().trades]

        assert one != two


class TestHoursAreTheOnlyLimit:
    """The day has a fixed number of hours, and nothing else holds a barter back.

    There is no memory of what a pair traded before, so a commodity may travel
    from A to B and straight back the next day. Whether that happens is up to the
    hours left in the day.
    """

    def test_a_commodity_can_travel_back_the_way_it_came(self) -> None:
        traded = []
        world = make_world(seed=13)

        for _ in range(120):
            for event in world.advance_day().trades:
                trade = event.result.trade
                if event.result.agreed and trade is not None:
                    traded.append(
                        (world.day, event.proposer, event.responder, trade.offer.offer_goods)
                    )

        round_trips = [
            (day, giver, taker, goods)
            for day, giver, taker, goods in traded
            if any(
                later == day + 1 and (other, back, back_goods) == (taker, giver, goods)
                for later, other, back, back_goods in traded
            )
        ]

        assert round_trips

    def test_the_day_ends_when_the_hours_run_out(self) -> None:
        world = make_world(trade_chance=1.0, time_per_day=3)

        for _ in range(30):
            report = world.advance_day()

            assert report.hours_left >= 0
            assert report.hours_spent == len(report.trades)
            assert report.hours_spent <= report.hours_total

    def test_goods_never_appear_from_nowhere_over_a_long_run(self) -> None:
        world = make_world(seed=1)
        before = {
            goods: sum(village.stock[goods] for village in world.villages)
            for goods in ALL_GOODS
        }

        for _ in range(60):
            report = world.advance_day()
            after = {
                goods: sum(village.stock[goods] for village in world.villages)
                for goods in ALL_GOODS
            }
            for goods in ALL_GOODS:
                made = sum(day[goods] for day in report.production.values())
                used = sum(day[goods] for day in report.consumption.values())
                assert after[goods] == pytest.approx(before[goods] + made - used)
            before = after


class TestBalanceSurvivesRandomMeetings:
    def test_the_world_still_balances(self) -> None:
        world = make_world(seed=1)

        for _ in range(60):
            world.advance_day()

        assert world.satisfaction_spread() <= 2.5

    def test_only_a_few_days_leave_a_village_short(self) -> None:
        world = make_world(seed=1)
        hungry = 0

        for _ in range(60):
            report = world.advance_day()
            if any(amount > 0 for unmet in report.unmet.values() for amount in unmet.values()):
                hungry += 1

        assert hungry <= 12

    def test_goods_never_appear_from_nowhere(self) -> None:
        world = make_world(seed=1)
        before = {
            goods: sum(village.stock[goods] for village in world.villages)
            for goods in ALL_GOODS
        }

        for _ in range(30):
            report = world.advance_day()
            after = {
                goods: sum(village.stock[goods] for village in world.villages)
                for goods in ALL_GOODS
            }
            for goods in ALL_GOODS:
                produced = sum(day[goods] for day in report.production.values())
                consumed = sum(day[goods] for day in report.consumption.values())
                assert after[goods] == pytest.approx(before[goods] + produced - consumed)
            before = after

    def test_every_day_still_holds_at_least_one_barter(self) -> None:
        world = make_world(seed=1)

        for _ in range(40):
            assert world.advance_day().trades

    def test_a_long_run_keeps_the_world_even(self) -> None:
        world = make_world(seed=7)

        for _ in range(200):
            world.advance_day()

        assert world.satisfaction_spread() <= 2.5


class TestSettings:
    def test_the_defaults_allow_several_barters_a_day(self) -> None:
        world = make_world()

        assert world.time_per_day >= 3
        assert 0.0 < world.trade_chance <= 1.0

    def test_a_chance_above_one_is_refused(self) -> None:
        world = make_world()
        world.trade_chance = 5.0

        with pytest.raises(ValueError, match="chance"):
            world.advance_day()

    def test_a_negative_chance_is_refused(self) -> None:
        world = make_world()
        world.trade_chance = -1.0

        with pytest.raises(ValueError, match="chance"):
            world.advance_day()

    def test_negative_hours_are_refused(self) -> None:
        world = make_world()
        world.time_per_day = -1

        with pytest.raises(ValueError, match="negative"):
            world.advance_day()

    def test_nothing_trades_when_the_settings_are_checked_first(self) -> None:
        world = make_world()
        world.time_per_day = -1

        with pytest.raises(ValueError):
            world.advance_day()

        assert world.day == 0