from __future__ import annotations

import pytest

from world.goods import ALL_GOODS, Goods, zero_stock
from world.village import Village
from world.world import ScheduledEvent, World, build_default_world


def halve_minerals(world: World) -> None:
    """A worked example of the kind of thing an event does."""
    world.village("Miners").production[Goods.MINERAL] /= 2


def arriving(world: World) -> None:
    """Bring a village in, the way a scenario might."""
    world.add_village(Village("Refugees", 10, {g: 5.0 for g in ALL_GOODS}, zero_stock()))


def leaving(world: World) -> None:
    """Send a village away, the way a scenario might."""
    world.remove_village("Miners")


def harvest_burned(into: list[str]):
    """Build an event that empties a granary and records what it destroyed."""

    def apply(world: World) -> None:
        farms = world.village("Farmers")
        into.append(f"{farms.name} lost {sum(farms.stock.values()):.0f} units")
        farms.empty_stores()

    return apply


class TestWhenAnEventLands:
    def test_it_waits_for_its_day(self) -> None:
        world = build_default_world()
        world.schedule(ScheduledEvent(day=3, label="later", apply=halve_minerals))

        world.advance_day()
        world.advance_day()

        assert world.village("Miners").production[Goods.MINERAL] == 20.0

    def test_it_changes_the_day_it_was_promised_for(self) -> None:
        world = build_default_world()
        world.schedule(ScheduledEvent(day=2, label="mines", apply=halve_minerals))

        world.advance_day()
        world.advance_day()

        assert world.village("Miners").production[Goods.MINERAL] == 10.0
        assert [event.label for event in world.fired] == ["mines"]

    def test_it_lands_before_anybody_produces(self) -> None:
        """An emptied granary is emptied before the day fills it again, not after."""
        world = build_default_world()
        lost: list[str] = []
        world.schedule(ScheduledEvent(day=1, label="fire", apply=harvest_burned(lost)))

        world.advance_day()

        assert lost == ["Farmers lost 60 units"]

    def test_two_events_on_one_day_land_in_the_order_they_were_arranged(self) -> None:
        world = build_default_world()
        world.schedule(ScheduledEvent(day=1, label="first", apply=halve_minerals))
        world.schedule(ScheduledEvent(day=1, label="second", apply=harvest_burned([])))

        world.advance_day()

        assert [event.label for event in world.fired] == ["first", "second"]

    def test_events_land_in_day_order_however_they_were_arranged(self) -> None:
        world = build_default_world()
        world.schedule(ScheduledEvent(day=3, label="third", apply=harvest_burned([])))
        world.schedule(ScheduledEvent(day=1, label="first", apply=halve_minerals))
        world.schedule(ScheduledEvent(day=2, label="second", apply=halve_minerals))

        for _ in range(3):
            world.advance_day()

        assert [event.label for event in world.fired] == ["first", "second", "third"]

    def test_a_fired_event_is_no_longer_waiting(self) -> None:
        world = build_default_world()
        world.schedule(ScheduledEvent(day=1, label="once", apply=halve_minerals))

        world.advance_day()
        world.advance_day()

        assert [event.label for event in world.fired] == ["once"]
        assert world.scheduled == []

    def test_it_lands_only_once(self) -> None:
        world = build_default_world()
        world.schedule(ScheduledEvent(day=2, label="twice", apply=halve_minerals))

        for _ in range(5):
            world.advance_day()

        assert [event.label for event in world.fired].count("twice") == 1

    def test_an_event_sitting_in_the_past_never_lands(self) -> None:
        world = build_default_world()
        world.advance_day()
        world.schedule(ScheduledEvent(day=1, label="missed", apply=halve_minerals))

        world.advance_day()

        assert [event.label for event in world.scheduled] == ["missed"]

    def test_scheduling_hands_the_event_back(self) -> None:
        world = build_default_world()
        event = ScheduledEvent(day=1, label="x", apply=halve_minerals)

        assert world.schedule(event) is event

    def test_an_event_that_raises_stops_the_day(self) -> None:
        """A broken event is a broken day, not something to carry on past."""
        world = build_default_world()

        def explode(_world: World) -> None:
            raise RuntimeError("the scenario cannot be applied")

        world.schedule(ScheduledEvent(day=1, label="broken", apply=explode))

        with pytest.raises(RuntimeError, match="cannot be applied"):
            world.advance_day()


class TestEventsAndReproducibility:
    def test_running_an_event_draws_no_randomness(self) -> None:
        """The calendar itself must not touch the world's random stream.

        An event that changes a harvest will of course change who needs to meet
        whom. What must not happen is the firing of the event spending a number of
        its own, so a day with an event on it hands out the same meetings as a day
        without one.
        """
        plain = build_default_world(seed=11)
        busy = build_default_world(seed=11)
        busy.schedule(ScheduledEvent(day=2, label="quiet", apply=lambda world: None))

        for _ in range(6):
            first = [(e.proposer, e.responder) for e in plain.advance_day().trades]
            second = [(e.proposer, e.responder) for e in busy.advance_day().trades]

            assert first == second

    def test_the_same_seed_and_the_same_events_replay_exactly(self) -> None:
        def run() -> list[tuple[int, str, str, Goods]]:
            world = build_default_world(seed=3)
            world.schedule(ScheduledEvent(day=2, label="fire", apply=harvest_burned([])))
            world.schedule(ScheduledEvent(day=4, label="mines", apply=halve_minerals))
            out = []
            for _ in range(6):
                for event in world.advance_day().trades:
                    trade = event.result.trade
                    if event.result.agreed and trade is not None:
                        out.append((world.day, event.proposer, event.responder, trade.offer.offer_goods))
            return out

        assert run() == run()


class TestChangingTheCast:
    def test_a_village_can_come(self) -> None:
        world = build_default_world()

        world.add_village(Village("Refugees", 10, {g: 5.0 for g in ALL_GOODS}, zero_stock()))

        assert world.village("Refugees").name == "Refugees"
        assert len(world.pairs()) == 6

    def test_a_village_can_go(self) -> None:
        world = build_default_world()

        gone = world.remove_village("Miners")

        assert gone.name == "Miners"
        assert [v.name for v in world.villages] == ["Farmers", "Artisans"]

    def test_a_village_that_left_takes_no_further_part(self) -> None:
        world = build_default_world()
        world.remove_village("Miners")

        report = world.advance_day()

        assert "Miners" not in report.production

    def test_the_same_name_cannot_be_taken_twice(self) -> None:
        world = build_default_world()

        with pytest.raises(ValueError, match="already in the world"):
            world.add_village(Village("Farmers", 10, {g: 5.0 for g in ALL_GOODS}, zero_stock()))

    def test_the_last_village_cannot_leave(self) -> None:
        world = World([Village("Last", 10, {g: 5.0 for g in ALL_GOODS}, zero_stock())])

        with pytest.raises(ValueError, match="last village"):
            world.remove_village("Last")

    def test_two_villages_may_not_share_a_name(self) -> None:
        world = build_default_world()
        twin = Village("Farmers", 10, {g: 5.0 for g in ALL_GOODS}, zero_stock())

        with pytest.raises(ValueError, match="unique"):
            World([world.villages[0], twin])

    def test_a_village_can_be_found_and_a_miss_says_so(self) -> None:
        world = build_default_world()

        assert world.village("Artisans").name == "Artisans"
        with pytest.raises(KeyError, match="Ferrymen"):
            world.village("Ferrymen")

    def test_a_village_can_be_sent_away_by_an_event(self) -> None:
        world = build_default_world()
        world.schedule(ScheduledEvent(day=1, label="the ferry closes", apply=leaving))

        world.advance_day()

        assert [v.name for v in world.villages] == ["Farmers", "Artisans"]

    def test_a_village_can_arrive_by_an_event(self) -> None:
        world = build_default_world()
        world.schedule(ScheduledEvent(day=1, label="refugees arrive", apply=arriving))

        report = world.advance_day()

        assert "Refugees" in report.production

    def test_a_cast_that_runs_out_of_villages_is_reported_at_the_next_day(self) -> None:
        world = build_default_world()
        world.schedule(ScheduledEvent(day=2, label="the ferry closes", apply=leaving))
        world.schedule(ScheduledEvent(day=2, label="and again", apply=leaving))

        world.advance_day()

        with pytest.raises(KeyError):
            world.advance_day()


class TestEmptyingStores:
    def test_it_takes_every_commodity_to_nothing(self) -> None:
        world = build_default_world()

        world.village("Farmers").empty_stores()

        assert world.village("Farmers").stock == pytest.approx(zero_stock())

    def test_it_leaves_the_other_villages_alone(self) -> None:
        world = build_default_world()

        world.village("Farmers").empty_stores()

        assert world.village("Miners").stock[Goods.MINERAL] > 0

    def test_a_village_with_empty_stores_goes_short(self) -> None:
        world = build_default_world()
        farms = world.village("Farmers")
        farms.production = zero_stock()
        farms.empty_stores()

        report = world.advance_day()

        assert any(amount > 0 for amount in report.unmet["Farmers"].values())