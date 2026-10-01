from __future__ import annotations

import threading

import pytest
from textual.widgets import DataTable

from tests.fakes import FakeModel, FakeNarrator, change, plan
from world.director import Director
from world.goods import ALL_GOODS, Goods
from world.llm import LLMClient
from world.narrator import LLMNarrator
from world.tui import WorldApp
from world.tui import build_app as build_real_app
from world.world import build_default_world

TOTAL_ROWS = 3


def build_app(**kwargs) -> WorldApp:
    """Build the interface with a narrator that needs no model, unless one is given.

    A run now always wants a language model, so the tests that exercise the
    interface would otherwise spend their time waiting on a server that is not
    there. Tests about the narrator itself build the real one on purpose.
    """
    kwargs.setdefault("narrator", FakeNarrator())
    world = kwargs.get("world") or build_default_world()
    kwargs["world"] = world
    kwargs.setdefault("director", Director(FakeModel(), world))
    return build_real_app(**kwargs)


def drive(
    app: WorldApp,
    days: int = 0,
    keys: tuple[str, ...] = (),
    say: tuple[str, ...] = (),
    check=None,
) -> None:
    """Run the interface headless, press keys, assert while it is alive, then stop.

    The widget tree is torn down on exit, so anything that reads the interface
    itself has to be inspected from inside the run. A day is simulated in a worker
    thread, so every key press has to wait for it to land. Anything named in ``say``
    is put to the narrator first and waited out, which is how a world gets shaped
    before it starts.
    """

    async def autopilot(pilot) -> None:
        for text in say:
            app.say_in(text)
            for _ in range(200):
                await pilot.pause()
                if not app.busy:
                    break
        for _ in range(days):
            await pilot.press("n")
            await app.workers.wait_for_complete()
            await pilot.pause()
        for key in keys:
            await pilot.press(key)
            await app.workers.wait_for_complete()
        if check is not None:
            check(app)
        app.exit()

    app.run(headless=True, auto_pilot=autopilot)


def text_of(widget) -> str:
    """Whatever text a static panel currently shows."""
    return str(widget.content)


def summary_text(app: WorldApp) -> str:
    """Every cell of the run summary table, as one string."""
    return " ".join(
        " ".join(str(cell) for cell in app.summary.get_row_at(row))
        for row in range(app.summary.row_count)
    )


class TestComposition:
    def test_it_builds_without_a_display(self) -> None:
        assert isinstance(build_app(), WorldApp)

    def test_the_running_day_and_the_settings_sit_in_columns_of_their_own(self) -> None:
        seen = {}

        def check(app: WorldApp) -> None:
            seen["left"] = [t.id for t in app.query_one("#left").query(DataTable)]
            seen["right"] = [t.id for t in app.query_one("#right").query(DataTable)]

        drive(build_app(), check=check)

        assert seen["left"] == ["stock", "trades", "flow"]
        assert seen["right"] == ["settings", "world_settings", "summary"]


class TestStepping:
    def test_a_day_is_advanced_and_every_table_reports_it(self) -> None:
        """One run of two days has to answer for the day, the stores and the flow.

        Read in a single pass on purpose: booting the interface costs a fixed
        fraction of a second, and four table checks against one world are the same
        question as four worlds against four tables.
        """
        seen = {}

        def check(app: WorldApp) -> None:
            report = app.last_report
            village = app.world.villages[0]
            seen["day"] = app.world.day
            seen["stock_columns"] = [str(c.label) for c in app.stock_table.columns.values()]
            seen["stock_row"] = [str(cell) for cell in app.stock_table.get_row_at(0)]
            seen["held"] = [f"{village.stock[goods]:.0f}" for goods in ALL_GOODS]
            seen["held_total"] = f"{sum(village.stock[g] for g in ALL_GOODS):.0f}"
            seen["rows"] = {
                str(app.flow_table.get_row_at(row)[0]): [
                    str(cell) for cell in app.flow_table.get_row_at(row)[1:]
                ]
                for row in range(app.flow_table.row_count)
            }
            seen["made"] = [
                f"{sum(record[goods] for record in report.production.values()):.0f}"
                for goods in ALL_GOODS
            ]
            seen["used"] = [
                f"{sum(record[goods] for record in report.consumption.values()):.0f}"
                for goods in ALL_GOODS
            ]

        drive(build_app(), days=2, check=check)

        assert seen["day"] == 2
        assert seen["stock_columns"] == ["Village", *(g.label for g in ALL_GOODS), "Total"]
        assert [cell.split()[-1] for cell in seen["stock_row"][1:4]] == seen["held"]
        assert seen["stock_row"][4] == seen["held_total"]
        assert seen["rows"]["Produced"][:3] == seen["made"]
        assert seen["rows"]["Used"][:3] == seen["used"]
        made, used = seen["rows"]["Produced"], seen["rows"]["Used"]
        assert seen["rows"]["Difference"] == [
            f"{float(m) - float(u):.0f}" for m, u in zip(made, used, strict=True)
        ]

    def test_a_day_that_makes_nothing_reports_a_shortfall_and_a_refused_barter(self) -> None:
        """A world that produces nothing has to say so rather than show a blank."""
        from world.goods import zero_stock

        seen = {}

        def check(app: WorldApp) -> None:
            seen["rows"] = {
                str(app.flow_table.get_row_at(row)[0]): [
                    str(cell) for cell in app.flow_table.get_row_at(row)[1:]
                ]
                for row in range(app.flow_table.row_count)
            }
            seen["trade"] = [str(cell) for cell in app.trade_table.get_row_at(0)]

        app = build_app()
        app.world.trade_chance = 1.0
        for village in app.world.villages:
            village.production = zero_stock()
        drive(app, days=1, check=check)

        assert seen["rows"]["Produced"] == ["0", "0", "0", "0"]
        assert float(seen["rows"]["Used"][3]) > 0
        assert seen["rows"]["Difference"] == [f"-{cell}" for cell in seen["rows"]["Used"]]
        assert seen["trade"][2:5] == ["-", "-", "-"]
        assert seen["trade"][5]


class TestNarration:
    def test_the_narration_is_shown_and_is_what_the_narrator_said(self) -> None:
        seen = {}

        def check(app: WorldApp) -> None:
            from world.narrator import summarize

            seen["text"] = app.narration
            seen["expected"] = "\n".join(
                app.narrator.describe(summarize(app.world, app.last_report))
            )

        drive(build_app(narrator=FakeNarrator()), days=2, check=check)

        assert seen["text"].strip()
        assert seen["text"] == seen["expected"]


class TestRunSummary:
    def test_every_day_is_recorded_and_counted(self) -> None:
        app = build_app()
        seen = {}

        drive(app, days=4, check=lambda a: seen.update(text=summary_text(a)))

        assert [entry.day for entry in app.history] == [1, 2, 3, 4]
        assert "days in balance" in seen["text"]
        assert app.history[-1].verdict in seen["text"]
        assert app.history[-1].spread == pytest.approx(app.world.satisfaction_spread())

    def test_a_shortage_is_recorded_as_such(self) -> None:
        app = build_app()
        app.world.villages[2].stock[Goods.CRAFT] = 0.0
        app.world.villages[2].production[Goods.CRAFT] = 0.0

        drive(app, days=1)

        assert "Handicrafts" in app.history[-1].verdict


class TestKeys:
    def test_the_n_key_advances_a_day_and_q_ends_the_run(self) -> None:
        app = build_app()

        drive(app, keys=("n",))

        assert app.world.day == 1

        app = build_app()

        drive(app, days=2, keys=("q",))

        assert app.world.day == 2


class TestBusyIndicator:
    def test_a_second_press_does_not_start_a_second_day(self) -> None:
        gate = threading.Event()
        seen = {}

        class Gated(FakeNarrator):
            def describe(self, summary):
                gate.wait(timeout=10)
                return super().describe(summary)

        app = build_app(narrator=Gated())

        async def autopilot(pilot) -> None:
            await pilot.press("n")
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()
            gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            seen["day"] = app.world.day
            app.exit()

        app.run(headless=True, auto_pilot=autopilot)

        assert seen["day"] == 1

    def test_a_failing_narrator_is_reported_and_the_run_survives(self) -> None:
        class Broken(LLMNarrator):
            def describe(self, summary):
                raise RuntimeError("narrator model unreachable: refused")

        seen = {}
        app = build_app(narrator=Broken(client=LLMClient(base_url="http://127.0.0.1:9")))

        async def autopilot(pilot) -> None:
            await pilot.press("n")
            await app.workers.wait_for_complete()
            await pilot.pause()
            seen["day"] = app.world.day
            seen["text"] = app.narration
            await pilot.press("n")
            await app.workers.wait_for_complete()
            await pilot.pause()
            seen["after"] = app.world.day
            app.exit()

        app.run(headless=True, auto_pilot=autopilot)

        assert seen["day"] == 1
        assert "refused" in seen["text"]
        assert seen["after"] == 2


class TestTheChat:
    def test_the_world_is_shaped_before_a_day_and_says_what_it_replaced(self) -> None:
        """Both halves of one request: the world moves, and the reply says how far."""
        seen = {}
        world = build_default_world()
        app = build_app(
            world=world,
            director=Director(
                FakeModel(
                    plan(change("Miners", "production.Minerals", 4.0)),
                    plan(change(None, "trade_chance", 0.3)),
                ),
                world,
            ),
        )

        drive(
            app,
            say=("half the mines", "talk less"),
            check=lambda a: seen.update(
                started=a.started,
                day=a.world.day,
                production=a.world.village("Miners").production[Goods.MINERAL],
                trade_chance=a.world.trade_chance,
                talk="\n".join(a.talk),
            ),
        )

        assert seen["started"] is True
        assert seen["day"] == 0
        assert seen["production"] == 4.0
        assert seen["trade_chance"] == 0.3
        assert "production.Minerals: 20 -> 4" in seen["talk"]
        assert "trade_chance: 0.6 -> 0.3" in seen["talk"]

    def test_the_world_can_still_be_steered_while_it_runs(self) -> None:
        seen = {}
        world = build_default_world()
        app = build_app(
            world=world,
            director=Director(FakeModel(plan(change(None, "trade_chance", 0.3))), world),
        )

        drive(
            app,
            days=3,
            say=("talk less",),
            check=lambda a: seen.update(trade_chance=a.world.trade_chance, day=a.world.day),
        )

        assert seen == {"trade_chance": 0.3, "day": 3}

    def test_a_change_held_for_a_later_day_lands_during_the_run(self) -> None:
        seen = {}
        world = build_default_world()
        app = build_app(
            world=world,
            director=Director(
                FakeModel(
                    plan(
                        reply="a drought on day 2",
                        schedule=[
                            {
                                "day": 2,
                                "label": "the drought",
                                "changes": [change("Farmers", "production.Agriculture", 2.0)],
                            }
                        ],
                    )
                ),
                world,
            ),
        )

        drive(
            app,
            days=3,
            say=("a drought on day 2",),
            check=lambda a: seen.update(
                production=a.world.village("Farmers").production[Goods.FARM]
            ),
        )

        assert seen["production"] == 2.0

    def test_a_refusal_is_shown_rather_than_swallowed(self) -> None:
        seen = {}
        world = build_default_world()
        app = build_app(
            world=world,
            director=Director(FakeModel(plan(change("Artisans", "seed", 7))), world),
        )

        drive(app, say=("reroll the seed",), check=lambda a: seen.update(talk="\n".join(a.talk)))

        assert "seed" in seen["talk"]

    def test_an_answer_that_cannot_be_read_asks_for_the_instruction_again(self) -> None:
        seen = {}
        world = build_default_world()
        app = build_app(world=world, director=Director(FakeModel("just prose"), world))

        drive(app, say=("make it lean",), check=lambda a: seen.update(talk="\n".join(a.talk)))

        assert "Say it again." in seen["talk"]
        assert "make it lean" in seen["talk"]


class GatedNarrator(FakeNarrator):
    """A narrator that holds the day open until the test says otherwise."""

    def __init__(self, gate) -> None:
        super().__init__()
        self.gate = gate

    def describe(self, summary):
        self.gate.wait(timeout=10)
        return super().describe(summary)


class GatedModel:
    """A model that holds its answer back until the test says otherwise."""

    def __init__(self, gate) -> None:
        self.gate = gate
        self.asked: list[str] = []

    def ask(self, system, user):
        self.asked.append(user)
        self.gate.wait(timeout=10)
        return '{"reply": "done", "changes": [], "schedule": []}'

    def ask_json(self, system, user):
        import json

        return json.loads(self.ask(system, user))


class TestOnlyOneThingAtATime:
    """Both a day and a change request reach into the world from their own thread."""

    def test_a_change_is_refused_while_a_day_is_being_simulated(self) -> None:
        seen = {}
        gate = threading.Event()
        world = build_default_world()
        model = FakeModel(plan(change(None, "trade_chance", 0.3)))
        app = build_app(
            world=world,
            narrator=GatedNarrator(gate),
            director=Director(model, world),
        )

        async def autopilot(pilot) -> None:
            await pilot.press("n")
            await pilot.pause()
            seen["busy"] = app.busy
            app.say_in("talk less")
            await pilot.pause()
            seen["asked"] = len(model.asked)
            gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            seen["trade_chance"] = app.world.trade_chance
            app.exit()

        app.run(headless=True, auto_pilot=autopilot)

        assert seen["busy"] is True
        assert seen["asked"] == 0
        assert seen["trade_chance"] == 0.6

    def test_a_day_is_refused_while_a_change_is_being_read(self) -> None:
        seen = {}
        gate = threading.Event()
        world = build_default_world()
        app = build_app(world=world, director=Director(GatedModel(gate), world))

        async def autopilot(pilot) -> None:
            app.say_in("talk less")
            await pilot.pause()
            seen["busy"] = app.busy
            app.action_next_day()
            await pilot.pause()
            seen["day"] = app.world.day
            gate.set()
            for _ in range(200):
                await pilot.pause()
                if not app.busy:
                    break
            app.exit()

        app.run(headless=True, auto_pilot=autopilot)

        assert seen["busy"] is True
        assert seen["day"] == 0

    def test_the_permit_comes_back_even_when_the_narrator_wedges(self) -> None:
        class Wedged:
            def follow(self, instruction):
                raise RuntimeError("the model fell over")

        world = build_default_world()
        app = build_app(world=world, director=Wedged())
        seen = {}

        def check(interface: WorldApp) -> None:
            seen["busy"] = interface.busy
            seen["talk"] = "\n".join(interface.talk)

        drive(app, say=("do something",), check=check)

        assert seen["busy"] is False
        assert "the model fell over" in seen["talk"]


class TestTheChatScrolls:
    def test_each_turn_is_its_own_widget_and_a_long_one_stays_scrolled_down(self) -> None:
        seen = {}
        answer = "\n".join(f"line {i} of a long answer" for i in range(40))
        world = build_default_world()
        app = build_app(
            world=world,
            director=Director(
                FakeModel(
                    plan(reply="x"),
                    plan(reply="x"),
                    plan(reply="x"),
                    plan(reply=answer),
                ),
                world,
            ),
        )

        async def autopilot(pilot) -> None:
            for text in ("one", "two"):
                app.say_in(text)
                for _ in range(300):
                    await pilot.pause()
                    if not app.busy:
                        break
                # The scroll is asked for after a refresh, so wait for it to land
                # rather than counting pauses and hoping.
                for _ in range(60):
                    await pilot.pause()
                    if app.chat_panel.scroll_offset.y == app.chat_panel.max_scroll_y:
                        break
            seen["turns"] = len(app.talk)
            seen["widgets"] = len(app.chat_panel.children)
            seen["scrollable"] = app.chat_panel.is_scrollable
            seen["at_bottom"] = app.chat_panel.scroll_offset.y == app.chat_panel.max_scroll_y
            app.exit()

        app.run(headless=True, auto_pilot=autopilot)

        assert seen["turns"] == 4
        assert seen["widgets"] == seen["turns"] + 1
        assert seen["scrollable"] is True
        assert seen["at_bottom"] is True

    def test_the_narration_arrives_as_the_newest_turn_of_the_conversation(self) -> None:
        """The narrator writes into the same chat the instructions go into."""
        seen = {}
        app = build_app()

        drive(
            app,
            days=2,
            say=("talk less",),
            check=lambda a: seen.update(
                turns=len(a.chat_panel.children),
                last=str(a.chat_panel.children[-1].content),
                narration=a.narration,
            ),
        )

        assert seen["narration"]
        assert seen["last"] == f"[The Narrator] {seen['narration']}"

    def test_each_turn_is_one_blank_line_apart_and_names_who_is_speaking(self) -> None:
        seen = {}
        world = build_default_world()
        app = build_app(
            world=world,
            director=Director(
                FakeModel(
                    plan(change("Miners", "production.Minerals", 5.0)),
                    plan(reply="Mines thinned."),
                ),
                world,
            ),
            days=1,
        )

        drive(
            app,
            say=("cut the mines",),
            check=lambda a: seen.update(
                turns=[str(t.content) for t in a.chat_panel.children if "turn" in t.classes],
            ),
        )

        assert seen["turns"][0] == "> cut the mines"
        assert seen["turns"][1].startswith("[The Subordinate] ")
        assert seen["turns"][2].startswith("[The Narrator] ")
        turns = app_narrations = seen["turns"]
        assert all(not turn.startswith(">") for turn in turns[1:])
        assert app_narrations

    def test_the_record_of_a_change_is_a_blank_line_away_and_dim(self) -> None:
        """What a turn changed is a record, not something said, and reads as one."""
        seen = {}
        world = build_default_world()
        app = build_app(
            world=world,
            director=Director(
                FakeModel(plan(change("Miners", "production.Minerals", 5.0), reply="Done.")),
                world,
            ),
        )

        drive(app, say=("cut the mines",), check=lambda a: seen.update(
            turn=str(a.chat_panel.children[-1].content),
        ))

        said, _, record = seen["turn"].partition("\n\n")
        assert said == "[The Subordinate] Done."
        assert record == "Miners production.Minerals: 20 -> 5"

    def test_a_narrow_window_scrolls_across_rather_than_cutting_the_numbers(self) -> None:
        """The barters table is wider than a narrow column, so it must scroll.

        The columns keep their own width; a table that does not fit scrolls inside
        itself so a quantity is read whole rather than clipped at the column edge.
        """
        seen = {}

        async def autopilot(pilot) -> None:
            for _ in range(80):
                await pilot.pause()
            barters = app.query_one("#trades")
            seen["panel"] = app.query_one("#left").region.width
            seen["table"] = barters.region.width
            seen["scrolls_across"] = barters.max_scroll_x
            app.exit()

        app = build_app(days=3)
        app.run(headless=True, auto_pilot=autopilot, size=(70, 40))

        assert seen["table"] <= seen["panel"]
        assert seen["scrolls_across"] > 0


class TestSettingsAreShown:
    def world_reading(self, app: WorldApp) -> dict[str, str]:
        """The world settings table as a map of setting name to value."""
        return {
            str(cells[0]): str(cells[1])
            for cells in (
                [str(cell) for cell in app.world_settings_table.get_row_at(row)]
                for row in range(app.world_settings_table.row_count)
            )
        }

    def reading(self, app: WorldApp) -> dict[str, dict[str, str]]:
        """The settings table as a map of setting name to village name to value."""
        rows = [
            [str(cell) for cell in app.settings_table.get_row_at(row)]
            for row in range(app.settings_table.row_count)
        ]
        names = rows[0][1:]
        return {row[0]: dict(zip(names, row[1:], strict=True)) for row in rows[1:]}

    def test_the_table_opens_on_the_world_and_keeps_a_village_to_itself(self) -> None:
        seen = {}
        app = build_app()

        drive(
            app,
            check=lambda a: seen.update(
                header=[str(c) for c in a.settings_table.get_row_at(0)],
                reading=self.reading(a),
                world=self.world_reading(a),
            ),
        )

        assert seen["header"] == ["Setting", "Farmers", "Miners", "Artisans"]
        assert seen["reading"]["People"] == {
            "Farmers": "10",
            "Miners": "10",
            "Artisans": "10",
        }
        assert seen["reading"]["Minerals production"] == {
            "Farmers": "5",
            "Miners": "20",
            "Artisans": "5",
        }
        assert seen["reading"]["Agriculture consumption"] == {
            "Farmers": "10",
            "Miners": "10",
            "Artisans": "10",
        }
        assert "Chance a pair meets" in seen["world"]

    def test_a_change_shows_up_at_once_and_where_it_belongs(self) -> None:
        """A change that lands before any day has run still has to be visible.

        The three changes are read back out of one run because the point is that
        each lands in the right place, and a village's own share of a change must
        not reach its neighbours.
        """
        seen = {}
        world = build_default_world()
        app = build_app(
            world=world,
            director=Director(
                FakeModel(
                    plan(change("Miners", "production.Minerals", 1.0)),
                    plan(change("Farmers", "consumption.Agriculture", 100.0)),
                    plan(change(None, "trade_chance", 0.9)),
                ),
                world,
            ),
        )

        drive(
            app,
            say=("cut the mines", "the farmers eat more grain", "talk more"),
            check=lambda a: seen.update(reading=self.reading(a), world=self.world_reading(a)),
        )

        assert seen["reading"]["Minerals production"]["Miners"] == "1"
        assert seen["reading"]["Agriculture consumption"] == {
            "Farmers": "100",
            "Miners": "10",
            "Artisans": "10",
        }


class TestKeysWhileTyping:
    def test_typing_into_the_box_does_not_step_the_day(self) -> None:
        """While the box has the keys, n is a letter rather than a day."""
        seen = {}
        app = build_app()

        async def autopilot(pilot) -> None:
            app.say.focus()
            await pilot.press("n")
            await pilot.pause()
            seen["day"] = app.world.day
            seen["box"] = app.say.value
            app.exit()

        app.run(headless=True, auto_pilot=autopilot)

        assert seen == {"day": 0, "box": "n"}


class TestAutomaticRun:
    """A run given a number of days settles down without any keys pressed."""

    def test_it_simulates_every_day_on_its_own_then_closes(self) -> None:
        app = build_app(days=3)

        app.run(headless=True)

        assert [entry.day for entry in app.history] == [1, 2, 3]
        assert app.remaining == 0
        assert app.is_running is False
