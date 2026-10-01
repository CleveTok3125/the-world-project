"""The terminal interface: tables above, and one conversation below."""

from __future__ import annotations

import threading
from typing import ClassVar

from rich.markup import escape
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.containers import Grid, Horizontal, ScrollableContainer, Vertical, VerticalScroll
from textual.widgets import DataTable, Footer, Input, LoadingIndicator, Static

from world.director import Director
from world.goods import ALL_GOODS
from world.llm import LLMClient, ModelTimeout
from world.logbook import DayEntry, RunLog
from world.narrator import Narrator, build_narrator, situation, summarize
from world.world import DayReport, World, build_default_world

TITLE = "The World Project"
SUB_TITLE = "n or space: next day    q: quit"
SUBORDINATE = "subordinate"
NARRATOR = "narrator"


def titled(role: str) -> str:
    """A role as it is shown, so one place decides how it reads."""
    return f"The {role.capitalize()}"


class WorldApp(App[None]):
    """Build a world, then step it forward one day at a time.

    Attributes:
        world: The simulation being driven.
        narrator: Turns each day into prose.
        director: Reads what was asked and changes the world accordingly.
        history: One record per simulated day.
        last_report: What the most recent day produced.
        seed: The seed the run was started with, shown so it can be repeated.
        remaining: Days still to run on their own; zero means wait for the keys.
        permit: The one permit to touch the world, held while a worker runs.
    """

    TITLE = TITLE
    SUB_TITLE = SUB_TITLE

    CSS = """
    Screen { layout: vertical; }
    #banner { height: auto; padding: 0 1; }
    .heading { height: 1; padding: 0 1; color: $text-muted; }
    #tables { height: 1fr; grid-size: 2 1; grid-columns: 55fr 44fr; }
    #settings { height: auto; }
    #world_settings { height: auto; }
    #flow { height: auto; }
    #console { height: 33%; border-top: round $panel; }
    #chat { height: 1fr; padding: 0 1; }
    #chat > .turn { margin-bottom: 1; padding: 0 1; }
    #chat > .turn.you { background: $boost; }
    #chat > .turn.subordinate { background: $panel; }
    #chat > .turn.narrator { background: $surface; }
    #thinking { height: auto; padding: 0 1; }
    #thinking > LoadingIndicator { width: auto; }
    #thought { margin-left: 2; }
    #prompt { height: auto; padding: 0 1; color: $text-muted; }
    """

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("n", "next_day", "Next day"),
        ("space", "next_day", "Next day"),
        ("c", "focus_chat", "Chat"),
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        world: World | None = None,
        narrator: Narrator | None = None,
        director=None,
        days: int = 0,
    ) -> None:
        super().__init__()
        self.world = world if world is not None else build_default_world()
        self.permit = threading.Lock()
        self.advancing = False
        self.narrator = narrator if narrator is not None else build_narrator()
        self.director = director
        self.history: list[DayEntry] = []
        self.book = RunLog()
        self.last_report: DayReport | None = None
        self.seed: int | None = None
        self.narrated = ""
        self.remaining = days
        self.talk: list[str] = []

    @property
    def busy(self) -> bool:
        """Whether somebody is already working on the world.

        Taken without waiting, so a request arriving mid-day is dropped rather than
        queued.
        """
        return self.permit.locked()

    def _claim(self) -> bool:
        """Take the world's one permit, or report that somebody else already has it."""
        return self.permit.acquire(blocking=False)

    def _release(self) -> None:
        """Give the permit back, if this worker still holds it."""
        if self.permit.locked():
            self.permit.release()

    @property
    def started(self) -> bool:
        """Whether a day has been simulated yet, which is also when the tables appear."""
        return self.last_report is not None or bool(self.talk)

    @property
    def banner(self) -> Static:
        """The line naming the day and the verdict."""
        return self.query_one("#banner", Static)

    @property
    def narration(self) -> str:
        """The prose the narrator wrote about the day most recently simulated."""
        return self.narrated

    @property
    def status(self) -> Static:
        """The one line saying what the interface is doing right now."""
        return self.query_one("#status", Static)

    @property
    def status_text(self) -> str:
        """Whatever the status line currently says."""
        return str(self.status.content)

    @property
    def summary(self) -> DataTable:
        """How the whole run went, as a measure and its value."""
        return self.query_one("#summary", DataTable)

    @property
    def journal(self) -> RunLog:
        """The written record of every day, ready to be saved to a file."""
        return self.book

    @property
    def settings_table(self) -> DataTable:
        """The settings a change can move, as they stand right now."""
        return self.query_one("#settings", DataTable)

    @property
    def world_settings_table(self) -> DataTable:
        """The settings the whole world shares, which belong to no single village."""
        return self.query_one("#world_settings", DataTable)

    @property
    def flow_table(self) -> DataTable:
        """What the world made, ate and came out by on the last day it ran."""
        return self.query_one("#flow", DataTable)

    @property
    def stock_table(self) -> DataTable:
        """Current stock of every commodity, one row per village."""
        return self.query_one("#stock", DataTable)

    @property
    def trade_table(self) -> DataTable:
        """The barters of the most recent day."""
        return self.query_one("#trades", DataTable)

    def compose(self) -> ComposeResult:
        yield Static(self._banner_text(), id="banner")
        yield Static("", id="status", classes="heading", markup=False)
        with Grid(id="tables"):
            with ScrollableContainer(id="left"):
                yield Static("STOCK", classes="heading")
                yield DataTable(id="stock")
                yield Static("BARTERS", classes="heading")
                yield DataTable(id="trades")
                yield Static("LAST DAY", classes="heading")
                yield DataTable(id="flow")
            with ScrollableContainer(id="right"):
                yield Static("SETTINGS", classes="heading")
                yield DataTable(id="settings")
                yield Static("WORLD", classes="heading")
                yield DataTable(id="world_settings")
                yield Static("RUN SUMMARY", classes="heading")
                yield DataTable(id="summary")
        with Vertical(id="console"):
            with VerticalScroll(id="chat"):
                yield Static("Nobody is talking yet.", id="quiet", markup=False)
            with Horizontal(id="thinking"):
                yield LoadingIndicator()
                yield Static("The narrator is thinking...", id="thought", markup=False)
            yield Static(SETUP_HINT, id="prompt", markup=False)
            yield Input(placeholder="say what to change, or leave it and press n", id="say")
        yield Footer()

    @property
    def chat_panel(self) -> VerticalScroll:
        """The running conversation with the narrator, newest turn at the bottom."""
        return self.query_one("#chat", VerticalScroll)

    @property
    def chat_hint(self) -> Static:
        """The one line telling the reader what to type and when."""
        return self.query_one("#prompt", Static)

    @property
    def say(self) -> Input:
        """The box a change is typed into."""
        return self.query_one("#say", Input)

    def on_key(self, event: events.Key) -> None:
        """Hand the keys back when the reader asks, so the day keys work again.

        Typing takes every key while the box has focus, which would otherwise make
        the day keys unreachable for as long as the box is being typed into.
        """
        if event.key == "escape" and self.say.has_focus:
            self.set_focus(None)
            event.stop()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Take what was typed and go and do it."""
        event.input.value = ""
        self.say_in(event.value)

    def say_in(self, text: str) -> None:
        """Put one instruction in front of the narrator."""
        text = text.strip()
        if not text:
            return
        if self.director is None:
            self._say("Nobody is listening: there is no model to take a change from.")
            return
        if not self._claim():
            return
        self.say.disabled = True
        self._thinking(True, SUBORDINATE)
        self._say(f"> {text}")
        self.status.update("Working on it...")
        self._ask(text)

    @work(thread=True, group="talk")
    def _ask(self, text: str) -> None:
        """Read the instruction off the model, which is slow, on a worker thread.

        The permit is handed back by :meth:`_told` once the answer is on screen. If
        this thread dies before it manages that, the permit is released here rather
        than left held, which would wedge the interface for good.
        """
        handed_back = False
        try:
            reply = self.director.follow(text)
            self.app.call_from_thread(self._told, reply)
            handed_back = True
        except Exception as problem:  # noqa: BLE001
            self.app.call_from_thread(self._failed, "The narrator could not be asked.", problem)
            handed_back = True
        finally:
            if not handed_back:
                self._release()

    def _told(self, reply) -> None:
        """Put the answer on screen and let the world be steered again."""
        self._release()
        self._thinking(False)
        self.say.disabled = False
        said, logs = reply.parts()
        self._say(said, SUBORDINATE, logs)
        self.status.update("Ready.")
        self._refresh()
        self.chat_hint.update(RUNNING_HINT)

    def _say(self, text: str, who: str = "", logs: tuple[str, ...] = ()) -> None:
        """Add one turn to the conversation and follow it down.

        One widget per turn, since a single static has no scrollable size to measure.
        The scroll waits for the next refresh because the turn has to be laid out
        first. ``who`` names whoever is speaking, except for a turn of your own,
        which carries the ``>`` you typed and needs no name. ``logs`` is the record
        of what a turn changed, which is a line away from the sentence and dim,
        because it is a record rather than something said.
        """
        self.talk.append("\n\n".join((text, *logs)))
        quiet = self.query_one("#quiet", Static)
        quiet.display = False
        name = titled(who)
        said = Text.from_markup(f"[bold cyan]\\[{name}][/bold cyan] {escape(text)}") if who else Text(text)
        if logs:
            said.append("\n\n")
            said.append(Text("\n".join(logs), style="dim"))
        self.chat_panel.mount(Static(said, classes=f"turn {who or 'you'}"))
        self._follow_chat()

    def on_mount(self) -> None:
        self._thinking(False)
        self.settings_table.add_columns(
            "Setting", *(village.name for village in self.world.villages)
        )
        self.world_settings_table.add_columns("Setting", "Value")
        self.stock_table.add_columns("Village", *(goods.label for goods in ALL_GOODS), "Total")
        self.flow_table.add_columns("Measure", *(goods.label for goods in ALL_GOODS), "Total")
        self.trade_table.add_columns(
            "From", "To", "Gave", "Got", "Rate", "Result", "Reason"
        )
        self.summary.add_columns("Measure", "Value")
        self._refresh()
        if self.remaining > 0:
            self.call_later(self.action_next_day)

    def action_focus_chat(self) -> None:
        """Hand the keys to the chat box.

        Reaching the box with tab means walking through every table on the way, and
        the tables take focus whether or not they can be typed into.
        """
        self.say.focus()

    def action_next_day(self) -> None:
        """Start simulating one more day without blocking the interface.

        Narrating a day can take a while when a language model is behind it, so
        the work runs on a worker thread and the panel shows a spinner meanwhile.
        A press that arrives while a day is still running is ignored.
        """
        if not self._claim():
            return
        self.advancing = True
        self._thinking(True)
        self.status.update("Simulating the next day...")
        self._simulate()

    @work(thread=True, group="day")
    def _simulate(self) -> None:
        """Simulate a day and narrate it, then hand the result back to the interface.

        As with the chat, the permit is normally given back by :meth:`_apply`, and
        released here only if this thread never got that far.
        """
        handed_back = False
        try:
            try:
                report = self.world.advance_day()
                summary = summarize(self.world, report)
            except Exception as problem:  # noqa: BLE001
                # Anything at all may come out of the world; the day must still close.
                self.app.call_from_thread(self._apply, None, None, [], str(problem))
                handed_back = True
                return
            try:
                lines = self.narrator.describe(summary)
            except ModelTimeout as problem:
                lines = []
                failure = f"The narrator ran out of time. {problem}."
            except Exception as problem:  # noqa: BLE001
                lines = []
                failure = str(problem)
            else:
                failure = ""
            self.app.call_from_thread(self._apply, report, summary, lines, failure)
            handed_back = True
        finally:
            if not handed_back:
                self._release()

    def _thinking(self, working: bool, who: str = NARRATOR) -> None:
        """Show or hide the marker that says whoever is working is still working."""
        self.query_one("#thinking").display = working
        if working:
            self.query_one("#thought").update(f"{titled(who)} is thinking...")

    def _failed(self, what: str, problem: Exception) -> None:
        """Report that a worker could not finish, and give the permit back."""
        self._release()
        self._thinking(False)
        self.say.disabled = False
        self.status.update(what)
        self._say(f"{what} {problem}")

    def _apply(self, report, summary, lines: list[str], failure: str) -> None:
        """Put a finished day on screen."""
        self._release()
        self.advancing = False
        self._thinking(False)
        if failure:
            self._publish_narration(f"The narrator fell silent: {failure}")
            self.status.update("Day not narrated.")
            self._keep_going()
            return
        state = situation(summary)
        self.last_report = report
        self.history.append(
            DayEntry(
                day=report.day,
                spread=summary.spread,
                verdict=state.label,
                balanced=report.fully_supplied,
                commodity=state.commodity.label if state.commodity else None,
            )
        )

        self._publish_narration("\n".join(lines))
        self.banner.update(self._banner_text())
        self.status.update(f"Day {report.day} done.")
        self._record(report, state, lines)
        self._refresh()
        self._keep_going()

    def _keep_going(self) -> None:
        """Run the next day on its own, or close once the run has had its days."""
        if self.remaining <= 0:
            return
        self.remaining -= 1
        if self.remaining == 0:
            self.exit()
            return
        self.call_later(self.action_next_day)

    def _banner_text(self) -> str:
        if self.last_report is None:
            if self.seed is None:
                return f"Day 0 -- press {SUB_TITLE}"
            return f"Day 0  --  seed {self.seed}  --  press {SUB_TITLE}"
        verdict = self.history[-1].verdict if self.history else ""
        return f"Day {self.last_report.day}  --  {verdict}"

    def _refresh(self) -> None:
        self._fill_settings()
        self._fill_world_settings()
        self._fill_stock()
        self._fill_trades()
        self._fill_flow()
        self._fill_summary()

    def _follow_chat(self) -> None:
        """Scroll to the newest turn, once the layout has caught up."""
        self.chat_panel.call_after_refresh(self.chat_panel.scroll_end, animate=False)

    def _publish_narration(self, prose: str) -> None:
        """Show what the narrator wrote, as the newest turn of the conversation.

        Mounted rather than written into a fixed panel, so the day reads the same
        way a reply to an instruction does.
        """
        self.narrated = prose
        self._say(prose, NARRATOR)

    def _fill_summary(self) -> None:
        """Fill the run summary table in the right column."""
        table = self.summary
        table.clear()
        for measure, value in self.book.summary_rows():
            table.add_row(measure, value)

    def _record(self, report, state, lines: list[str]) -> None:
        """Add the finished day to the written record of the run."""
        trades = [line.strip() for line in self._trade_lines(report)]
        self.book.record(
            DayEntry(
                day=report.day,
                spread=self.world.satisfaction_spread(),
                verdict=state.label,
                balanced=report.fully_supplied,
                commodity=state.commodity.label if state.commodity else None,
                trades=tuple(trades),
                narration=tuple(lines),
            )
        )

    @staticmethod
    def _trade_lines(report) -> list[str]:
        lines = []
        for event in report.trades:
            trade = event.result.trade
            offer = trade.offer if trade is not None else None
            if event.result.agreed and offer is not None:
                lines.append(
                    f"{event.proposer} -> {event.responder}: "
                    f"{offer.offer_quantity:.2f} {event.proposer}'s {offer.offer_goods.label} "
                    f"for {offer.want_quantity:.2f} {offer.want_goods.label}"
                )
            else:
                lines.append(
                    f"{event.proposer} -> {event.responder}: no deal ({event.result.reason})"
                )
        return lines

    def _fill_settings(self) -> None:
        """The settings a change can move, one row each, villages across the top.

        Turned this way round because there are far more settings than villages:
        read down a column and one village's whole economy is together, which is
        what makes a change visible next to what it sits beside.
        """
        table = self.settings_table
        table.clear()
        names = [village.name for village in self.world.villages]
        table.add_row("Setting", *names)

        def line(label: str, read) -> None:
            table.add_row(label, *(f"{read(village):g}" for village in self.world.villages))

        line("People", lambda v: float(v.population))
        line("Reserve days", lambda v: v.reserve_days)
        line("Waver", lambda v: v.production_noise)
        for goods in ALL_GOODS:
            line(
                f"{goods.label} production",
                lambda v, good=goods: v.production[good],
            )
        for goods in ALL_GOODS:
            line(
                f"{goods.label} consumption",
                lambda v, good=goods: v.consumption[good],
            )


    def _fill_world_settings(self) -> None:
        """The settings the whole world shares, kept out of the per-village table.

        They are one number each, and putting them in a table whose columns are
        villages would print the same number under every village and read as a
        setting each of them had of its own.
        """
        table = self.world_settings_table
        table.clear()
        table.add_row("Barters a day", f"{self.world.time_per_day:g}")
        table.add_row("Chance a pair meets", f"{self.world.trade_chance:g}")
        table.add_row("Rounds a barter", f"{self.world.max_rounds:g}")

    def _fill_stock(self) -> None:
        """What each village holds, drawn to the largest holding on screen."""
        table = self.stock_table
        table.clear()
        if not self.world.villages:
            return
        top = max(
            (max(village.stock[goods] for goods in ALL_GOODS) for village in self.world.villages),
            default=0.0,
        )
        for village in self.world.villages:
            table.add_row(
                village.name,
                *(
                    _bar(village.stock[goods], top) + f" {village.stock[goods]:.0f}"
                    for goods in ALL_GOODS
                ),
                _amount(sum(village.stock[goods] for goods in ALL_GOODS)),
            )

    def _fill_flow(self) -> None:
        """What the world made, used and came out by on the day it last ran."""
        table = self.flow_table
        table.clear()
        made = self._day_totals("production")
        used = self._day_totals("consumption")
        if made is None or used is None:
            return
        self._add_total_row(table, "Produced", made, "success")
        self._add_total_row(table, "Used", used, "warning")
        self._add_total_row(
            table,
            "Difference",
            tuple(
                made_amount - used_amount
                for made_amount, used_amount in zip(made, used, strict=True)
            ),
            "primary",
        )

    def _add_total_row(self, table, label: str, amounts, colour: str) -> None:
        """Add one world-wide row, coloured so it reads apart from the holdings above.

        The holdings are shown with bars and no colour, so colour here is free to
        mean the same thing on every row: what came in, what went out, and the gap.
        """
        style = f"bold {self.app.theme_variables[colour]}"
        table.add_row(
            Text(label, style=style),
            *(Text(_amount(amount), style=style) for amount in amounts),
            Text(_amount(sum(amounts)), style=style),
        )

    def _day_totals(self, field: str) -> tuple[float, ...] | None:
        """The world's output or intake of each commodity on the last simulated day."""
        if self.last_report is None:
            return None
        per_village: dict[str, dict] = getattr(self.last_report, field)
        return tuple(sum(record[goods] for record in per_village.values()) for goods in ALL_GOODS)

    def _fill_trades(self) -> None:
        table = self.trade_table
        table.clear()
        if self.last_report is None:
            return
        for event in self.last_report.trades:
            result = event.result
            offer = result.rounds[-1].offer if result.rounds else None
            if offer is None:
                table.add_row(
                    event.proposer, event.responder, "-", "-", "-", "Refused", result.reason
                )
                continue
            rate = offer.want_quantity / offer.offer_quantity
            table.add_row(
                event.proposer,
                event.responder,
                f"{offer.offer_quantity:.2f} {offer.offer_goods.label}",
                f"{offer.want_quantity:.2f} {offer.want_goods.label}",
                f"{rate:.2f}",
                "Accepted" if result.agreed else "Refused",
                result.reason,
            )


def _amount(value: float) -> str:
    """One cell of a total row, to the same precision the bars around it use."""
    return f"{value:.0f}"


SETUP_HINT = (
    "Give your subordinate a change before it starts, "
    "then press n to begin. Press c to type, escape to let go again."
)
RUNNING_HINT = (
    "Keep steering while it runs: press c to type a change, "
    "escape to let go, n for the next day."
)
BAR_WIDTH = 8


def _bar(amount: float, top: float) -> str:
    """How much of the largest holding this amount is, drawn as a bar."""
    if top <= 0:
        return "·" * BAR_WIDTH
    filled = round(amount / top * BAR_WIDTH)
    return "█" * filled + "·" * (BAR_WIDTH - filled)


def build_app(
    world: World | None = None,
    narrator: Narrator | None = None,
    director=None,
    seed: int = 0,
    days: int = 0,
) -> WorldApp:
    """Create the interface, filling in whatever the caller left out."""
    world = world if world is not None else build_default_world()
    if director is None:
        director = Director(LLMClient.from_environment(), world)
    return WorldApp(
        world=world,
        narrator=narrator if narrator is not None else build_narrator(),
        director=director,
        days=days,
    )


__all__ = ["SUB_TITLE", "TITLE", "WorldApp", "build_app"]

