"""Writing a run down to a file, one line per day.

The terminal interface is the only screen, so anything you might want to look at
afterwards goes here instead: a plain text log a human can read, or the same run
as JSON lines when a script needs to read it back.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from world.narrator import Situation
from world.world import DayReport, World

JSON_SUFFIX = ".jsonl"


@dataclass(frozen=True)
class DayEntry:
    """One simulated day, kept so a run can be summarised or written down.

    Attributes:
        day: The day number.
        spread: How far the best fed village was ahead of the worst fed one.
        verdict: Short phrase naming how that day went.
        balanced: Whether every village ended the day fully supplied.
        commodity: The commodity that ran short, if any.
        trades: The barters of the day, as plain text lines.
        narration: The prose the narrator wrote, if it wrote anything.
    """

    day: int
    spread: float
    verdict: str
    balanced: bool = True
    commodity: str | None = None
    trades: tuple[str, ...] = ()
    narration: tuple[str, ...] = ()

    @classmethod
    def of(
        cls,
        report: DayReport,
        world: World,
        state: Situation,
        trades: Sequence[str],
        narration: Sequence[str],
    ) -> DayEntry:
        """Build the entry for a day that has just been simulated."""
        return cls(
            day=report.day,
            spread=world.satisfaction_spread(),
            verdict=state.label,
            balanced=report.fully_supplied,
            commodity=state.commodity.label if state.commodity is not None else None,
            trades=tuple(trades),
            narration=tuple(narration),
        )

    def to_json(self) -> dict[str, object]:
        """The entry as plain data, ready for :func:`json.dumps`."""
        return {
            "day": self.day,
            "spread": round(self.spread, 3),
            "verdict": self.verdict,
            "short_of": self.commodity,
            "trades": list(self.trades),
            "narration": list(self.narration),
        }


@dataclass
class RunLog:
    """Collects the days of a run and writes them out when asked.

    Attributes:
        entries: One entry per simulated day, in order.
        path: Where the log is written, empty until one is chosen.
        seed: The seed the run was started with, empty when it is not known.
        _written: How many entries have already reached the file, for streaming.
    """

    entries: list[DayEntry] = field(default_factory=list)
    path: Path | None = None
    seed: int | None = None
    _written: int = 0

    def record(self, entry: DayEntry) -> None:
        """Remember a day and, when a file is set, append it straight away."""
        self.entries.append(entry)
        if self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(_line_for(self.path, entry))
        self._written = len(self.entries)

    def write(self) -> None:
        """Make sure every day reached the file."""
        if self.path is None:
            return
        missing = self.entries[self._written :]
        if not missing:
            return
        with self.path.open("a", encoding="utf-8") as stream:
            for entry in missing:
                stream.write(_line_for(self.path, entry))
        self._written = len(self.entries)

    def summary_rows(self) -> list[tuple[str, str]]:
        """How the whole run went, as a measure and its value."""
        return [tuple(line.split(": ", 1)) for line in self.summary_lines()]

    def summary_lines(self) -> list[str]:
        """How the whole run went, one line per measure."""
        if not self.entries:
            return ["days simulated: 0"]
        balanced = sum(1 for entry in self.entries if entry.balanced)
        worst = max(self.entries, key=lambda entry: entry.spread)
        lines = [
            f"days simulated: {len(self.entries)}",
            f"days in balance: {balanced} of {len(self.entries)}",
            f"worst day: {worst.verdict} (spread {worst.spread:.2f})",
            f"final spread: {self.entries[-1].spread:.2f}",
            f"final verdict: {self.entries[-1].verdict}",
        ]
        if self.seed is not None:
            lines.insert(0, f"seed: {self.seed}")
        return lines


def write_log(path: Path, entries: Iterable[DayEntry], summary: Sequence[str]) -> None:
    """Write a whole run to a file, choosing the format from the extension."""
    days = list(entries)
    body = "".join(_line_for(path, entry) for entry in days)
    tail = "".join(f"{line}\n" for line in summary)
    path.write_text(body + tail, encoding="utf-8")


def _line_for(path: Path, entry: DayEntry) -> str:
    """One day as a line, in whichever format the file name asks for."""
    if path.suffix == JSON_SUFFIX:
        return json.dumps(entry.to_json(), ensure_ascii=False) + "\n"
    parts = [f"day {entry.day}  spread {entry.spread:.2f}  {entry.verdict}"]
    parts += [f"  {trade}" for trade in entry.trades]
    parts += [f"  {line}" for line in entry.narration]
    return "\n".join(parts) + "\n"