from __future__ import annotations

import json
from pathlib import Path

import pytest

from world.logbook import DayEntry, RunLog, write_log
from world.narrator import situation, summarize
from world.world import build_default_world


def entry(day: int = 1, **kwargs) -> DayEntry:
    return DayEntry(
        day=day,
        spread=kwargs.pop("spread", 0.0),
        verdict=kwargs.pop("verdict", "the world in balance"),
        **kwargs,
    )


class TestDayEntry:
    def test_a_fully_supplied_day_counts_as_balanced(self) -> None:
        assert entry(spread=0.0, balanced=True).balanced is True

    def test_equal_stocks_are_not_balance_when_everyone_is_short(self) -> None:
        """Three villages holding nothing at all are level with one another and starving."""
        assert entry(spread=0.0, balanced=False).balanced is False

    def test_a_day_with_a_spread_can_still_be_balanced(self) -> None:
        assert entry(spread=0.75, balanced=True).balanced is True

    def test_it_round_trips_through_plain_data(self) -> None:
        day = entry(3, spread=1.25, verdict="a shortage of Minerals", commodity="Minerals")

        data = day.to_json()

        assert data["day"] == 3
        assert data["spread"] == 1.25
        assert data["verdict"] == "a shortage of Minerals"
        assert data["short_of"] == "Minerals"
        assert json.loads(json.dumps(data)) == data

    def test_it_can_be_built_from_a_simulated_day(self) -> None:
        world = build_default_world()
        report = world.advance_day()
        summary = summarize(world, report)

        day = DayEntry.of(report, world, situation(summary), ["a trade"], ["a line"])
        assert day.balanced is report.fully_supplied

        assert day.day == report.day
        assert day.verdict == situation(summary).label
        assert day.spread == pytest.approx(world.satisfaction_spread())


class TestRunLog:
    def test_it_keeps_the_days_in_order(self) -> None:
        log = RunLog()
        log.record(entry(1))
        log.record(entry(2))

        assert [day.day for day in log.entries] == [1, 2]

    def test_a_log_with_nothing_in_it_still_summarises(self) -> None:
        assert RunLog().summary_lines() == ["days simulated: 0"]

    def test_the_summary_counts_the_balanced_days(self) -> None:
        log = RunLog()
        log.record(entry(1, spread=0.0, balanced=True))
        log.record(entry(2, spread=1.25, balanced=False))

        lines = log.summary_lines()

        assert "days simulated: 2" in lines
        assert "days in balance: 1 of 2" in lines

    def test_the_summary_names_the_worst_day(self) -> None:
        log = RunLog()
        log.record(entry(1, spread=0.0, verdict="the world in balance"))
        log.record(entry(2, spread=1.25, verdict="a shortage of Minerals"))

        assert any("a shortage of Minerals" in line for line in log.summary_lines())

    def test_the_summary_reports_the_final_spread(self) -> None:
        log = RunLog()
        log.record(entry(1, spread=1.25))
        log.record(entry(2, spread=0.0))

        assert "final spread: 0.00" in log.summary_lines()

    def test_it_writes_nothing_without_a_path(self, tmp_path: Path) -> None:
        log = RunLog()
        log.record(entry(1))

        log.write()

        assert not list(tmp_path.iterdir())

    def test_a_day_reaches_the_file_as_it_happens(self, tmp_path: Path) -> None:
        target = tmp_path / "run.txt"
        log = RunLog(path=target)
        log.record(entry(1, verdict="the world in balance"))

        assert "day 1" in target.read_text()

    def test_writing_again_does_not_duplicate_a_day(self, tmp_path: Path) -> None:
        target = tmp_path / "run.txt"
        log = RunLog(path=target)
        log.record(entry(1))

        log.write()
        log.write()

        assert target.read_text().count("day 1") == 1


class TestWriteLog:
    def test_it_writes_a_line_per_day(self, tmp_path: Path) -> None:
        target = tmp_path / "run.txt"
        write_log(target, [entry(1), entry(2)], ["days simulated: 2"])

        text = target.read_text()
        assert "day 1" in text
        assert "day 2" in text

    def test_the_summary_closes_the_file(self, tmp_path: Path) -> None:
        target = tmp_path / "run.txt"
        write_log(target, [entry(1)], ["days simulated: 1", "final verdict: all good"])

        text = target.read_text()
        assert text.rstrip().endswith("final verdict: all good")

    def test_trades_and_narration_are_kept(self, tmp_path: Path) -> None:
        target = tmp_path / "run.txt"
        day = entry(1, trades=("A -> B: 5.00 Minerals",), narration=("A quiet day.",))

        write_log(target, [day], [])

        text = target.read_text()
        assert "A -> B: 5.00 Minerals" in text
        assert "A quiet day." in text

    def test_a_json_file_comes_out_as_json_lines(self, tmp_path: Path) -> None:
        target = tmp_path / "run.jsonl"
        write_log(target, [entry(1), entry(2, spread=0.75)], [])

        lines = target.read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["day"] == 1
        assert json.loads(lines[1])["spread"] == 0.75

    def test_vietnamese_written_text_stays_readable(self, tmp_path: Path) -> None:
        target = tmp_path / "run.txt"
        write_log(target, [entry(1, verdict="thiếu hụt nông sản")], [])

        assert "thiếu hụt nông sản" in target.read_text()

    def test_a_run_with_no_days_still_writes_a_summary(self, tmp_path: Path) -> None:
        target = tmp_path / "run.txt"
        write_log(target, [], ["days simulated: 0"])

        assert "days simulated: 0" in target.read_text()