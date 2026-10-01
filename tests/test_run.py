from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

import run
from world.llm import BASE_URL_ENV, LLMClient
from world.logbook import DayEntry, RunLog
from world.narrator import LLMNarrator


class FakeApp:
    """Stands in for the interface once it has finished running."""

    def __init__(self, history: list[DayEntry], journal) -> None:
        self.history = history
        self.journal = journal
        self.seed = 1
        self.ran = False

    def run(self) -> None:
        self.ran = True


class TestSeed:
    def test_no_seed_on_the_command_line_makes_one_up(self, monkeypatch) -> None:
        monkeypatch.setattr(run, "fresh_seed", lambda: 4242)

        assert run.parse_args([]).seed == 4242

    def test_each_run_makes_up_its_own_seed(self, monkeypatch) -> None:
        picks = iter([1, 2, 3])
        monkeypatch.setattr(run, "fresh_seed", lambda: next(picks))

        assert [run.parse_args([]).seed for _ in range(3)] == [1, 2, 3]

    def test_a_seed_given_on_the_command_line_is_left_alone(self, monkeypatch) -> None:
        monkeypatch.setattr(run, "fresh_seed", lambda: 4242)

        assert run.parse_args(["--seed", "7"]).seed == 7

    def test_a_made_up_seed_is_wide_enough_to_be_worth_printing(self) -> None:
        seeds = {run.fresh_seed() for _ in range(50)}

        assert len(seeds) > 40
        assert all(0 <= seed < 2**31 for seed in seeds)


class TestOptions:
    def test_the_defaults_describe_a_step_by_step_run(self) -> None:
        options = run.parse_args([])

        assert options.days == 0
        assert options.noise == 0.0
        assert options.log is None

    def test_the_default_seed_is_not_a_fixed_number(self) -> None:
        assert run.parse_args([]).seed != 0

    def test_every_flag_is_read(self) -> None:
        options = run.parse_args(
            ["--days", "7", "--seed", "12", "--noise", "0.25", "--log", "run.txt"]
        )

        assert options.days == 7
        assert options.seed == 12
        assert options.noise == 0.25
        assert options.log == Path("run.txt")

    def test_a_negative_number_of_days_is_refused(self) -> None:
        with pytest.raises(SystemExit):
            run.parse_args(["--days", "-1"])

    def test_a_non_numeric_number_of_days_is_refused(self) -> None:
        with pytest.raises(SystemExit):
            run.parse_args(["--days", "abc"])

    def test_an_unknown_flag_is_refused(self) -> None:
        with pytest.raises(SystemExit):
            run.parse_args(["--forever"])

    def test_the_text_flag_is_gone(self) -> None:
        with pytest.raises(SystemExit):
            run.parse_args(["--text"])


class TestMain:
    @pytest.fixture(autouse=True)
    def no_server(self, monkeypatch) -> None:
        """Keep the bundled model server out of the tests that only read options.

        Without this the command line is followed all the way into a real server
        launch, which costs a couple of seconds a test and makes the suite pass or
        fail depending on whether a model happens to be installed.
        """

        @contextmanager
        def fake_server(_config: object) -> Iterator[object]:
            yield type("C", (), {"base_url": "http://127.0.0.1:8080"})

        monkeypatch.setattr(run, "running_server", fake_server)

    def test_the_interface_is_opened(self, monkeypatch) -> None:
        opened = []
        monkeypatch.setattr(run, "build_app", lambda **kwargs: _recording(opened, **kwargs))

        run.main([])

        assert opened

    def test_the_interface_receives_the_seeded_world(self, monkeypatch) -> None:
        opened = []
        monkeypatch.setattr(run, "build_app", lambda **kwargs: _recording(opened, **kwargs))

        run.main(["--seed", "5"])

        world = opened[0]["world"]
        assert [village.seed for village in world.villages] == [5, 6, 7]

    def test_the_run_returns_zero(self, monkeypatch) -> None:
        monkeypatch.setattr(run, "build_app", lambda **kwargs: _recording([]))

        assert run.main([]) == 0

    def test_a_log_file_is_written_when_asked(self, monkeypatch, tmp_path: Path) -> None:
        target = tmp_path / "run.txt"
        monkeypatch.setattr(run, "build_app", lambda **kwargs: _finished(target))

        run.main(["--log", str(target)])

        assert target.is_file()
        assert "days simulated" in target.read_text()

    def test_nothing_is_written_without_the_flag(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setattr(run, "build_app", lambda **kwargs: _finished(None))

        run.main([])

        assert not list(tmp_path.iterdir())

    def test_the_number_of_days_reaches_the_interface(self, monkeypatch) -> None:
        opened = []
        monkeypatch.setattr(run, "build_app", lambda **kwargs: _recording(opened, **kwargs))

        run.main(["--days", "12"])

        assert opened[0]["days"] == 12

    def test_no_days_means_the_interface_waits_for_the_keys(self, monkeypatch) -> None:
        opened = []
        monkeypatch.setattr(run, "build_app", lambda **kwargs: _recording(opened, **kwargs))

        run.main([])

        assert opened[0]["days"] == 0

    def test_the_seed_is_recorded_in_the_log(self, monkeypatch, tmp_path: Path) -> None:
        target = tmp_path / "run.txt"
        monkeypatch.setattr(run, "build_app", lambda **kwargs: _finished(target))

        run.main(["--seed", "5", "--log", str(target)])

        assert "seed: 5" in target.read_text()

    def test_the_log_keeps_the_barters_and_the_prose(self, monkeypatch, tmp_path: Path) -> None:
        target = tmp_path / "run.txt"
        monkeypatch.setattr(run, "build_app", lambda **kwargs: _narrated())

        run.main(["--log", str(target)])

        text = target.read_text()
        assert "Miners -> Farmers" in text
        assert "A day in the world." in text


class TestBackendChoice:
    """A run always needs a model, so the question is only who serves it."""

    def test_without_a_url_the_bundled_server_is_started(self, monkeypatch) -> None:
        started = []
        used = []

        class FakeConfig:
            base_url = "http://127.0.0.1:8080"

        @contextmanager
        def fake_server(_config: object) -> Iterator[object]:
            started.append(True)
            yield FakeConfig()

        def fake_narrator(base_url: str = "") -> LLMNarrator:
            used.append(base_url)
            return LLMNarrator(client=LLMClient(base_url=base_url))

        monkeypatch.delenv(BASE_URL_ENV, raising=False)
        monkeypatch.setattr(run, "running_server", fake_server)
        monkeypatch.setattr(run, "build_narrator", fake_narrator)
        monkeypatch.setattr(run, "build_app", lambda **kwargs: _recording([]))

        run.main([])

        assert started
        assert used == ["http://127.0.0.1:8080"]

    def test_a_url_in_the_environment_means_no_bundled_server(self, monkeypatch) -> None:
        def fail(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("must not start a server")

        used = []

        def fake_narrator(base_url: str = "") -> LLMNarrator:
            used.append(base_url)
            return LLMNarrator(client=LLMClient(base_url=base_url))

        monkeypatch.setenv(BASE_URL_ENV, "http://127.0.0.1:9999")
        monkeypatch.setattr(run, "running_server", fail)
        monkeypatch.setattr(run, "build_narrator", fake_narrator)
        monkeypatch.setattr(run, "build_app", lambda **kwargs: _recording([]))

        run.main([])

        assert used == [""]

    def test_an_empty_url_still_starts_the_bundled_server(self, monkeypatch) -> None:
        started = []

        @contextmanager
        def fake_server(_config: object) -> Iterator[object]:
            started.append(True)
            yield type("C", (), {"base_url": "http://127.0.0.1:8080"})

        monkeypatch.setenv(BASE_URL_ENV, "   ")
        monkeypatch.setattr(run, "running_server", fake_server)
        monkeypatch.setattr(run, "build_app", lambda **kwargs: _recording([]))

        run.main([])

        assert started

    def test_a_blank_url_is_not_taken_as_one(self, monkeypatch) -> None:
        monkeypatch.setenv(BASE_URL_ENV, "  ")

        assert run.wants_llm() is False

    def test_a_real_url_is_taken_as_one(self, monkeypatch) -> None:
        monkeypatch.setenv(BASE_URL_ENV, "http://127.0.0.1:9999")

        assert run.wants_llm() is True


def _recording(opened: list[dict], **kwargs) -> FakeApp:
    opened.append(kwargs)
    return FakeApp([], _journal())


def _finished(_path: Path | None) -> FakeApp:
    journal = RunLog()
    journal.record(DayEntry(day=1, spread=0.0, verdict="the world in balance"))
    return FakeApp(list(journal.entries), journal)


def _narrated() -> FakeApp:
    journal = RunLog()
    journal.record(
        DayEntry(
            day=1,
            spread=0.0,
            verdict="the world in balance",
            trades=("Miners -> Farmers: 20.00 Minerals for 13.33 Agriculture",),
            narration=("A day in the world.",),
        )
    )
    return FakeApp(list(journal.entries), journal)


def _journal() -> RunLog:
    return RunLog()


__all__ = ["LLMNarrator"]