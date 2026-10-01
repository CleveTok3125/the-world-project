"""Command line entry point: advance the world one day at a time."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

from world.director import Director
from world.llm import BASE_URL_ENV, LLMClient
from world.logbook import write_log
from world.narrator import build_narrator
from world.server import config_from_environment, running_server
from world.tui import WorldApp, build_app
from world.world import build_default_world

SEED_LIMIT = 2**31


class Options:
    """Everything the command line can change about a run.

    Attributes:
        days: Number of days to simulate, zero meaning step by hand.
        seed: Seed handed to the brains and to the day to day production waver.
        noise: Relative amplitude of the day to day production variation.
        log: File to write the run down to, empty when nothing is saved.
    """

    def __init__(
        self,
        days: int = 0,
        seed: int = 0,
        noise: float = 0.0,
        log: Path | None = None,
    ) -> None:
        self.days = days
        self.seed = seed
        self.noise = noise
        self.log = log

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Options):
            return NotImplemented
        return (self.days, self.seed, self.noise, self.log) == (
            other.days,
            other.seed,
            other.noise,
            other.log,
        )


def fresh_seed() -> int:
    """A seed drawn from the system, so two runs rarely look the same."""
    return random.randrange(SEED_LIMIT)


def parse_args(argv: list[str]) -> Options:
    """Read the command line into a value object."""
    parser = argparse.ArgumentParser(description="Three villages negotiating by barter.")
    parser.add_argument("--days", type=int, default=0, help="number of days to simulate")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed, drawn and printed when left out",
    )
    parser.add_argument("--noise", type=float, default=0.0, help="production noise")
    parser.add_argument(
        "--log", type=Path, default=None, help="write the run down to this file"
    )
    parsed = parser.parse_args(argv)
    if parsed.days < 0:
        parser.error("--days must be zero or greater")
    seed = fresh_seed() if parsed.seed is None else parsed.seed
    return Options(days=parsed.days, seed=seed, noise=parsed.noise, log=parsed.log)


def wants_llm() -> bool:
    """Whether the environment already points at a model server of its own.

    A run needs a language model, so the question is not whether to use one but
    whether to start the bundled one or talk to a server somebody else is running.
    """
    return bool(os.environ.get(BASE_URL_ENV, "").strip())


def main(argv: list[str] | None = None) -> int:
    """Run the simulation described by the command line."""
    options = parse_args(argv if argv is not None else _command_line())
    world = build_default_world(seed=options.seed, production_noise=options.noise)

    if wants_llm():
        return _run(world, options, build_narrator(), LLMClient.from_environment())
    with running_server(config_from_environment()) as config:
        base_url = config.base_url
        return _run(world, options, build_narrator(base_url), LLMClient.from_environment(base_url))


def _run(world, options: Options, narrator, client) -> int:
    """Open the interface, then save the run when the command line asked for it."""
    app = build_app(
        world=world,
        narrator=narrator,
        director=Director(client, world),
        days=options.days,
    )
    app.seed = options.seed
    app.journal.path = options.log
    app.journal.seed = options.seed
    app.run()
    _save(options.log, app)
    return 0


def _save(path: Path | None, app: WorldApp) -> None:
    """Write the finished run down, once, after the interface closes."""
    if path is None:
        return
    write_log(path, app.journal.entries, app.journal.summary_lines())
    print(f"run written to {path}")


def _command_line() -> list[str]:
    import sys

    return sys.argv[1:]


__all__ = ["Options", "fresh_seed", "main", "parse_args", "wants_llm"]


if __name__ == "__main__":
    raise SystemExit(main())