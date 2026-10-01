"""Time the model server, so a change of backend can be measured rather than felt.

Start with whatever the environment says: unset the GPU variables for the
processor, set them for a card. Both runs print the same lines, so the two can be
read against each other.
"""

from __future__ import annotations

import statistics
import time

from world.llm import LLMClient
from world.narrator import LLMNarrator, summarize
from world.server import config_from_environment, running_server
from world.world import build_default_world

CALLS = 3


def main() -> int:
    config = config_from_environment()
    print(f"model   : {config.model.name}")
    print(f"binary  : {config.binary}")
    print(f"gpu     : {config.gpu_layers + ' layers, forced' if config.gpu_layers else 'not forced, llama.cpp decides'}")

    with running_server(config):
        client = LLMClient(base_url=config.base_url, max_tokens=128)
        narrator = LLMNarrator(client=client)
        world = build_default_world(seed=1)

        times: list[float] = []
        for index in range(CALLS):
            report = world.advance_day()
            summary = summarize(world, report)
            started = time.perf_counter()
            try:
                lines = narrator.describe(summary)
            except Exception as problem:  # noqa: BLE001 - a benchmark wants the reason
                print(f"call {index + 1} failed: {problem}")
                continue
            took = time.perf_counter() - started
            times.append(took)
            print(f"call {index + 1}: {took:6.2f}s  {len(' '.join(lines))} chars")

    if not times:
        print("no call succeeded")
        return 1
    print(f"median  : {statistics.median(times):6.2f}s")
    print(f"fastest : {min(times):6.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
