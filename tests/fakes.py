"""Stand ins used by tests that must not talk to a language model."""

from __future__ import annotations

import json

from world.llm import NoJSONAnswer
from world.narrator import WorldSummary


class FakeNarrator:
    """A narrator that reports what it was told, so no model is needed.

    Attributes:
        lines: What every call returns.
        seen: The summaries it was handed, in order.
    """

    def __init__(self, lines: list[str] | None = None) -> None:
        self.lines = lines or ["A day in the world."]
        self.seen: list[WorldSummary] = []

    def describe(self, summary: WorldSummary) -> list[str]:
        self.seen.append(summary)
        return list(self.lines)


class FakeModel:
    """A model that answers with a set of changes, standing in for a real server.

    Attributes:
        answers: The plans to hand back, one per call, last one repeating.
        asked: The prompts it was given, in order.
    """

    def __init__(self, *answers: object) -> None:
        self.answers = list(answers)
        self.asked: list[str] = []

    def ask(self, system: str, user: str) -> str:
        self.asked.append(user)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return answer if isinstance(answer, str) else json.dumps(answer)

    def ask_json(self, system: str, user: str) -> object:
        answer = self.ask(system, user)
        try:
            return json.loads(answer)
        except json.JSONDecodeError:
            raise NoJSONAnswer(user, answer) from None


def plan(*changes: dict, reply: str = "done", schedule: list | None = None) -> dict:
    """Build the shape a director expects back from a model."""
    return {"reply": reply, "changes": list(changes), "schedule": schedule or []}


def change(village: str | None, field: str, value: object) -> dict:
    """Build one change the way a model writes it."""
    return {"village": village, "field": field, "value": value}