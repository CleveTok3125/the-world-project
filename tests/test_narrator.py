from __future__ import annotations

import io
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from world.goods import ALL_GOODS, Goods, zero_stock
from world.llm import API_KEY_ENV, BASE_URL_ENV, MODEL_ENV, LLMClient
from world.narrator import (
    SEVERE_DEPRIVATION,
    LLMNarrator,
    LostTheFacts,
    TradeDigest,
    WorldSummary,
    build_narrator,
    keeps_the_figures,
    situation,
    summarize,
)
from world.world import build_default_world

CHAT_REPLY = (
    '{"choices": [{"message": {"role": "assistant", "content": "%s"}}], "usage": {}}'
)


class FakeResponse(io.BytesIO):
    """Minimal stand in for the object returned by ``urlopen``."""

    def __init__(self, body: str) -> None:
        super().__init__(body.encode())


class ScriptedClient:
    """A model that always says the same thing, standing in for a real server."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.asked: list[tuple[str, str]] = []

    def ask(self, system: str, user: str) -> str:
        self.asked.append((system, user))
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]

    def ask_json(self, system: str, user: str) -> object:
        self.asked.append((system, user))
        return json.loads(self.answers.pop(0) if len(self.answers) > 1 else self.answers[0])


def quiet_summary():
    """A day with no barter in it, so nothing in the reply needs checking.

    Used by the tests that are about how the request is built rather than about
    whether the answer kept its figures.
    """
    world = _silent_day()
    return summarize(world, world.advance_day())


def chat_reply(text: str) -> str:
    """Wrap prose in the shape an OpenAI chat completion returns."""
    return CHAT_REPLY % text


class ScriptedNarrator:
    """A narrator that reports what it was told, so tests do not need a model."""

    def __init__(self, lines: list[str] | None = None) -> None:
        self.lines = lines or ["A day in the world."]

    def describe(self, summary) -> list[str]:
        return list(self.lines)


def summary_after(days: int = 0):
    world = build_default_world()
    for _ in range(days):
        world.advance_day()
    return summarize(world, world.advance_day())


def _starved_in(goods: Goods, short_by: float):
    """A world whose last village cannot get ``short_by`` units of that commodity.

    Production is switched off and the stores are emptied, otherwise the specialist
    village simply makes the goods back before the day is over. What is left is
    counted as a shortage when the village eats, which is where the day record
    takes it from.
    """
    world = _silent_day()
    need = world.villages[2].daily_consumption(goods)
    world.villages[2].stock = {**world.villages[2].stock, goods: need - short_by}
    world.villages[2].production = {
        item: (0.0 if item is goods else amount)
        for item, amount in world.villages[2].production.items()
    }
    return world


def _silent_day():
    """A world that runs one day with no barter at all.

    A village in need is served a meeting first even when ``trade_chance`` is zero,
    so that setting on its own does not stop a shortage from being traded away. The
    only way to see the day's shortages as they were produced is to give the day no
    hours at all.
    """
    world = build_default_world()
    world.time_per_day = 0
    return world


class TestSummary:
    def test_the_day_number_is_carried(self) -> None:
        world = build_default_world()
        world.advance_day()

        summary = summarize(world, world.advance_day())

        assert summary.day == 2

    def test_every_village_is_listed_with_its_numbers(self) -> None:
        world = build_default_world()

        summary = summarize(world, world.advance_day())

        assert {village.name for village in summary.villages} == {
            village.name for village in world.villages
        }
        for figures in summary.villages:
            assert figures.satisfied >= 0.0
            assert figures.missing >= 0.0

    def test_the_spread_between_villages_is_carried(self) -> None:
        world = build_default_world()

        summary = summarize(world, world.advance_day())

        assert summary.spread >= 0.0

    def test_an_agreed_barter_is_summarised(self) -> None:
        world = build_default_world()
        world.trade_chance = 1.0

        summary = summarize(world, world.advance_day())

        assert len(summary.trades) >= 1
        trade = summary.trades[0]
        assert trade.agreed
        assert trade.given > 0
        assert trade.taken > 0
        assert trade.given_goods is not trade.taken_goods

    def test_a_refused_barter_is_summarised_too(self) -> None:
        world = build_default_world()
        world.trade_chance = 1.0
        for village in world.villages:
            village.production = zero_stock()

        summary = summarize(world, world.advance_day())

        assert summary.trades
        assert all(not trade.agreed for trade in summary.trades)
        assert all(trade.reason for trade in summary.trades)

    def test_shortages_are_listed_per_village(self) -> None:
        world = build_default_world()
        world.villages[0].stock = zero_stock()
        world.villages[0].production = zero_stock()

        summary = summarize(world, world.advance_day())

        assert summary.shortages
        assert summary.shortages[0].name == "Farmers"
        assert set(summary.shortages[0].goods) == set(ALL_GOODS)


class TestLLMNarrator:
    def test_the_reply_is_split_into_lines(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data)
            captured["headers"] = request.headers
            return FakeResponse(CHAT_REPLY % "First line.\\n\\nSecond line.\\n")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        lines = LLMNarrator(client=LLMClient(model="m")).describe(quiet_summary())

        assert lines == ["First line.", "Second line."]

    def test_it_posts_to_the_openai_chat_completions_endpoint(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            return FakeResponse(CHAT_REPLY % "A day.")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        LLMNarrator(client=LLMClient(model="m", base_url="http://127.0.0.1:8080")).describe(quiet_summary())

        assert captured["url"] == "http://127.0.0.1:8080/v1/chat/completions"

    def test_a_trailing_slash_in_the_url_does_not_double_up(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            return FakeResponse(CHAT_REPLY % "A day.")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        LLMNarrator(client=LLMClient(model="m", base_url="http://127.0.0.1:8080/")).describe(quiet_summary())

        assert captured["url"] == "http://127.0.0.1:8080/v1/chat/completions"

    def test_the_request_carries_the_model_and_the_system_prompt(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def fake_urlopen(request, timeout=None):
            captured["body"] = json.loads(request.data)
            return FakeResponse(CHAT_REPLY % "A day.")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        LLMNarrator(client=LLMClient(model="qwen2.5")).describe(quiet_summary())

        body = captured["body"]
        assert body["model"] == "qwen2.5"
        assert body["stream"] is False
        assert body["messages"][0]["role"] == "system"
        assert "narrator" in body["messages"][0]["content"]

    def test_a_server_serving_one_model_may_be_left_unnamed(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def fake_urlopen(request, timeout=None):
            captured["body"] = json.loads(request.data)
            return FakeResponse(CHAT_REPLY % "A day.")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        assert LLMNarrator(client=LLMClient(model="")).describe(quiet_summary())
        assert captured["body"]["model"]

    def test_the_summary_is_sent_as_the_user_message(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def fake_urlopen(request, timeout=None):
            captured["body"] = json.loads(request.data)
            return FakeResponse(CHAT_REPLY % "A day.")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        summary = quiet_summary()

        LLMNarrator(client=LLMClient(model="m")).describe(summary)

        prompt = json.loads(captured["body"]["messages"][1]["content"])
        assert prompt["day"] == summary.day
        assert len(prompt["villages"]) == len(summary.villages)
        assert len(prompt["trades"]) == len(summary.trades)

    def test_explicit_generation_options_override_the_default_temperature(
        self, monkeypatch
    ) -> None:
        captured: dict[str, object] = {}

        def fake_urlopen(request, timeout=None):
            captured["body"] = json.loads(request.data)
            return FakeResponse(CHAT_REPLY % "A day.")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        LLMNarrator(client=LLMClient(model="m", options={"temperature": 0.1})).describe(quiet_summary())

        assert captured["body"]["temperature"] == 0.1

    def test_the_api_key_becomes_a_bearer_token(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def fake_urlopen(request, timeout=None):
            captured["headers"] = request.headers
            return FakeResponse(CHAT_REPLY % "A day.")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        LLMNarrator(client=LLMClient(model="m", api_key="secret-token")).describe(quiet_summary())

        assert captured["headers"]["Authorization"] == "Bearer secret-token"

    def test_no_api_key_means_no_authorization_header(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def fake_urlopen(request, timeout=None):
            captured["headers"] = request.headers
            return FakeResponse(CHAT_REPLY % "A day.")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        LLMNarrator(client=LLMClient(model="m")).describe(quiet_summary())

        assert "Authorization" not in captured["headers"]

    def test_an_unreachable_server_is_reported_clearly(self, monkeypatch) -> None:
        def fake_urlopen(request, timeout=None):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        with pytest.raises(RuntimeError, match="unreachable"):
            LLMNarrator(client=LLMClient(model="m")).describe(quiet_summary())

    def test_a_reply_without_choices_yields_no_lines(self, monkeypatch) -> None:
        monkeypatch.setattr(
            urllib.request, "urlopen", lambda request, timeout=None: FakeResponse("{}")
        )

        assert LLMNarrator(client=LLMClient(model="m")).describe(quiet_summary()) == []

    def test_it_reaches_a_real_http_server(self) -> None:
        received: list[dict] = []
        reply = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": "All traded well."}}]}
        )

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers["Content-Length"])
                received.append(json.loads(self.rfile.read(length)))
                body = reply.encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_: object) -> None:
                return

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            narrator = LLMNarrator(
                client=LLMClient(
                    model="m",
                    base_url=f"http://127.0.0.1:{server.server_port}",
                    timeout=10,
                )
            )
            lines = narrator.describe(quiet_summary())
        finally:
            server.shutdown()
            server.server_close()

        assert lines == ["All traded well."]
        assert received[0]["model"] == "m"


class TestTradeSentence:
    def test_an_agreed_trade_names_both_directions(self) -> None:
        trade = summary_after().trades[0]
        assert trade.agreed

        sentence = trade.sentence()

        assert trade.proposer in sentence
        assert trade.responder in sentence
        assert "gave" in sentence
        assert "received" in sentence
        assert trade.given_goods.label in sentence
        assert trade.taken_goods.label in sentence

    def test_the_giver_comes_before_the_taker(self) -> None:
        trade = next(t for t in summary_after().trades if t.agreed)

        assert trade.sentence().index("gave") < trade.sentence().index("received")

    def test_a_refused_trade_names_the_reason(self) -> None:
        world = build_default_world()
        for village in world.villages:
            village.production = zero_stock()
        summary = summarize(world, world.advance_day())

        sentence = summary.trades[0].sentence()

        assert "could not agree" in sentence
        assert summary.trades[0].reason.split()[0] in sentence

    def test_a_refused_trade_without_a_reason_still_reads(self) -> None:
        trade = TradeDigest(
            proposer="A", responder="B", agreed=False,
            given=0.0, given_goods=None, taken=0.0, taken_goods=None, rate=0.0,
        )

        assert "could not agree" in trade.sentence()


class TestPayloadIsPlainProse:
    def test_no_trade_appears_as_a_bare_field_pair(self) -> None:
        summary = summary_after()

        payload = summary.to_payload()

        assert all(isinstance(line, str) for line in payload["trades"])
        assert all(isinstance(line, str) for line in payload["villages"])

    def test_the_payload_keeps_every_trade(self) -> None:
        summary = summary_after()

        assert len(summary.to_payload()["trades"]) == len(summary.trades)

    def test_the_payload_stays_small(self) -> None:
        import json as json_module

        summary = summary_after()

        assert len(json_module.dumps(summary.to_payload())) < 1200


class TestReasoningModels:
    def test_a_reply_that_only_thought_is_still_used(self, monkeypatch) -> None:
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": "The Miners had the surplus.",
                        }
                    }
                ]
            }
        )
        monkeypatch.setattr(
            urllib.request, "urlopen", lambda request, timeout=None: FakeResponse(body)
        )

        assert LLMNarrator(client=LLMClient(model="m")).describe(quiet_summary()) == [
            "The Miners had the surplus."
        ]

    def test_thinking_is_off_unless_asked_for(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def fake_urlopen(request, timeout=None):
            captured["body"] = json.loads(request.data)
            return FakeResponse(CHAT_REPLY % "A day.")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        LLMNarrator(client=LLMClient(model="m")).describe(quiet_summary())

        assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}

    def test_thinking_can_be_switched_on(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def fake_urlopen(request, timeout=None):
            captured["body"] = json.loads(request.data)
            return FakeResponse(CHAT_REPLY % "A day.")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        LLMNarrator(client=LLMClient(model="m", thinking=True)).describe(quiet_summary())

        assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": True}

    def test_the_reply_length_is_bounded(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def fake_urlopen(request, timeout=None):
            captured["body"] = json.loads(request.data)
            return FakeResponse(CHAT_REPLY % "A day.")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        LLMNarrator(client=LLMClient(model="m", max_tokens=120)).describe(quiet_summary())

        assert captured["body"]["max_tokens"] == 120


class TestLiteralValuesAreKept:
    def test_a_reply_keeping_every_figure_is_accepted(self) -> None:
        summary = summary_after()
        lines = [trade.sentence() for trade in summary.trades]

        assert keeps_the_figures(lines, summary)

    def test_a_reply_spelling_numbers_out_is_rejected(self) -> None:
        summary = summary_after()

        spelled = ["Farmers traded twenty minerals for thirteen agriculture."]

        assert not keeps_the_figures(spelled, summary)

    def test_a_reply_dropping_a_quantity_is_rejected(self) -> None:
        summary = summary_after()

        assert not keeps_the_figures(["Nothing was traded today."], summary)

    def test_a_reply_renaming_a_commodity_is_rejected(self) -> None:
        summary = summary_after()
        goods = next(trade.given_goods for trade in summary.trades if trade.agreed)
        renamed = str(goods.label).upper()

        assert not keeps_the_figures(
            [f"Farmers gave Miners 20.0 {renamed} and received 13.3 crops."], summary
        )

    def test_a_refusal_needs_no_figures(self) -> None:
        world = build_default_world()
        for village in world.villages:
            village.production = zero_stock()
        summary = summarize(world, world.advance_day())

        assert keeps_the_figures(["Nobody agreed to anything."], summary)

    def test_a_day_without_villages_needs_no_figures(self) -> None:
        world = build_default_world()
        world.villages = ()

        assert keeps_the_figures(["A quiet day."], summarize(world, world.advance_day()))

    def test_a_reply_that_loses_the_figures_is_refused(self, monkeypatch) -> None:
        body = json.dumps(
            {"choices": [{"message": {"content": "Farmers traded twenty minerals."}}]}
        )
        monkeypatch.setattr(
            urllib.request, "urlopen", lambda request, timeout=None: FakeResponse(body)
        )
        narrator = LLMNarrator(client=LLMClient(model="m"))

        with pytest.raises(LostTheFacts):
            narrator.describe(summary_after())

    def test_a_reply_that_loses_the_figures_carries_the_bad_reply(self, monkeypatch) -> None:
        body = json.dumps(
            {"choices": [{"message": {"content": "Farmers traded twenty minerals."}}]}
        )
        monkeypatch.setattr(
            urllib.request, "urlopen", lambda request, timeout=None: FakeResponse(body)
        )
        narrator = LLMNarrator(client=LLMClient(model="m"))

        with pytest.raises(LostTheFacts) as caught:
            narrator.describe(summary_after())

        assert "twenty minerals" in str(caught.value)

    def test_a_good_reply_is_kept_as_it_is(self, monkeypatch) -> None:
        summary = summary_after()
        reply = " ".join(trade.sentence() for trade in summary.trades)
        body = json.dumps({"choices": [{"message": {"content": reply}}]})
        monkeypatch.setattr(
            urllib.request, "urlopen", lambda request, timeout=None: FakeResponse(body)
        )
        narrator = LLMNarrator(client=LLMClient(model="m"))

        assert narrator.describe(summary) == [reply]


class TestSituation:
    def test_a_fully_supplied_even_world_is_calm(self) -> None:
        world = build_default_world()
        for _ in range(6):
            world.advance_day()
        summary = summarize(world, world.advance_day())

        found = situation(summary)

        assert found.key == "calm"
        assert found.commodity is None
        assert found.villages == ()
        assert found.severity == 0.0

    def test_a_single_empty_good_is_a_mild_shortage(self) -> None:
        world = _starved_in(Goods.CRAFT, 0.5)
        summary = summarize(world, world.advance_day())

        found = situation(summary)

        assert found.key == "shortage"
        assert found.villages == ("Artisans",)
        assert found.commodity is Goods.CRAFT
        assert 0 < found.severity < 0.25

    def test_a_small_gap_is_never_called_severe(self) -> None:
        """Missing half a unit at the end of the reserves is not deprivation."""
        world = _starved_in(Goods.CRAFT, 0.5)
        found = situation(summarize(world, world.advance_day()))

        assert found.key == "shortage"
        assert found.severity < 0.25

    def test_severity_counts_what_went_missing_not_what_is_left(self) -> None:
        world = _silent_day()
        village = world.villages[2]
        need = village.daily_consumption(Goods.CRAFT)
        village.stock = {**village.stock, Goods.CRAFT: need - need / 2}
        village.production = {
            item: (0.0 if item is Goods.CRAFT else amount)
            for item, amount in village.production.items()
        }
        found = situation(summarize(world, world.advance_day()))

        lost = Goods.CRAFT.base_value * (need / 2)
        total = sum(
            goods.base_value * world.villages[0].daily_consumption(goods)
            for goods in ALL_GOODS
        )
        assert found.severity == pytest.approx(lost / total)

    def test_a_deep_shortage_is_called_severe(self) -> None:
        world = build_default_world()
        world.villages[2].stock = zero_stock()
        world.villages[2].production = zero_stock()
        summary = summarize(world, world.advance_day())

        found = situation(summary)

        assert found.key == "severe"
        assert found.severity >= 0.25

    def test_the_deepest_village_decides_the_verdict(self) -> None:
        """Two villages short of different things: the worse of them is named."""
        world = _silent_day()
        world.villages[1].stock[Goods.MINERAL] = 9.5
        world.villages[1].production[Goods.MINERAL] = 0.0
        world.villages[2].stock[Goods.CRAFT] = 4.0
        world.villages[2].production[Goods.CRAFT] = 0.0
        summary = summarize(world, world.advance_day())

        found = situation(summary)

        assert found.villages == ("Artisans",)
        assert found.commodity is Goods.CRAFT

    def test_two_villages_short_of_one_commodity_are_reported_together(self) -> None:
        world = _silent_day()
        world.villages[1].stock[Goods.MINERAL] = 9.0
        world.villages[1].production[Goods.MINERAL] = 0.0
        world.villages[2].stock[Goods.MINERAL] = 5.0
        world.villages[2].production[Goods.MINERAL] = 0.0
        summary = summarize(world, world.advance_day())

        found = situation(summary)

        assert found.key == "shared"
        assert found.villages == ("Miners", "Artisans")
        assert found.commodity is Goods.MINERAL

    def test_the_shared_verdict_names_every_village_involved(self) -> None:
        world = _silent_day()
        for village in world.villages:
            village.stock[Goods.CRAFT] = 6.0
            village.production[Goods.CRAFT] = 0.0
        summary = summarize(world, world.advance_day())

        assert situation(summary).label == (
            "a shortage of Handicrafts shared by Farmers, Miners and Artisans"
        )

    def test_two_villages_short_of_different_commodities_are_not_shared(self) -> None:
        world = _silent_day()
        world.villages[0].stock[Goods.MINERAL] = 9.5
        world.villages[0].production[Goods.MINERAL] = 0.0
        world.villages[1].stock[Goods.CRAFT] = 9.5
        world.villages[1].production[Goods.CRAFT] = 0.0
        summary = summarize(world, world.advance_day())

        assert situation(summary).key != "shared"

    def test_the_shared_verdict_reaches_the_model_as_one_label(self) -> None:
        world = _silent_day()
        for village in world.villages:
            village.stock[Goods.CRAFT] = 6.0
            village.production[Goods.CRAFT] = 0.0
        summary = summarize(world, world.advance_day())
        client = ScriptedClient("A quiet day in the world.")

        LLMNarrator(client=client).describe(summary)

        assert situation(summary).label in client.asked[0][1]

class TestSurplus:
    def test_a_world_sitting_on_a_pile_up_is_called_surplus(self) -> None:
        world = _silent_day()
        for village in world.villages:
            village.stock[Goods.CRAFT] = 120.0
        summary = summarize(world, world.advance_day())

        found = situation(summary)

        assert found.key == "surplus"
        assert found.commodity is Goods.CRAFT
        assert found.severity >= 4.0

    def test_the_holders_of_the_pile_are_named_together(self) -> None:
        world = _silent_day()
        for village in world.villages:
            village.stock[Goods.CRAFT] = 120.0
        summary = summarize(world, world.advance_day())

        assert situation(summary).label == (
            "a surplus of Handicrafts held by Farmers, Miners and Artisans"
        )

    def test_one_village_sitting_on_a_pile_is_named_alone(self) -> None:
        """The other two hold only what they need, so they are not holders."""
        world = _silent_day()
        for village in world.villages[:2]:
            village.stock[Goods.MINERAL] = village.target_stock(Goods.MINERAL)
            village.production[Goods.MINERAL] = 0.0
        world.villages[2].stock[Goods.MINERAL] = 260.0
        summary = summarize(world, world.advance_day())

        found = situation(summary)

        assert found.key == "surplus"
        assert found.villages == ("Artisans",)
        assert found.label == "a surplus of Minerals held by Artisans"

    def test_an_ordinary_ledge_is_not_a_surplus(self) -> None:
        world = _silent_day()
        summary = summarize(world, world.advance_day())

        assert situation(summary).key == "calm"

    def test_a_shortage_outranks_a_pile_up(self) -> None:
        """Goods in the stores are a problem the day has already solved."""
        world = _silent_day()
        for village in world.villages:
            village.stock[Goods.CRAFT] = 120.0
        world.villages[0].stock[Goods.FARM] = 0.0
        world.villages[0].production[Goods.FARM] = 0.0
        summary = summarize(world, world.advance_day())

        assert situation(summary).key in {"shortage", "severe"}

    def test_the_pile_up_is_measured_across_the_whole_world(self) -> None:
        """The same units in one village's hands are not the pile three villages hold."""
        world = _silent_day()
        for village in world.villages:
            village.stock[Goods.CRAFT] = 20.0
        ledge = summarize(world, world.advance_day())

        world = _silent_day()
        for village in world.villages:
            village.stock[Goods.CRAFT] = 120.0
        pile = summarize(world, world.advance_day())

        assert situation(ledge).key == "calm"
        assert situation(pile).key == "surplus"

    def test_the_closing_line_names_the_surplus(self) -> None:
        world = _silent_day()
        for village in world.villages:
            village.stock[Goods.CRAFT] = 120.0
        summary = summarize(world, world.advance_day())
        client = ScriptedClient("A quiet day in the world.")

        LLMNarrator(client=client).describe(summary)

        assert situation(summary).label in client.asked[0][1]

    def test_the_surplus_reaches_the_summary_the_model_sees(self) -> None:
        world = _silent_day()
        for village in world.villages:
            village.stock[Goods.CRAFT] = 120.0
        summary = summarize(world, world.advance_day())

        assert "surplus" in summary.to_payload()["situation"]

    def test_a_tie_on_quantity_breaks_towards_the_dearest_good(self) -> None:
        world = build_default_world()
        world.villages[2].stock = zero_stock()
        world.villages[2].production = zero_stock()
        summary = summarize(world, world.advance_day())

        assert situation(summary).commodity is Goods.CRAFT

    def test_a_world_without_villages_is_still_classified(self) -> None:
        world = build_default_world()
        world.villages = ()

        assert situation(summarize(world, world.advance_day())).key == "calm"

    def test_the_label_reads_as_a_sentence_fragment(self) -> None:
        world = build_default_world()
        for _ in range(6):
            world.advance_day()

        assert situation(summarize(world, world.advance_day())).label

    def test_a_shortage_label_names_the_commodity(self) -> None:
        world = _starved_in(Goods.CRAFT, 0.5)
        found = situation(summarize(world, world.advance_day()))

        assert Goods.CRAFT.label in found.label

    def test_no_state_exists_between_balance_and_shortage(self) -> None:
        world = build_default_world()
        seen = set()
        for index in (0, 1, 2):
            world.villages[index].stock[Goods.FARM] = float(10 - index)
            seen.add(situation(summarize(world, world.advance_day())).key)
            world.villages[index].stock[Goods.FARM] = 20.0

        assert seen <= {"calm", "shortage", "severe"}


class TestTheModelIsAskedForAVerdict:
    """The closing line is the model's job now, so the prompt has to ask for it."""

    def test_the_prompt_asks_for_one_final_verdict(self) -> None:
        prompt = LLMNarrator.SYSTEM_PROMPT.lower()

        assert "verdict" in prompt
        assert "final" in prompt

    def test_the_prompt_forbids_inventing_numbers(self) -> None:
        prompt = LLMNarrator.SYSTEM_PROMPT.lower()

        assert "invent nothing" in prompt
        assert "digits" in prompt

    def test_the_prompt_forbids_swapping_the_direction_of_a_trade(self) -> None:
        assert "direction" in LLMNarrator.SYSTEM_PROMPT.lower()

    def test_the_situation_reaches_the_model_as_a_verdict_line(self) -> None:
        world = _starved_in(Goods.CRAFT, 10.0)
        summary = summarize(world, world.advance_day())
        client = ScriptedClient(chat_reply("Farmers could not lay hands on 10.0 Handicrafts. "
                                          "The day ends in a shortage of Handicrafts."))

        LLMNarrator(client=client).describe(summary)

        assert situation(summary).label in client.asked[0][1]


class TestSituationReachesTheModel:
    def a_famine(self) -> WorldSummary:
        world = build_default_world()
        for village in world.villages:
            for goods in ALL_GOODS:
                village.consumption[goods] *= 10
                village.stock[goods] = 0.0
        return summarize(world, world.advance_day())

    def test_the_payload_carries_the_severity(self) -> None:
        summary = self.a_famine()

        payload = summary.to_payload()

        assert payload["severity"] == round(situation(summary).severity, 2)
        assert payload["severity"] > SEVERE_DEPRIVATION

    def test_a_calm_day_carries_no_severity(self) -> None:
        world = build_default_world()

        assert summarize(world, world.advance_day()).to_payload()["severity"] == 0.0

    def test_the_prompt_asks_the_narrator_to_weigh_the_day_by_it(self) -> None:
        prompt = LLMNarrator.SYSTEM_PROMPT.lower()

        assert "severity" in prompt
        assert "0.5 is a famine" in prompt

    def test_the_prompt_forbids_calling_a_shortage_a_delivery(self) -> None:
        assert "went without" in LLMNarrator.SYSTEM_PROMPT.lower()

    def test_a_shortage_is_told_to_the_model_as_something_lost(self) -> None:
        payload = self.a_famine().to_payload()

        assert "went without" in payload["shortages"][0]
        assert "received" not in payload["shortages"][0]

    def test_the_payload_carries_the_situation(self) -> None:
        world = build_default_world()
        for _ in range(6):
            world.advance_day()
        summary = summarize(world, world.advance_day())

        assert summary.to_payload()["situation"] == situation(summary).label

    def test_the_prompt_asks_for_a_closing_verdict(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def fake_urlopen(request, timeout=None):
            captured["body"] = json.loads(request.data)
            return FakeResponse(CHAT_REPLY % "A day.")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        narrator = LLMNarrator(client=LLMClient(model="m"))

        with pytest.raises(LostTheFacts):
            narrator.describe(summary_after())

        assert "verdict" in captured["body"]["messages"][0]["content"]


class TestBuildNarrator:
    def test_the_narrator_is_a_model(self) -> None:
        assert isinstance(build_narrator(), LLMNarrator)

    def test_the_model_comes_from_the_environment(self, monkeypatch) -> None:
        monkeypatch.setenv(MODEL_ENV, "gemma-4-e2b")

        assert build_narrator().client.model == "gemma-4-e2b"

    def test_an_explicit_url_wins_over_the_environment(self, monkeypatch) -> None:
        monkeypatch.setenv(BASE_URL_ENV, "http://127.0.0.1:9999")

        narrator = build_narrator(base_url="http://127.0.0.1:8080")

        assert narrator.client.base_url == "http://127.0.0.1:8080"

    def test_the_url_falls_back_to_the_environment(self, monkeypatch) -> None:
        monkeypatch.setenv(BASE_URL_ENV, "http://127.0.0.1:9999")

        assert build_narrator().client.base_url == "http://127.0.0.1:9999"

    def test_an_api_key_from_the_environment_is_used(self, monkeypatch) -> None:
        monkeypatch.setenv(API_KEY_ENV, "secret-token")

        assert build_narrator().client.api_key == "secret-token"

    def test_no_server_is_contacted_while_building(self, monkeypatch) -> None:
        def explode(*args, **kwargs):
            raise AssertionError("building the narrator must not talk to anything")

        monkeypatch.setattr(urllib.request, "urlopen", explode)

        assert isinstance(build_narrator(), LLMNarrator)


class TestSummaryIsDetached:
    def test_the_summary_does_not_follow_later_changes(self) -> None:
        world = build_default_world()
        summary = summarize(world, world.advance_day())
        expected = summary.villages

        world.villages[0].stock[Goods.FARM] = 999.0

        assert summary.villages == expected