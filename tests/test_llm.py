from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from world.llm import (
    API_KEY_ENV,
    BASE_URL_ENV,
    MODEL_ENV,
    NARRATION_TIMEOUT,
    LLMClient,
    ModelTimeout,
    ModelUnreachable,
    NoJSONAnswer,
)


def reply_body(text: str) -> str:
    """Wrap prose the way an OpenAI chat completion returns it."""
    return json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}]})


class FakeResponse(io.BytesIO):
    """Minimal stand in for the object returned by ``urlopen``."""

    def __init__(self, body: str) -> None:
        super().__init__(body.encode())


def replying(text: str):
    """Build a fake ``urlopen`` that always answers with ``text``."""
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["headers"] = request.headers
        captured["timeout"] = timeout
        return FakeResponse(reply_body(text))

    return fake_urlopen, captured


class TestAsking:
    def test_it_posts_to_the_chat_completions_endpoint(self, monkeypatch) -> None:
        fake, captured = replying("Hello.")
        monkeypatch.setattr(urllib.request, "urlopen", fake)

        LLMClient(base_url="http://127.0.0.1:8080").ask("be brief", "hi")

        assert captured["url"] == "http://127.0.0.1:8080/v1/chat/completions"

    def test_a_trailing_slash_does_not_double_up(self, monkeypatch) -> None:
        fake, captured = replying("Hello.")
        monkeypatch.setattr(urllib.request, "urlopen", fake)

        LLMClient(base_url="http://127.0.0.1:8080/").ask("be brief", "hi")

        assert captured["url"] == "http://127.0.0.1:8080/v1/chat/completions"

    def test_the_system_prompt_becomes_the_system_message(self, monkeypatch) -> None:
        fake, captured = replying("Hello.")
        monkeypatch.setattr(urllib.request, "urlopen", fake)

        LLMClient().ask("be brief", "hi")

        assert captured["body"]["messages"][0] == {"role": "system", "content": "be brief"}
        assert captured["body"]["messages"][1] == {"role": "user", "content": "hi"}

    def test_an_unnamed_model_still_sends_something(self, monkeypatch) -> None:
        fake, captured = replying("Hello.")
        monkeypatch.setattr(urllib.request, "urlopen", fake)

        LLMClient(model="").ask("be brief", "hi")

        assert captured["body"]["model"] == "local"

    def test_the_reply_comes_back_as_written(self, monkeypatch) -> None:
        fake, _ = replying("First.\n\nSecond.")
        monkeypatch.setattr(urllib.request, "urlopen", fake)

        assert LLMClient().ask("be brief", "hi") == "First.\n\nSecond."

    def test_a_reasoning_only_reply_is_still_an_answer(self, monkeypatch) -> None:
        body = json.dumps(
            {"choices": [{"message": {"content": "", "reasoning_content": "Thought."}}]}
        )
        monkeypatch.setattr(
            urllib.request, "urlopen", lambda request, timeout=None: FakeResponse(body)
        )

        assert LLMClient().ask("be brief", "hi") == "Thought."

    def test_a_reply_with_no_choices_is_empty(self, monkeypatch) -> None:
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda request, timeout=None: FakeResponse(json.dumps({"choices": []})),
        )

        assert LLMClient().ask("be brief", "hi") == ""


class TestSettings:
    def test_thinking_is_off_unless_asked_for(self, monkeypatch) -> None:
        fake, captured = replying("Hello.")
        monkeypatch.setattr(urllib.request, "urlopen", fake)

        LLMClient().ask("be brief", "hi")

        assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}

    def test_thinking_can_be_switched_on(self, monkeypatch) -> None:
        fake, captured = replying("Hello.")
        monkeypatch.setattr(urllib.request, "urlopen", fake)

        LLMClient(thinking=True).ask("be brief", "hi")

        assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": True}

    def test_the_reply_length_is_bounded(self, monkeypatch) -> None:
        fake, captured = replying("Hello.")
        monkeypatch.setattr(urllib.request, "urlopen", fake)

        LLMClient(max_tokens=120).ask("be brief", "hi")

        assert captured["body"]["max_tokens"] == 120

    def test_options_replace_the_default_temperature(self, monkeypatch) -> None:
        fake, captured = replying("Hello.")
        monkeypatch.setattr(urllib.request, "urlopen", fake)

        LLMClient(options={"temperature": 0.1}).ask("be brief", "hi")

        assert captured["body"]["temperature"] == 0.1

    def test_a_temperature_is_sent_when_no_options_are_given(self, monkeypatch) -> None:
        fake, captured = replying("Hello.")
        monkeypatch.setattr(urllib.request, "urlopen", fake)

        LLMClient().ask("be brief", "hi")

        assert captured["body"]["temperature"] == 0.8

    def test_an_api_key_becomes_a_bearer_token(self, monkeypatch) -> None:
        fake, captured = replying("Hello.")
        monkeypatch.setattr(urllib.request, "urlopen", fake)

        LLMClient(api_key="secret-token").ask("be brief", "hi")

        assert captured["headers"]["Authorization"] == "Bearer secret-token"

    def test_no_api_key_means_no_authorization_header(self, monkeypatch) -> None:
        fake, captured = replying("Hello.")
        monkeypatch.setattr(urllib.request, "urlopen", fake)

        LLMClient().ask("be brief", "hi")

        assert "Authorization" not in captured["headers"]

    def test_the_timeout_is_passed_through(self, monkeypatch) -> None:
        fake, captured = replying("Hello.")
        monkeypatch.setattr(urllib.request, "urlopen", fake)

        LLMClient(timeout=7).ask("be brief", "hi")

        assert captured["timeout"] == 7


class TestFailures:
    def test_an_unreachable_server_is_reported_clearly(self, monkeypatch) -> None:
        def explode(request, timeout=None):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", explode)

        with pytest.raises(ModelUnreachable) as caught:
            LLMClient().ask("be brief", "hi")

        assert "unreachable" in str(caught.value)

    def test_a_body_that_is_not_json_is_reported(self, monkeypatch) -> None:
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda request, timeout=None: FakeResponse("not json at all"),
        )

        with pytest.raises(ModelUnreachable):
            LLMClient().ask("be brief", "hi")


class TestRunningOutOfTime:
    def test_a_read_timeout_is_reported_as_a_timeout(self, monkeypatch) -> None:
        import socket

        def slow(request, timeout=None):
            raise TimeoutError("timed out")

        monkeypatch.setattr(socket, "timeout", TimeoutError)
        monkeypatch.setattr(urllib.request, "urlopen", slow)

        with pytest.raises(ModelTimeout) as caught:
            LLMClient(timeout=12).ask("be brief", "hi")

        assert "12 seconds" in str(caught.value)

    def test_a_connection_dropped_for_want_of_time_is_also_a_timeout(self, monkeypatch) -> None:
        def slow(request, timeout=None):
            raise urllib.error.URLError(TimeoutError("timed out"))

        monkeypatch.setattr(urllib.request, "urlopen", slow)

        with pytest.raises(ModelTimeout):
            LLMClient(timeout=12).ask("be brief", "hi")

    def test_a_connection_refused_is_not_a_timeout(self, monkeypatch) -> None:
        def refused(request, timeout=None):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", refused)

        with pytest.raises(ModelUnreachable):
            LLMClient().ask("be brief", "hi")

    def test_the_waiting_time_is_bounded_by_default(self) -> None:
        assert LLMClient().timeout == NARRATION_TIMEOUT

    def test_the_waiting_time_is_a_sensible_number_of_seconds(self) -> None:
        assert 0 < NARRATION_TIMEOUT <= 300


class TestAskingForJson:
    def test_plain_json_is_read_as_it_stands(self, monkeypatch) -> None:
        fake, _ = replying('{"ok": true}')
        monkeypatch.setattr(urllib.request, "urlopen", fake)

        assert LLMClient().ask_json("answer in json", "how are you") == {"ok": True}

    def test_json_in_a_code_fence_is_still_read(self, monkeypatch) -> None:
        fake, _ = replying('Here you go:\n```json\n{"ok": true}\n```')
        monkeypatch.setattr(urllib.request, "urlopen", fake)

        assert LLMClient().ask_json("answer in json", "how are you") == {"ok": True}

    def test_json_after_a_preamble_is_still_read(self, monkeypatch) -> None:
        fake, _ = replying('Sure! Here is the answer:\n{"ok": true}')
        monkeypatch.setattr(urllib.request, "urlopen", fake)

        assert LLMClient().ask_json("answer in json", "how are you") == {"ok": True}

    def test_a_bad_answer_is_asked_for_again_once(self, monkeypatch) -> None:
        answers = iter(["I cannot answer that.", "Still no."])
        asked: list[str] = []

        def fake_urlopen(request, timeout=None):
            asked.append(json.loads(request.data)["messages"][1]["content"])
            return FakeResponse(reply_body(next(answers)))

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        with pytest.raises(NoJSONAnswer):
            LLMClient().ask_json("answer in json", "how are you")

        assert len(asked) == 2
        assert "JSON only" in asked[1]

    def test_the_second_attempt_is_asked_stricter(self, monkeypatch) -> None:
        answers = iter(["not json", '{"ok": true}'])
        asked: list[str] = []

        def fake_urlopen(request, timeout=None):
            asked.append(json.loads(request.data)["messages"][1]["content"])
            return FakeResponse(reply_body(next(answers)))

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        assert LLMClient().ask_json("answer in json", "how are you") == {"ok": True}
        assert "JSON only" in asked[1]

    def test_a_reply_that_never_becomes_json_is_reported(self, monkeypatch) -> None:
        fake, _ = replying("I would rather not.")
        monkeypatch.setattr(urllib.request, "urlopen", fake)

        with pytest.raises(NoJSONAnswer) as caught:
            LLMClient().ask_json("answer in json", "how are you")

        assert "did not answer with JSON" in str(caught.value)

    def test_the_quoted_reply_is_shown_when_it_never_parses(self, monkeypatch) -> None:
        fake, _ = replying("I would rather not.")
        monkeypatch.setattr(urllib.request, "urlopen", fake)

        with pytest.raises(NoJSONAnswer) as caught:
            LLMClient().ask_json("answer in json", "how are you")

        assert "I would rather not." in str(caught.value)

    def test_the_prompt_comes_back_so_the_caller_can_show_it(self, monkeypatch) -> None:
        """A caller that has to ask again needs to see what it asked, not guess."""
        fake, _ = replying("I would rather not.")
        monkeypatch.setattr(urllib.request, "urlopen", fake)

        with pytest.raises(NoJSONAnswer) as caught:
            LLMClient().ask_json("answer in json", "how are you")

        assert caught.value.prompt == "how are you"

    def test_the_answer_comes_back_so_the_caller_can_show_it(self, monkeypatch) -> None:
        fake, _ = replying("I would rather not.")
        monkeypatch.setattr(urllib.request, "urlopen", fake)

        with pytest.raises(NoJSONAnswer) as caught:
            LLMClient().ask_json("answer in json", "how are you")

        assert caught.value.answer == "I would rather not."


class TestFromEnvironment:
    def test_the_model_comes_from_the_environment(self, monkeypatch) -> None:
        monkeypatch.setenv(MODEL_ENV, "gemma-4-e2b")

        assert LLMClient.from_environment().model == "gemma-4-e2b"

    def test_the_url_comes_from_the_environment(self, monkeypatch) -> None:
        monkeypatch.setenv(BASE_URL_ENV, "http://127.0.0.1:9999")

        assert LLMClient.from_environment().base_url == "http://127.0.0.1:9999"

    def test_the_api_key_comes_from_the_environment(self, monkeypatch) -> None:
        monkeypatch.setenv(API_KEY_ENV, "secret-token")

        assert LLMClient.from_environment().api_key == "secret-token"

    def test_an_explicit_url_wins_over_the_environment(self, monkeypatch) -> None:
        monkeypatch.setenv(BASE_URL_ENV, "http://127.0.0.1:9999")

        client = LLMClient.from_environment(base_url="http://127.0.0.1:8080")

        assert client.base_url == "http://127.0.0.1:8080"

    def test_an_explicit_model_wins_over_the_environment(self, monkeypatch) -> None:
        monkeypatch.setenv(MODEL_ENV, "from-env")

        assert LLMClient.from_environment(model="explicit").model == "explicit"

    def test_the_defaults_apply_when_nothing_is_set(self, monkeypatch) -> None:
        for name in (MODEL_ENV, BASE_URL_ENV, API_KEY_ENV):
            monkeypatch.delenv(name, raising=False)

        client = LLMClient.from_environment()

        assert client.model == ""
        assert client.base_url == "http://localhost:8080"
        assert client.api_key == ""

    def test_building_a_client_talks_to_nothing(self, monkeypatch) -> None:
        def explode(*args, **kwargs):
            raise AssertionError("building a client must not make a request")

        monkeypatch.setattr(urllib.request, "urlopen", explode)

        assert LLMClient.from_environment().base_url