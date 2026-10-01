"""One place that talks to a language model, shared by everything that needs one.

The model is reached over the OpenAI chat completions API, which llama.cpp,
Ollama, vLLM, LM Studio and the rest all implement. Nothing is downloaded at
import time, so importing this module never needs a model to be installed.
"""

from __future__ import annotations

import json
import os
import re
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field

FENCED_JSON = re.compile(r"```[a-zA-Z]*\s*(.*?)```", re.DOTALL)

NARRATION_TIMEOUT = 120.0

DEFAULT_BASE_URL = "http://localhost:8080"
MODEL_ENV = "WORLD_NARRATOR_MODEL"
BASE_URL_ENV = "WORLD_NARRATOR_BASE_URL"
API_KEY_ENV = "WORLD_NARRATOR_API_KEY"

class ModelUnreachable(RuntimeError):
    """The model server could not be reached or refused the request."""


class ModelTimeout(RuntimeError):
    """The model did not answer within the time it was given."""


class NoJSONAnswer(RuntimeError):
    """The model did not answer with the JSON that was asked for.

    Attributes:
        prompt: The user half of the exchange that failed, so it can be shown back.
        answer: What came back instead.
    """

    def __init__(self, prompt: str, answer: str) -> None:
        super().__init__(f"the model did not answer with JSON: {answer[:200]!r}")
        self.prompt = prompt
        self.answer = answer


@dataclass
class LLMClient:
    """A chat model reached over HTTP, with one method that does the talking.

    Attributes:
        model: Name of the model to ask for, ignored by servers serving one model.
        base_url: Root URL of the server, without the version path.
        api_key: Bearer token, empty when the server needs no authentication.
        options: Extra generation options forwarded verbatim.
        thinking: Whether a reasoning model may think before answering.
        max_tokens: Upper bound on the reply length.
        timeout: Seconds to wait for the server before giving up on it.
    """

    model: str = ""
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    options: dict[str, object] = field(default_factory=dict)
    thinking: bool = False
    max_tokens: int = 400
    timeout: float = NARRATION_TIMEOUT

    @classmethod
    def from_environment(
        cls,
        base_url: str = "",
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
        thinking: bool | None = None,
        timeout: float | None = None,
    ) -> LLMClient:
        """Build a client from the environment, letting callers override any field."""
        settings: dict[str, object] = {
            "model": os.environ.get(MODEL_ENV, "") if model is None else model,
            "base_url": base_url or os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL),
            "api_key": os.environ.get(API_KEY_ENV, "") if api_key is None else api_key,
        }
        if max_tokens is not None:
            settings["max_tokens"] = max_tokens
        if thinking is not None:
            settings["thinking"] = thinking
        if timeout is not None:
            settings["timeout"] = timeout
        return cls(**settings)

    def ask(self, system: str, user: str) -> str:
        """Send one exchange and return whatever the model wrote back.

        A reasoning model may answer in ``reasoning_content`` and leave ``content``
        empty, so the two are stitched together when the answer is missing.
        """
        payload = {
            "model": self.model or "local",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.thinking},
            **({"temperature": 0.8} if not self.options else self.options),
        }
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read())
        except TimeoutError as error:
            raise ModelTimeout(
                f"the model did not answer within {self.timeout:g} seconds"
            ) from error
        except (urllib.error.URLError, OSError) as error:
            if isinstance(getattr(error, "reason", None), socket.timeout):
                raise ModelTimeout(
                    f"the model did not answer within {self.timeout:g} seconds"
                ) from error
            raise ModelUnreachable(f"model unreachable: {error}") from error
        except json.JSONDecodeError as error:
            raise ModelUnreachable(f"model sent something that is not JSON: {error}") from error
        return _reply(body)

    def ask_json(self, system: str, user: str) -> object:
        """Ask for a reply and read it as JSON, asking again once if it is not.

        One retry, because a fenced or chatty answer is common enough to be worth
        it. A second failure raises with the prompt and the answer attached.
        """
        answer = self.ask(system, user).strip()
        try:
            return json.loads(_stripped(answer))
        except json.JSONDecodeError:
            pass
        answer = self.ask(system, f"{user}\n\nAnswer with JSON only.").strip()
        try:
            return json.loads(_stripped(answer))
        except json.JSONDecodeError:
            raise NoJSONAnswer(user, answer) from None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


def _stripped(text: str) -> str:
    """Take the JSON out of a reply, code fence and preamble included.

    A model may wrap its answer in a fence, say a few words first, or both, so the
    fenced block is preferred wherever it appears and the rest is only searched
    when there is no fence at all.
    """
    body = text.strip()
    fenced = FENCED_JSON.search(body)
    if fenced is not None:
        return fenced.group(1).strip()
    starts = [index for index in (body.find("{"), body.find("[")) if index >= 0]
    if starts:
        return body[min(starts) :].strip()
    return body


def _reply(body: object) -> str:
    """Pull the answer out of an OpenAI chat completion response."""
    if not isinstance(body, dict):
        return ""
    choices = body.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    return content or message.get("reasoning_content") or ""


__all__ = [
    "API_KEY_ENV",
    "BASE_URL_ENV",
    "DEFAULT_BASE_URL",
    "MODEL_ENV",
    "LLMClient",
    "ModelTimeout",
    "ModelUnreachable",
    "NoJSONAnswer",
]