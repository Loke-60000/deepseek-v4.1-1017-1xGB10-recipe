from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Reply:
    text: str
    completion_tokens: int
    seconds: float

    @property
    def tokens_per_second(self) -> float:
        return self.completion_tokens / self.seconds if self.seconds else 0.0


class Probe:
    """A client for the served OpenAI API: readiness, then answers that must be right."""

    def __init__(self, port: int, model: str, host: str = "127.0.0.1", timeout: float = 600):
        self.base = f"http://{host}:{port}/v1"
        self.model = model
        self.timeout = timeout

    def wait_ready(self, timeout: float = 1800, every: float = 5) -> list[str]:
        deadline = time.monotonic() + timeout
        while True:
            try:
                return [entry["id"] for entry in self._request("GET", "/models")["data"]]
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                if time.monotonic() > deadline:
                    raise
                time.sleep(every)

    def chat(self, content: str | list[dict[str, Any]], max_tokens: int = 256, model: str | None = None) -> Reply:
        return self.complete([{"role": "user", "content": content}], max_tokens, model=model)

    def complete(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        extra: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> Reply:
        body = {
            "model": model or self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "reasoning_effort": "low",
            **(extra or {}),
        }
        started = time.monotonic()
        reply = self._request("POST", "/chat/completions", body)
        message = reply["choices"][0]["message"]
        return Reply(
            text=message.get("content") or json.dumps(message.get("tool_calls") or []),
            completion_tokens=reply.get("usage", {}).get("completion_tokens", 0),
            seconds=time.monotonic() - started,
        )

    def smoke(self) -> Reply:
        reply = self.chat("What is 17 * 23? Answer with the number only.")
        if "391" not in reply.text:
            raise AssertionError(f"wrong answer to 17 * 23: {reply.text!r}")
        return reply

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self.base + path, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)
