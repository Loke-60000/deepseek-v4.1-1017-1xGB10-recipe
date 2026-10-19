from __future__ import annotations

import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

PROMPTS = {
    "prose": "Write a long, detailed essay about the history of navigation at sea, from the stars to satellites.",
    "code": "Write a complete Python module implementing a thread-safe LRU cache with TTL expiry, with docstrings "
    "and a unittest suite covering eviction, expiry and concurrent access.",
    "structured": "Return a JSON array of 40 objects describing fictional employees, each with id, name, team, "
    "start_date, salary and a skills array. Output only the JSON.",
}


@dataclass(frozen=True)
class Stream:
    ttft: float
    decode_tps: float
    tokens: int
    seconds: float


@dataclass(frozen=True)
class Result:
    kind: str
    concurrency: int
    streams: list[Stream]
    wall: float

    @property
    def per_stream(self) -> float:
        return statistics.median(s.decode_tps for s in self.streams)

    @property
    def aggregate(self) -> float:
        return sum(s.tokens for s in self.streams) / self.wall

    @property
    def ttft(self) -> float:
        return statistics.median(s.ttft for s in self.streams)

    def line(self) -> str:
        return (
            f"{self.kind:10s} c={self.concurrency:<2d} per-stream {self.per_stream:6.1f} tok/s  "
            f"aggregate {self.aggregate:6.1f} tok/s  ttft {self.ttft:5.2f} s"
        )


class SpeedBench:
    """Streams completions and times them: first token, then decode rate per stream and in aggregate."""

    def __init__(self, port: int, model: str, max_tokens: int = 512, host: str = "127.0.0.1"):
        self.url = f"http://{host}:{port}/v1/chat/completions"
        self.model = model
        self.max_tokens = max_tokens

    def run(self, kind: str, concurrency: int) -> Result:
        started = time.monotonic()
        with ThreadPoolExecutor(concurrency) as pool:
            streams = list(pool.map(lambda _: self._stream(PROMPTS[kind]), range(concurrency)))
        return Result(kind, concurrency, streams, time.monotonic() - started)

    def _stream(self, prompt: str) -> Stream:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(
            self.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        started = time.monotonic()
        first: float | None = None
        tokens = 0
        with urllib.request.urlopen(request, timeout=1800) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                if event.get("usage"):
                    tokens = event["usage"].get("completion_tokens", tokens)
                delta = (event.get("choices") or [{}])[0].get("delta", {})
                if first is None and (delta.get("content") or delta.get("reasoning_content")):
                    first = time.monotonic()
        ended = time.monotonic()
        ttft = (first or ended) - started
        return Stream(ttft, max(tokens - 1, 0) / max(ended - (first or started), 1e-9), tokens, ended - started)
