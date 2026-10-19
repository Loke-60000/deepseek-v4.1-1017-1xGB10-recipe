from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from recipe.bench.workload import Request, Workload
from recipe.probe import Probe


@dataclass(frozen=True)
class Answer:
    kind: str
    text: str


@dataclass(frozen=True)
class Agreement:
    kind: str
    shared_prefix: float


class Golden:
    """Greedy answers to a fixed request set: recorded under one profile, compared under another."""

    def __init__(self, probe: Probe, per_kind: int = 2, max_tokens: int = 64):
        self.probe = probe
        self.requests = [
            Request(r.kind, r.messages, max_tokens, r.extra) for r in Workload(seed=4242).requests(per_kind)
        ]

    def record(self) -> list[Answer]:
        return [Answer(r.kind, self.probe.complete(r.messages, r.max_tokens, r.extra).text) for r in self.requests]

    @staticmethod
    def save(answers: list[Answer], path: Path) -> None:
        path.write_text(json.dumps([asdict(a) for a in answers], ensure_ascii=False, indent=1) + "\n")

    @staticmethod
    def load(path: Path) -> list[Answer]:
        return [Answer(**a) for a in json.loads(path.read_text())]

    @staticmethod
    def compare(reference: list[Answer], candidate: list[Answer]) -> list[Agreement]:
        def shared(a: str, b: str) -> float:
            if a == b:
                return 1.0
            n = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))
            return n / max(len(a), len(b), 1)

        return [Agreement(r.kind, shared(r.text, c.text)) for r, c in zip(reference, candidate, strict=True)]
