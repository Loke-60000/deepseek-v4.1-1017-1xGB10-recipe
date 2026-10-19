from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from recipe.bench.workload import Request
from recipe.probe import Probe, Reply


@dataclass(frozen=True)
class Outcome:
    request: Request
    reply: Reply


class WorkloadRunner:
    """Sends a workload one request at a time and summarises tokens and seconds per kind."""

    def __init__(self, probe: Probe):
        self.probe = probe

    def run(self, requests: list[Request]) -> list[Outcome]:
        outcomes = []
        for i, request in enumerate(requests, 1):
            reply = self.probe.complete(request.messages, request.max_tokens, request.extra)
            outcomes.append(Outcome(request, reply))
            progress = f"[{i}/{len(requests)}] {request.kind}"
            print(f"{progress}: {reply.completion_tokens} tokens in {reply.seconds:.1f} s", flush=True)
        return outcomes

    @staticmethod
    def summary(outcomes: list[Outcome]) -> str:
        tokens: dict[str, int] = defaultdict(int)
        seconds: dict[str, float] = defaultdict(float)
        for outcome in outcomes:
            tokens[outcome.request.kind] += outcome.reply.completion_tokens
            seconds[outcome.request.kind] += outcome.reply.seconds
        rows = [f"{kind:13s} {tokens[kind]:7d} tokens {seconds[kind]:8.1f} s" for kind in sorted(tokens)]
        return "\n".join(rows)
