from __future__ import annotations

import random
import threading
from dataclasses import dataclass

from recipe.bench.speed import PrefillBench
from recipe.pack.warm import available_bytes
from recipe.probe import Probe


@dataclass(frozen=True)
class Retrieval:
    tokens: int
    depth: float
    found: bool
    seconds: float
    lowest_available_gib: float

    def line(self) -> str:
        verdict = "found" if self.found else "MISSED"
        return (
            f"needle  {self.tokens:>9,} tokens  depth {self.depth:.0%}  {verdict:6s}  {self.seconds:7.1f} s"
            f"  lowest MemAvailable {self.lowest_available_gib:.2f} GiB"
        )


class NeedleBench:
    """A passphrase hidden in a long prompt of source code, then asked for: does the model read that far, and how low
    does free memory go while it prefills? Run it on the server's host so the memory reading is the server's."""

    QUESTION = "\n\nWhat is the passphrase for the vault, stated somewhere above? Reply with the passphrase only."

    def __init__(self, probe: Probe):
        self.probe = probe

    def run(self, tokens: int, depth: float, salt: int) -> Retrieval:
        rng = random.Random(salt)
        secret = f"{rng.choice(['amber', 'cobalt', 'saffron', 'teal', 'umber'])}-{rng.randrange(10_000, 99_999)}"
        text = PrefillBench._text(int(tokens * PrefillBench.CHARS_PER_TOKEN), salt)
        at = text.rfind("\n", 0, int(len(text) * depth)) + 1
        haystack = f"{text[:at]}# The passphrase for the vault is {secret}.\n{text[at:]}"
        lowest, done = [available_bytes(0)], threading.Event()

        def watch() -> None:
            while not done.wait(1.0):
                lowest[0] = min(lowest[0], available_bytes(0))

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            reply = self.probe.complete(
                [{"role": "user", "content": f"Here is a codebase dump.\n\n{haystack}{self.QUESTION}"}], 64
            )
        finally:
            done.set()
            watcher.join()
        return Retrieval(reply.prompt_tokens, depth, secret in reply.text, reply.seconds, lowest[0] / 2**30)
