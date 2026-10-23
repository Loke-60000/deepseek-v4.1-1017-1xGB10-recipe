from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

from recipe.bench.speed import PROMPTS, PrefillBench, SpeedBench
from recipe.pack.cold import GIB, ColdPlanner
from recipe.pack.warm import ColdWarmer, available_bytes
from recipe.probe import Probe
from recipe.server import Server


class Warmup:
    """The page cache filled before traffic needs it: the most routed cold experts at disk speed, then a short reply
    of each kind and a 4K prompt, so the experts the traffic itself uses come in too."""

    def __init__(self, server: Server, keep_free: int):
        self.server = server
        self.keep_free = keep_free

    def run(self) -> str:
        serve = self.server.profile.serve
        Probe(serve.port, serve.model_id).wait_ready()
        sets = self.server.settings.paths.cold
        cold = ColdPlanner.read_set(sets / Path(self.server.profile.env["TF_DSV41_COLD"]).name)
        hits = ColdPlanner.read_hits(sets / "hits-calibration.json")
        warmer = ColdWarmer.of_pack(self.server.pack, self.server.prepared())
        read = warmer.warm(ColdWarmer.order(cold, hits), available_bytes(self.keep_free))
        bench = SpeedBench(serve.port, serve.model_id, max_tokens=128)
        for kind in PROMPTS:
            bench.run(kind, 1)
        PrefillBench(serve.port, serve.model_id).run(4096, salt=time.time_ns() % 100_000)
        return f"warmed: {read / GIB:.1f} GiB of cold experts, a short reply of each kind and a 4K prompt"


class Keeper:
    """Runs beside the server: kills it when MemAvailable falls under a floor, because unified memory freezes the
    whole box before the OOM killer acts, and warms it each time it starts."""

    FLOOR = int(2.5 * GIB)
    KEEP_FREE = 6 * GIB

    def __init__(self, server: Server, floor: int = FLOOR, keep_free: int = KEEP_FREE, every: float = 1.0):
        self.server = server
        self.floor = floor
        self.keep_free = keep_free
        self.every = every
        self.warmed: str | None = None
        self.warming: threading.Thread | None = None

    @staticmethod
    def available() -> int:
        return available_bytes(0)

    def started(self) -> str | None:
        """The running server container's start time; None while it is down."""
        state = subprocess.run(
            ["docker", "inspect", "-f", "{{if .State.Running}}{{.State.StartedAt}}{{end}}", Server.CONTAINER],
            capture_output=True,
            text=True,
        )
        return state.stdout.strip() or None

    def step(self) -> None:
        started = self.started()
        if started is None:
            return
        available = self.available()
        if available < self.floor:
            print(
                f"MemAvailable {available >> 20} MiB under {self.floor >> 20} MiB: killing {Server.CONTAINER}",
                flush=True,
            )
            subprocess.run(["docker", "kill", Server.CONTAINER], check=False)
        elif started != self.warmed and not (self.warming and self.warming.is_alive()):
            self.warmed = started
            self.warming = threading.Thread(target=self._warm, daemon=True)
            self.warming.start()

    def watch(self) -> None:
        while True:
            self.step()
            time.sleep(self.every)

    def _warm(self) -> None:
        try:
            print(Warmup(self.server, self.keep_free).run(), flush=True)
        except Exception as error:
            print(f"warm-up failed: {error}", flush=True)
