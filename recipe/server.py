from __future__ import annotations

import shlex
import struct
import subprocess
from pathlib import Path

from recipe.docker import DockerRun, Mount
from recipe.profile import Profile
from recipe.settings import Settings

ROPE_MARGIN = 2048
REQUIRED_ASSETS = ("rope.bin", "engram-host.bin", "aot/aot.json", "sigs.json")


class Server:
    """The single-rank tensorfold-dsv41 container for one profile."""

    CONTAINER = "dsv41"

    def __init__(self, settings: Settings, profile: Profile):
        self.settings = settings
        self.profile = profile
        self.assets = settings.paths.assets / profile.name

    def start(self) -> None:
        self.preflight()
        self.check_aot()
        self._container(detach=True).run()

    def stop(self) -> None:
        drain = int(self.profile.env.get("TF_DSV41_DRAIN_S", "20"))
        subprocess.run(["docker", "stop", "-t", str(drain + 60), self.CONTAINER], check=False)
        subprocess.run(["docker", "rm", self.CONTAINER], check=False, capture_output=True)

    def dry_run(self) -> str:
        return shlex.join(self._container(detach=True).argv())

    def preflight(self) -> None:
        paths = self.settings.paths
        for directory in (paths.pack, paths.engram, self.assets):
            if not directory.is_dir():
                raise FileNotFoundError(f"missing directory {directory}")
        missing = [name for name in REQUIRED_ASSETS if not (self.assets / name).exists()]
        if missing:
            hint = f"dsv41 assets --profile {self.profile.name}"
            raise FileNotFoundError(f"{self.assets} lacks {', '.join(missing)}; run `{hint}`")
        rows = self._rope_rows(self.assets / "rope.bin")
        needed = self.profile.context + ROPE_MARGIN
        if rows < needed:
            raise ValueError(f"rope.bin holds {rows} rows, context {self.profile.context} needs {needed}")

    def check_aot(self) -> None:
        DockerRun(
            image=self.settings.image.tag,
            command=["tf-dsv41-m1", "aot-needs", "/assets/sigs.json", "/state/aot-needs.json", "/assets/aot"],
            mounts=[Mount(self.assets, "/assets"), Mount(self.settings.paths.state, "/state", read_only=False)],
            env=dict(self.profile.env),
            gpus=False,
        ).run()

    def command(self) -> list[str]:
        serve = self.profile.serve
        argv = [
            "tensorfold-dsv41",
            "serve",
            "/model",
            "--host",
            "0.0.0.0",
            "--port",
            str(serve.port),
            "--parallel",
            str(self.profile.slots),
            "--context",
            str(self.profile.context),
            "--name",
            serve.model_id,
            "--max-tokens",
            str(serve.max_tokens),
        ]
        for alias in serve.aliases:
            argv += ["--alias", alias]
        if self.profile.env.get("TF_DSV41_THINKING", "1") == "0":
            argv.append("--no-thinking")
        return argv

    def _container(self, detach: bool) -> DockerRun:
        paths = self.settings.paths
        return DockerRun(
            image=self.settings.image.tag,
            command=self.command(),
            mounts=[
                Mount(paths.pack, "/model"),
                Mount(paths.engram, "/engram"),
                Mount(self.assets, "/assets"),
                Mount(paths.prepared, "/prepared", read_only=False),
                Mount(paths.state, "/state", read_only=False),
                Mount(paths.state / "sessions", "/sessions", read_only=False),
                Mount(paths.cache, "/cache", read_only=False),
            ],
            env=dict(self.profile.env),
            name=self.CONTAINER,
            detach=detach,
            options=["--cap-add", "IPC_LOCK"],
        )

    @staticmethod
    def _rope_rows(path: Path) -> int:
        with open(path, "rb") as handle:
            head = handle.read(12)
        rows: int = struct.unpack_from("<I", head, 8)[0]
        return rows
