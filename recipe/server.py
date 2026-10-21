from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import struct
import subprocess
from pathlib import Path

from recipe.ablate.job import DIRECTIONS
from recipe.docker import DockerRun, Mount
from recipe.profile import ENGINE_PREFIXES, Profile
from recipe.settings import REPO_ROOT, Environment, Settings

ROPE_MARGIN = 2048
REQUIRED_ASSETS = ("rope.bin", "engram-host.bin", "aot/aot.json", "sigs.json")
SHIPPED_COLD = REPO_ROOT / "config" / "cold"
DOCKER_SOCKET = Path("/var/run/docker.sock")


class Server:
    """The single-rank tensorfold-dsv41 container for one profile."""

    CONTAINER = "dsv41"
    KEEPER = "dsv41-keeper"

    def __init__(self, settings: Settings, profile: Profile):
        self.settings = settings
        self.profile = profile
        self.assets = settings.paths.assets / profile.assets

    def start(self) -> None:
        self.preflight()
        self.check_aot()
        self.stage_cold()
        self._container(detach=True).run()
        subprocess.run(["docker", "rm", "-f", self.KEEPER], check=False, capture_output=True)
        self._keeper().run()

    def stop(self) -> None:
        drain = int(self.profile.env.get("TF_DSV41_DRAIN_S", "20"))
        subprocess.run(["docker", "stop", "-t", str(drain + 60), self.CONTAINER], check=False)
        subprocess.run(["docker", "rm", self.CONTAINER], check=False, capture_output=True)
        subprocess.run(["docker", "rm", "-f", self.KEEPER], check=False, capture_output=True)

    def stage_cold(self) -> None:
        """Copies the shipped cold expert sets next to the model, where the server and the keeper read them; sets
        of your own in that folder are left alone."""
        target = self.settings.paths.cold
        target.mkdir(parents=True, exist_ok=True)
        for shipped in SHIPPED_COLD.glob("*.json"):
            copy = target / shipped.name
            if not copy.exists() or copy.read_bytes() != shipped.read_bytes():
                shutil.copyfile(shipped, copy)

    def prepared(self) -> Path:
        """The prepared folder the running server loaded (its boot log names it)."""
        log = subprocess.run(["docker", "logs", self.CONTAINER], capture_output=True, text=True, check=True)
        match = re.search(r"prepared /prepared/([0-9a-f]+)/rank0", log.stdout + log.stderr)
        if not match:
            raise RuntimeError(f"{self.CONTAINER} has not loaded a prepared folder yet")
        return self.settings.paths.prepared / match[1] / "rank0"

    def dry_run(self) -> str:
        return shlex.join(self._container(detach=True).argv())

    def preflight(self) -> None:
        paths = self.settings.paths
        for directory in (paths.pack, paths.engram, self.assets):
            if not directory.is_dir():
                raise FileNotFoundError(f"missing directory {directory}")
        if self.ablating and not (paths.ablate / DIRECTIONS).is_file():
            raise FileNotFoundError(f"{paths.ablate / DIRECTIONS} is missing: run `dsv41 ablate`")
        missing = [name for name in REQUIRED_ASSETS if not (self.assets / name).exists()]
        if missing:
            hint = f"dsv41 assets --profile {self.profile.name}"
            raise FileNotFoundError(f"{self.assets} lacks {', '.join(missing)}; run `{hint}`")
        rows = self._rope_rows(self.assets / "rope.bin")
        needed = self.profile.context + ROPE_MARGIN
        if rows < needed:
            raise ValueError(f"rope.bin holds {rows} rows, context {self.profile.context} needs {needed}")

    def check_aot(self) -> None:
        """aot-needs lists every kernel variant the profile can launch and fails if the assets lack one (5-10 minutes).
        Its answer depends only on the engine image, the kernel set and the profile's env, so a pass is remembered under
        their hash and a start with the same three skips it."""
        passed = self.settings.paths.state / "aot-passed" / self._aot_key()
        if passed.exists():
            return
        DockerRun(
            image=self.settings.image.tag,
            command=["tf-dsv41-m1", "aot-needs", "/assets/sigs.json", "/state/aot-needs.json", "/assets/aot"],
            mounts=[Mount(self.assets, "/assets"), Mount(self.settings.paths.state, "/state", read_only=False)],
            env=dict(self.profile.env),
            gpus=False,
        ).run()
        passed.parent.mkdir(parents=True, exist_ok=True)
        passed.touch()

    def _aot_key(self) -> str:
        image = subprocess.run(
            ["docker", "image", "inspect", "-f", "{{.Id}}", self.settings.image.tag], capture_output=True, text=True
        ).stdout.strip()
        digest = hashlib.sha256(image.encode())
        digest.update(json.dumps(sorted(self.profile.env.items())).encode())
        for name in ("sigs.json", "aot/aot.json"):
            digest.update((self.assets / name).read_bytes())
        return digest.hexdigest()[:16]

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
        if self.profile.env.get("TF_DSV41_DRAFTS", "1") == "0":
            argv.append("--no-drafts")
        return argv

    @property
    def ablating(self) -> bool:
        return bool(self.profile.serve.ablate_model)

    def environment(self) -> dict[str, str]:
        env = dict(self.profile.env)
        if self.ablating:
            env["TF_DSV41_ABLATE"] = f"/ablate/{DIRECTIONS}"
            env["TENSORFOLD_ABLATE_MODEL"] = self.profile.serve.ablate_model
        return env

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
                Mount(paths.cold, "/cold"),
                *([Mount(paths.ablate, "/ablate")] if self.ablating else []),
            ],
            env=self.environment(),
            name=self.CONTAINER,
            detach=detach,
            options=["--cap-add", "IPC_LOCK", "--restart", "unless-stopped"],
        )

    def _keeper(self) -> DockerRun:
        """`dsv41 keep` in the toolbox image, restarted by Docker like the server. The model folder and the Hugging
        Face cache keep their host paths, so the settings mean the same inside it."""
        paths, image = self.settings.paths, self.settings.image
        passed = ("DSV41_SERVE_", *ENGINE_PREFIXES)
        env = {key: value for key, value in Environment.load().items() if key.startswith(passed)}
        env |= {
            "DSV41_PATHS_ROOT": str(paths.root),
            "DSV41_PATHS_HF_CACHE": str(paths.hf_cache),
            "DSV41_IMAGE_TAG": image.tag,
            "DSV41_IMAGE_TOOLBOX_TAG": image.toolbox_tag,
        }
        return DockerRun(
            image=image.toolbox_tag,
            command=["dsv41", "keep", "--profile", self.profile.name],
            mounts=[
                Mount(DOCKER_SOCKET, str(DOCKER_SOCKET), read_only=False),
                Mount(paths.root, str(paths.root), read_only=False),
                Mount(paths.hf_cache, str(paths.hf_cache)),
            ],
            env=env,
            gpus=False,
            name=self.KEEPER,
            detach=True,
            options=["--restart", "unless-stopped", "--group-add", str(os.stat(DOCKER_SOCKET).st_gid)],
        )

    @staticmethod
    def _rope_rows(path: Path) -> int:
        with open(path, "rb") as handle:
            head = handle.read(12)
        rows: int = struct.unpack_from("<I", head, 8)[0]
        return rows
