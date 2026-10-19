from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Mount:
    host: Path
    container: str
    read_only: bool = True

    def flag(self) -> str:
        return f"{self.host}:{self.container}" + (":ro" if self.read_only else "")


@dataclass(frozen=True)
class DockerRun:
    image: str
    command: Sequence[str]
    mounts: Sequence[Mount] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    gpus: bool = True
    name: str | None = None
    detach: bool = False
    options: Sequence[str] = ()

    def argv(self) -> list[str]:
        argv = ["docker", "run", "--init", "--ipc=host", "--network=host", "--ulimit", "memlock=-1"]
        argv += ["--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp"]
        argv += ["-d"] if self.detach else ["--rm"]
        if self.name:
            argv += ["--name", self.name]
        if self.gpus:
            argv += ["--gpus", "all"]
        for mount in self.mounts:
            argv += ["-v", mount.flag()]
        for key, value in sorted(self.env.items()):
            argv += ["-e", f"{key}={value}"]
        return [*argv, *self.options, self.image, *self.command]

    def run(self) -> None:
        for mount in self.mounts:
            if not mount.read_only:
                mount.host.mkdir(parents=True, exist_ok=True)
            elif not mount.host.exists():
                raise FileNotFoundError(f"read-only mount {mount.host} does not exist")
        subprocess.run(self.argv(), check=True)


class ImageBuild:
    def __init__(self, dockerfile: Path, context: Path, tag: str, args: Mapping[str, str]):
        self.dockerfile = dockerfile
        self.context = context
        self.tag = tag
        self.args = args

    def run(self) -> str:
        argv = ["docker", "build", "-f", str(self.dockerfile), "-t", self.tag]
        for key, value in self.args.items():
            argv += ["--build-arg", f"{key}={value}"]
        subprocess.run([*argv, str(self.context)], check=True)
        return self.tag
