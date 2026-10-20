from __future__ import annotations

import subprocess
import time
from pathlib import Path

from recipe.settings import REPO_ROOT

UNITS = {
    "dsv41-guard.service": """[Unit]
Description=DeepSeek V4.1: stop the server before the Spark runs out of memory

[Service]
WorkingDirectory={root}
ExecStart={python} -m recipe.cli guard --floor-gib {floor}
Restart=always

[Install]
WantedBy=default.target
""",
    "dsv41-warm.service": """[Unit]
Description=DeepSeek V4.1: warm the page cache once the server is up
After=dsv41-guard.service

[Service]
Type=oneshot
WorkingDirectory={root}
ExecStart={python} -m recipe.cli serve warm --profile {profile}
TimeoutStartSec=3600

[Install]
WantedBy=default.target
""",
}


class MemoryGuard:
    """Kills the server container when MemAvailable falls under a floor: unified memory locks the box up first."""

    def __init__(self, container: str, floor_bytes: int, every: float = 1.0):
        self.container = container
        self.floor = floor_bytes
        self.every = every

    @staticmethod
    def available() -> int:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
        raise RuntimeError("no MemAvailable in /proc/meminfo")

    def running(self) -> bool:
        state = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", self.container], capture_output=True, text=True
        )
        return state.stdout.strip() == "true"

    def watch(self) -> None:
        while True:
            available = self.available()
            if available < self.floor and self.running():
                below = f"MemAvailable {available >> 20} MiB under {self.floor >> 20} MiB"
                print(f"{below}: killing {self.container}", flush=True)
                subprocess.run(["docker", "kill", self.container], check=False)
            time.sleep(self.every)


class Units:
    """The systemd user units that keep the deployment safe and warm across reboots."""

    def __init__(self, python: str, profile: str, floor_gib: float, directory: Path):
        self.values = {"root": REPO_ROOT, "python": python, "profile": profile, "floor": floor_gib}
        self.directory = directory

    def install(self) -> list[str]:
        self.directory.mkdir(parents=True, exist_ok=True)
        for name, template in UNITS.items():
            (self.directory / name).write_text(template.format(**self.values))
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", *UNITS], check=True)
        subprocess.run(["systemctl", "--user", "restart", "dsv41-guard.service"], check=True)
        return list(UNITS)
