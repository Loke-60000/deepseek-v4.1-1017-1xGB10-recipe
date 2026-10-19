from __future__ import annotations

from pathlib import Path

from recipe.assets import ToolboxJob
from recipe.docker import Mount
from recipe.settings import REPO_ROOT, Settings

DIRECTIONS = "derisked.safetensors"


class DirectionJob:
    """Extracts the derisked pack's refusal directions in the toolbox image, against the local experts pack."""

    def __init__(self, settings: Settings, rank: int = 1):
        self.settings = settings
        self.rank = rank
        sources = settings.sources
        self.hub = settings.paths.hf_cache / f"models--{sources.experts_repo.replace('/', '--')}"
        self.base = f"/hub/snapshots/{sources.experts_revision}"

    def build(self) -> Path:
        sources = self.settings.sources
        step = (
            f"python -m recipe.ablate.extract --base {self.base} --edited {sources.derisked_repo}"
            f" --revision {sources.derisked_revision} --rank {self.rank} --out /ablate/{DIRECTIONS}"
        )
        mounts = [
            Mount(self.hub, "/hub"),
            Mount(REPO_ROOT / "recipe", "/recipe/recipe"),
            Mount(self.settings.paths.ablate, "/ablate", read_only=False),
        ]
        ToolboxJob(self.settings).run([step], mounts, {"PYTHONPATH": "/recipe"})
        return self.settings.paths.ablate / DIRECTIONS
