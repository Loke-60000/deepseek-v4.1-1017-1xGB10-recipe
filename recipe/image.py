from __future__ import annotations

from pathlib import Path

from recipe.docker import ImageBuild
from recipe.settings import Image

REPO_ROOT = Path(__file__).resolve().parent.parent


class Images:
    """The serving engine image and the toolbox image that prepares its assets."""

    def __init__(self, image: Image):
        self.image = image

    def engine(self) -> str:
        return ImageBuild(REPO_ROOT / "docker" / "Dockerfile", REPO_ROOT, self.image.tag, self._kit()).run()

    def toolbox(self) -> str:
        args = self._kit() | {"ENGINE_IMAGE": self.image.tag}
        return ImageBuild(REPO_ROOT / "docker" / "toolbox.Dockerfile", REPO_ROOT, self.image.toolbox_tag, args).run()

    def _kit(self) -> dict[str, str]:
        return {"KIT_REPO": self.image.kit_repo, "KIT_COMMIT": self.image.kit_commit}
