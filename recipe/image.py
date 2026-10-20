from __future__ import annotations

from recipe.docker import ImageBuild
from recipe.settings import REPO_ROOT, Image


class Images:
    """The serving engine image and the toolbox image that prepares its assets."""

    def __init__(self, image: Image):
        self.image = image

    def engine(self) -> str:
        return ImageBuild(REPO_ROOT / "docker" / "Dockerfile", REPO_ROOT, self.image.tag, self._kit()).run()

    def toolbox(self) -> str:
        args = self._kit() | {"ENGINE_IMAGE": self.image.tag, "TOOLBOX_IMAGE": self.image.toolbox_tag}
        return ImageBuild(REPO_ROOT / "docker" / "toolbox.Dockerfile", REPO_ROOT, self.image.toolbox_tag, args).run()

    def _kit(self) -> dict[str, str]:
        return {"KIT_REPO": self.image.kit_repo, "KIT_COMMIT": self.image.kit_commit}
