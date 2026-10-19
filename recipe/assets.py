from __future__ import annotations

from pathlib import Path

from recipe.docker import DockerRun, Mount
from recipe.profile import Profile
from recipe.settings import Settings


class ToolboxJob:
    """A shell script run in the toolbox image, with the grafted pack mounted at /model."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def run(self, steps: list[str], mounts: list[Mount], env: dict[str, str] | None = None) -> None:
        script = " && ".join(steps)
        DockerRun(
            image=self.settings.image.toolbox_tag,
            command=["bash", "-c", script],
            mounts=[Mount(self.settings.paths.pack, "/model"), *mounts],
            env=env or {},
        ).run()


class AssetBuilder:
    """Triton AOT kernels, RoPE tables and the Engram host table for one profile."""

    def __init__(self, settings: Settings, profile: Profile):
        self.job = ToolboxJob(settings)
        self.profile = profile
        self.directory = settings.paths.assets / profile.name

    def build(self) -> Path:
        (self.directory / "aot").mkdir(parents=True, exist_ok=True)
        src = "/kit/build/python-source/src"
        steps = [
            f"python tools/zig/triton_fill.py sigs --py {src} --out /assets/sigs.json",
            "tf-dsv41-m1 aot-needs /assets/sigs.json /assets/needs.json",
            f"python tools/zig/triton_fill.py compile --py {src} --needs /assets/needs.json"
            " --options tools/zig/dsv41_triton_options.json --arch 121 --have /assets/aot --out /assets/aot",
            f"env PYTHONPATH={src} python tools/zig/dsv41_rope_tables.py --config /model/config.json"
            " --out /assets/rope.bin",
            f"python tools/zig/dsv41_engram_host.py --py-src {src} --config /model/config.json"
            " --tokenizer /model/tokenizer.json --out /assets/engram-host.bin",
        ]
        env = dict(self.profile.env) | {"TF_DSV41_ASSETS": "/assets"}
        self.job.run(steps, [Mount(self.directory, "/assets", read_only=False)], env)
        return self.directory


class EngramPacker:
    """The 1017 Engram tables from the experts pack, packed into one rank's NVMe shards."""

    SCRIPT = "/kit/scripts/pack_engram.py"

    def __init__(self, settings: Settings):
        self.job = ToolboxJob(settings)
        self.directory = settings.paths.engram
        sources = settings.sources
        self.repo_dir = settings.paths.hf_cache / f"models--{sources.experts_repo.replace('/', '--')}"
        self.source = f"/hub/snapshots/{sources.experts_revision}"

    def build(self) -> Path:
        common = f"--src {self.source} --config /model/config.json --rank 0 --world 1"
        steps = [
            f"python {self.SCRIPT} {common} --out /engram",
            f"python {self.SCRIPT} {common} --check /engram",
        ]
        mounts = [Mount(self.repo_dir, "/hub"), Mount(self.directory, "/engram", read_only=False)]
        self.job.run(steps, mounts)
        return self.directory
