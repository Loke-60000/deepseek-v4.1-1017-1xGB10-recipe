from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recipe.settings import Environment, Section

PROFILES = Path(__file__).resolve().parent.parent / "config" / "profiles"
ENGINE_PREFIXES = ("TF_DSV41_", "TF_TP_", "TF_COMM_", "NCCL_")


@dataclass(frozen=True)
class ServeOptions(Section):
    name = "serve"
    model_id: str
    aliases: tuple[str, ...]
    port: int
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    ablate_model: str = ""


@dataclass(frozen=True)
class Profile:
    """A named serving profile: engine env and serve options, layered through `extends`. `pack` names the pack folder
    under the paths root (a variant beside the deployed `pack`, see tools/requant)."""

    name: str
    serve: ServeOptions
    env: Mapping[str, str]
    assets: str
    pack: str = "pack"

    @classmethod
    def load(cls, name: str, directory: Path | None = None, environ: Mapping[str, str] | None = None) -> Profile:
        """`directory` defaults to DSV41_PROFILES (the keeper reads the copy its server made), then config/profiles."""
        environ = Environment.load() if environ is None else environ
        directory = directory or Path(environ.get("DSV41_PROFILES", PROFILES))
        data = cls._resolve(name, directory, ())
        env = {key: value for key, value in data["env"].items() if value is not None}
        env |= {key: value for key, value in environ.items() if key.startswith(ENGINE_PREFIXES)}
        serve = ServeOptions.load(data["serve"], environ)
        return cls(name, serve, env, data.get("assets") or name, data.get("pack") or "pack")

    @classmethod
    def _resolve(cls, name: str, directory: Path, seen: tuple[str, ...]) -> dict[str, Any]:
        if name in seen:
            raise ValueError(f"profile cycle: {' -> '.join((*seen, name))}")
        data: dict[str, Any] = json.loads((directory / f"{name}.json").read_text())
        parent = data.pop("extends", None)
        if parent is None:
            keys = ("assets", "pack")
            return {"serve": data.get("serve", {}), "env": data.get("env", {})} | {k: data.get(k) for k in keys}
        merged = cls._resolve(parent, directory, (*seen, name))
        merged["assets"] = data.get("assets")
        merged["pack"] = data.get("pack") or merged.get("pack")
        merged["serve"] = merged["serve"] | data.get("serve", {})
        merged["env"] = merged["env"] | data.get("env", {})
        return merged

    @property
    def context(self) -> int:
        return int(self.env["TF_DSV41_CONTEXT"])

    @property
    def slots(self) -> int:
        return int(self.env["TF_DSV41_SLOTS"])
