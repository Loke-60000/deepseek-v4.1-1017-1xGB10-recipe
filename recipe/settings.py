from __future__ import annotations

import json
import os
import typing
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Self

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "recipe.json"
DOTENV = REPO_ROOT / ".env"


class Environment:
    """Process environment layered over the repo's untracked .env file."""

    @staticmethod
    def load(dotenv: Path = DOTENV, environ: Mapping[str, str] = os.environ) -> dict[str, str]:
        values: dict[str, str] = {}
        if dotenv.is_file():
            for line in dotenv.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    values[key.strip()] = value.strip().strip("'\"")
        return values | dict(environ)


@dataclass(frozen=True)
class Section:
    """A frozen config section read from JSON, where DSV41_<SECTION>_<FIELD> env vars win."""

    name: ClassVar[str]

    @classmethod
    def load(cls, data: Mapping[str, Any], env: Mapping[str, str]) -> Self:
        hints = typing.get_type_hints(cls)
        values = {}
        for field in fields(cls):
            raw = env.get(f"DSV41_{cls.name.upper()}_{field.name.upper()}")
            if raw is not None:
                values[field.name] = cls._coerce(hints[field.name], raw)
            elif field.name in data:
                values[field.name] = cls._coerce(hints[field.name], data[field.name])
            elif field.default is MISSING:
                raise KeyError(f"{cls.name}.{field.name} is not set")
        return cls(**values)

    @staticmethod
    def _coerce(kind: Any, value: Any) -> Any:
        if kind is Path:
            return Path(value).expanduser()
        if kind is bool and isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        if typing.get_origin(kind) is tuple:
            items = value.split(",") if isinstance(value, str) else value
            return tuple(item.strip() for item in items if item.strip())
        return kind(value)


@dataclass(frozen=True)
class Sources(Section):
    name = "sources"
    experts_repo: str
    experts_revision: str
    dense_repo: str
    dense_revision: str
    derisked_repo: str
    derisked_revision: str


@dataclass(frozen=True)
class Paths(Section):
    name = "paths"
    hf_cache: Path
    root: Path

    @property
    def pack(self) -> Path:
        return self.root / "pack"

    @property
    def engram(self) -> Path:
        return self.root / "engram"

    @property
    def assets(self) -> Path:
        return self.root / "assets"

    @property
    def prepared(self) -> Path:
        return self.root / "prepared"

    @property
    def state(self) -> Path:
        return self.root / "state"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def ablate(self) -> Path:
        return self.root / "ablate"


@dataclass(frozen=True)
class Image(Section):
    name = "image"
    tag: str
    toolbox_tag: str
    kit_repo: str
    kit_commit: str


@dataclass(frozen=True)
class Settings:
    sources: Sources
    paths: Paths
    image: Image

    @classmethod
    def load(cls, path: Path | None = None, env: Mapping[str, str] | None = None) -> Settings:
        env = Environment.load() if env is None else env
        path = Path(env.get("DSV41_CONFIG", path or DEFAULT_CONFIG))
        data = json.loads(path.read_text())
        return cls(
            sources=Sources.load(data.get(Sources.name, {}), env),
            paths=Paths.load(data.get(Paths.name, {}), env),
            image=Image.load(data.get(Image.name, {}), env),
        )
