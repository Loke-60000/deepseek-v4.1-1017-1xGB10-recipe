from __future__ import annotations

import json
import struct
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, hf_hub_url


@dataclass(frozen=True)
class TensorRecord:
    name: str
    file: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


class RemoteFile:
    """One file of a public Hub repo, read with HTTP range requests."""

    RETRIES = 5

    def __init__(self, repo: HubRepo, filename: str):
        self.repo = repo
        self.filename = filename
        self.url = hf_hub_url(repo.repo_id, filename, revision=repo.revision)

    def read(self, start: int, end: int) -> bytes:
        headers = {"Range": f"bytes={start}-{end - 1}"}
        for attempt in range(self.RETRIES):
            try:
                with urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=120) as reply:
                    data: bytes = reply.read()
                if len(data) != end - start:
                    raise OSError(f"{self.filename}: wanted {end - start} bytes at {start}, got {len(data)}")
                return data
            except (OSError, TimeoutError):
                if attempt == self.RETRIES - 1:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    @cached_property
    def tensors(self) -> dict[str, TensorRecord]:
        size = struct.unpack("<Q", self.read(0, 8))[0]
        header = json.loads(self.read(8, 8 + size))
        header.pop("__metadata__", None)
        base = 8 + size
        return {
            name: TensorRecord(
                name,
                self.filename,
                meta["dtype"],
                tuple(meta["shape"]),
                base + meta["data_offsets"][0],
                base + meta["data_offsets"][1],
            )
            for name, meta in header.items()
        }


class HubRepo:
    def __init__(self, repo_id: str, revision: str, cache_dir: Path):
        self.repo_id = repo_id
        self.revision = revision
        self.cache_dir = cache_dir

    @cached_property
    def files(self) -> list[str]:
        return HfApi().list_repo_files(self.repo_id, revision=self.revision)

    def read_json(self, filename: str) -> Any:
        with urllib.request.urlopen(hf_hub_url(self.repo_id, filename, revision=self.revision), timeout=120) as reply:
            return json.load(reply)

    def remote(self, filename: str) -> RemoteFile:
        return RemoteFile(self, filename)

    def download(self, filename: str) -> Path:
        return Path(hf_hub_download(self.repo_id, filename, revision=self.revision, cache_dir=self.cache_dir))

    @property
    def snapshot(self) -> Path:
        path = self.cache_dir / f"models--{self.repo_id.replace('/', '--')}" / "snapshots" / self.revision
        if not path.is_dir():
            raise FileNotFoundError(f"{self.repo_id}@{self.revision} is not downloaded to {self.cache_dir}")
        return path
