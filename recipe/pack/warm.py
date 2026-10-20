from __future__ import annotations

import json
import os
import re
import struct
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from recipe.pack.cold import Expert

MAGIC = b"TFDSV41P"
ENTRY = struct.Struct("<BBHI4QQQ32s")
PROJECTION = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.trellis$")


class PreparedFile:
    """An engine prepared file (`TF_DSV41_PREPARED`): each tensor's payload offset by name."""

    def __init__(self, path: Path):
        self.path = path
        with open(path, "rb") as handle:
            head = handle.read(16)
            if head[:8] != MAGIC:
                raise ValueError(f"{path}: not a prepared file")
            count = struct.unpack_from("<I", head, 12)[0]
            self.offsets: dict[str, int] = {}
            for _ in range(count):
                fields = ENTRY.unpack(handle.read(ENTRY.size))
                name = handle.read(fields[2]).decode()
                self.offsets[name] = fields[8]


@dataclass(frozen=True)
class Span:
    path: Path
    start: int
    length: int


class ColdWarmer:
    """The page cache filled with cold experts before traffic needs them: the most routed first, sequential reads."""

    def __init__(self, sizes: Mapping[tuple[int, int, str], int], prepared: Path):
        self.sizes = sizes
        self.prepared = prepared
        self.files: dict[int, PreparedFile] = {}
        self.starts: dict[tuple[int, int, str], int] = {}
        for layer, expert, projection in sorted(sizes):
            previous = (layer, expert - 1, projection)
            self.starts[(layer, expert, projection)] = (
                self.starts[previous] + sizes[previous] if previous in self.starts else 0
            )

    @classmethod
    def of_pack(cls, pack: Path, prepared: Path) -> ColdWarmer:
        weight_map = json.loads((pack / "model.safetensors.index.json").read_text())["weight_map"]
        headers: dict[str, dict[str, dict[str, list[int]]]] = {}
        sizes: dict[tuple[int, int, str], int] = {}
        for name, shard in weight_map.items():
            match = PROJECTION.match(name)
            if not match:
                continue
            if shard not in headers:
                with open(pack / shard, "rb") as handle:
                    length = struct.unpack("<Q", handle.read(8))[0]
                    headers[shard] = json.loads(handle.read(length))
            start, end = headers[shard][name]["data_offsets"]
            sizes[(int(match[1]), int(match[2]), match[3])] = end - start
        return cls(sizes, prepared)

    def spans(self, experts: Iterable[Expert]) -> list[Span]:
        out = []
        for layer, expert in experts:
            prepared = self._file(layer)
            for projection in ("w1", "w2", "w3"):
                base = self._payload(prepared, layer, projection)
                key = (layer, expert, projection)
                out.append(Span(prepared.path, base + self.starts[key], self.sizes[key]))
        return out

    def warm(self, experts: Iterable[Expert], budget: int) -> int:
        total, descriptors = 0, {}
        try:
            for span in self.spans(experts):
                if total + span.length > budget:
                    break
                if span.path not in descriptors:
                    descriptors[span.path] = os.open(span.path, os.O_RDONLY)
                os.posix_fadvise(descriptors[span.path], span.start, span.length, os.POSIX_FADV_WILLNEED)
                total += span.length
        finally:
            for fd in descriptors.values():
                os.close(fd)
        return total

    @staticmethod
    def order(cold: Iterable[Expert], hits: Mapping[Expert, int]) -> list[Expert]:
        return sorted(cold, key=lambda e: -hits.get(e, 0))

    def _file(self, layer: int) -> PreparedFile:
        if layer not in self.files:
            self.files[layer] = PreparedFile(self.prepared / f"L{layer}.bin")
        return self.files[layer]

    @staticmethod
    def _payload(prepared: PreparedFile, layer: int, projection: str) -> int:
        for suffix in ("data", "trellis"):
            name = f"L{layer}.moe.{projection}.{suffix}"
            if name in prepared.offsets:
                return prepared.offsets[name]
        raise KeyError(f"{prepared.path} has no routed {projection} tensor")


def free_bytes(margin: int) -> int:
    fields: dict[str, int] = defaultdict(int)
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        fields[key] = int(value.split()[0]) * 1024
    return max(fields["MemFree"] - margin, 0)
