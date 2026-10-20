from __future__ import annotations

import json
import os
import re
import struct
from collections import defaultdict
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
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

    CHUNK = 8 << 20

    def warm(self, experts: Iterable[Expert], budget: int, workers: int = 8) -> int:
        """Reads the spans (fadvise's WILLNEED stops at the read-ahead window): the page cache keeps what was read."""
        chosen, total = [], 0
        for span in self.spans(experts):
            if total + span.length > budget:
                break
            chosen.append(span)
            total += span.length
        descriptors = {path: os.open(path, os.O_RDONLY) for path in {s.path for s in chosen}}
        try:
            with ThreadPoolExecutor(workers) as pool:
                list(pool.map(lambda s: self._read(descriptors[s.path], s), chosen))
        finally:
            for fd in descriptors.values():
                os.close(fd)
        return total

    def _read(self, fd: int, span: Span) -> None:
        buffer = bytearray(min(self.CHUNK, span.length))
        view = memoryview(buffer)
        at, end = span.start, span.start + span.length
        while at < end:
            n = os.preadv(fd, [view[: min(self.CHUNK, end - at)]], at)
            if n <= 0:
                break
            at += n

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


def available_bytes(margin: int) -> int:
    """MemAvailable less a margin: readahead reclaims colder page cache, so the budget is what can be made free."""
    fields: dict[str, int] = defaultdict(int)
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        fields[key] = int(value.split()[0]) * 1024
    return max(fields["MemAvailable"] - margin, 0)
