from __future__ import annotations

import json
import re
import struct
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROUTED_TRELLIS = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.w[123]\.trellis$")
Expert = tuple[int, int]
GIB = 1 << 30


class ExpertSizes:
    """Each routed expert's trellis bytes in a pack, from its safetensors headers."""

    def __init__(self, sizes: Mapping[Expert, int]):
        self.sizes = dict(sizes)

    @classmethod
    def of_pack(cls, pack: Path) -> ExpertSizes:
        weight_map = json.loads((pack / "model.safetensors.index.json").read_text())["weight_map"]
        headers: dict[str, dict[str, Any]] = {}
        sizes: dict[Expert, int] = defaultdict(int)
        for name, shard in weight_map.items():
            match = ROUTED_TRELLIS.match(name)
            if not match:
                continue
            if shard not in headers:
                with open(pack / shard, "rb") as handle:
                    length = struct.unpack("<Q", handle.read(8))[0]
                    headers[shard] = json.loads(handle.read(length))
            offsets: list[int] = headers[shard][name]["data_offsets"]
            sizes[(int(match[1]), int(match[2]))] += offsets[1] - offsets[0]
        return cls(sizes)

    @property
    def total(self) -> int:
        return sum(self.sizes.values())


@dataclass(frozen=True)
class ColdSet:
    experts: frozenset[Expert]
    bytes: int

    def to_json(self) -> dict[str, list[int]]:
        layers: dict[int, list[int]] = defaultdict(list)
        for layer, expert in sorted(self.experts):
            layers[layer].append(expert)
        return {f"layers.{layer}": ids for layer, ids in sorted(layers.items())}

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json()) + "\n")


class ColdPlanner:
    """Chooses the experts to leave in mappings: fewest routed picks per byte first, until the byte budget is met."""

    def __init__(self, sizes: ExpertSizes, hits: Mapping[Expert, int]):
        self.sizes = sizes
        self.hits = hits

    @staticmethod
    def read_hits(path: Path) -> dict[Expert, int]:
        counts: dict[str, list[int]] = json.loads(path.read_text())
        return {(int(key.split(".")[1]), e): n for key, row in counts.items() for e, n in enumerate(row)}

    @staticmethod
    def read_set(path: Path) -> frozenset[Expert]:
        layers: dict[str, list[int]] = json.loads(path.read_text())
        return frozenset((int(key.split(".")[1]), e) for key, ids in layers.items() for e in ids)

    def plan(self, budget: int) -> ColdSet:
        def cost(expert: Expert) -> tuple[float, int]:
            return self.hits.get(expert, 0) / self.sizes.sizes[expert], -self.sizes.sizes[expert]

        chosen: set[Expert] = set()
        total = 0
        for expert in sorted(self.sizes.sizes, key=cost):
            if total >= budget:
                break
            chosen.add(expert)
            total += self.sizes.sizes[expert]
        return ColdSet(frozenset(chosen), total)

    def share_of_picks(self, cold: ColdSet) -> float:
        picks = sum(self.hits.values())
        return sum(self.hits.get(e, 0) for e in cold.experts) / picks if picks else 0.0
