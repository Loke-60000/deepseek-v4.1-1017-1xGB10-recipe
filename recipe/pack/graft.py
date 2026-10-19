from __future__ import annotations

import json
import os
import re
import shutil
import struct
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from recipe.pack.hub import HubRepo, RemoteFile, TensorRecord
from recipe.settings import Settings

ROUTED = re.compile(r"^layers\.\d+\.ffn\.experts\.\d+\.")
EXPERT = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.")
INDEX = "model.safetensors.index.json"
DENSE_FILE = "graft-dense.safetensors"
MANIFEST = "graft-manifest.json"
SIDE_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


@dataclass
class Span:
    file: str
    start: int
    end: int
    records: list[tuple[TensorRecord, int]]


class DenseWriter:
    """Copies remote tensors into one local safetensors file, fetching coalesced byte spans in parallel."""

    MAX_GAP = 1 << 20
    CHUNK = 64 << 20

    def __init__(self, records: list[TensorRecord], remotes: dict[str, RemoteFile], workers: int):
        self.records = sorted(records, key=lambda r: r.name)
        self.remotes = remotes
        self.workers = workers

    def write(self, path: Path) -> int:
        header, offsets, data_len = self._layout()
        tmp = path.with_suffix(".partial")
        with open(tmp, "wb") as out:
            out.write(len(header).to_bytes(8, "little"))
            out.write(header)
            out.truncate(8 + len(header) + data_len)
        fd = os.open(tmp, os.O_WRONLY)
        try:
            base = 8 + len(header)
            with ThreadPoolExecutor(self.workers) as pool:
                for _ in pool.map(lambda span: self._copy(fd, base, span), self._spans(offsets)):
                    pass
            os.fsync(fd)
        finally:
            os.close(fd)
        tmp.rename(path)
        return data_len

    def _layout(self) -> tuple[bytes, dict[str, int], int]:
        meta, offsets, cursor = {}, {}, 0
        for record in self.records:
            offsets[record.name] = cursor
            meta[record.name] = {
                "dtype": record.dtype,
                "shape": list(record.shape),
                "data_offsets": [cursor, cursor + record.nbytes],
            }
            cursor += record.nbytes
        header = json.dumps(meta, separators=(",", ":")).encode()
        header += b" " * (-len(header) % 8)
        return header, offsets, cursor

    def _spans(self, offsets: dict[str, int]) -> list[Span]:
        spans: list[Span] = []
        for record in sorted(self.records, key=lambda r: (r.file, r.start)):
            last = spans[-1] if spans else None
            if (
                last
                and last.file == record.file
                and record.start - last.end <= self.MAX_GAP
                and record.end - last.start <= self.CHUNK
            ):
                last.end = max(last.end, record.end)
                last.records.append((record, offsets[record.name]))
            else:
                spans.append(Span(record.file, record.start, record.end, [(record, offsets[record.name])]))
        return spans

    def _copy(self, fd: int, base: int, span: Span) -> None:
        remote = self.remotes[span.file]
        for lo in range(span.start, span.end, self.CHUNK):
            hi = min(lo + self.CHUNK, span.end)
            data = memoryview(remote.read(lo, hi))
            for record, offset in span.records:
                start, end = max(record.start, lo), min(record.end, hi)
                if start < end:
                    os.pwrite(fd, data[start - lo : end - lo], base + offset + start - record.start)


def expert_of(name: str) -> tuple[int, int] | None:
    match = EXPERT.match(name)
    return (int(match[1]), int(match[2])) if match else None


class GraftBuilder:
    """A pack folder: routed experts from the experts pack, every other tensor from the dense pack.

    An expert whose gate and up projections the experts pack stores at different widths is taken whole from the dense
    pack instead: the fused gate/up prefill kernel reads both at one width.
    """

    def __init__(self, settings: Settings, workers: int = 8):
        self.out = settings.paths.pack
        self.workers = workers
        cache = settings.paths.hf_cache
        self.experts = HubRepo(settings.sources.experts_repo, settings.sources.experts_revision, cache)
        self.dense = HubRepo(settings.sources.dense_repo, settings.sources.dense_revision, cache)

    def build(self) -> Path:
        self.out.mkdir(parents=True, exist_ok=True)
        routed = self._routed_map()
        swapped = self._mismatched(routed)
        kept = {name: shard for name, shard in routed.items() if expert_of(name) not in swapped}
        records = self._dense_records(set(routed), set(kept))
        manifest = {
            "experts": {"repo": self.experts.repo_id, "revision": self.experts.revision, "tensors": len(kept)},
            "dense": {
                "repo": self.dense.repo_id,
                "revision": self.dense.revision,
                "tensors": len(records),
                "bytes": sum(r.nbytes for r in records),
                "experts_with_unequal_gate_up": len(swapped),
            },
        }
        if not (self.out / DENSE_FILE).exists() or self._manifest() != manifest:
            remotes = {name: self.dense.remote(name) for name in {r.file for r in records}}
            DenseWriter(records, remotes, self.workers).write(self.out / DENSE_FILE)
        self._link_shards(set(kept.values()))
        self._copy_side_files()
        weight_map = kept | {record.name: DENSE_FILE for record in records}
        self._write_json(INDEX, {"metadata": {}, "weight_map": dict(sorted(weight_map.items()))})
        self._write_json(MANIFEST, manifest)
        return self.out

    def _manifest(self) -> object:
        path = self.out / MANIFEST
        return json.loads(path.read_text()) if path.exists() else None

    def _mismatched(self, routed: dict[str, str]) -> set[tuple[int, int]]:
        widths: dict[tuple[int, int, str], int] = {}
        for shard in sorted(set(routed.values())):
            with open(self.experts.snapshot / shard, "rb") as handle:
                size = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(size))
            for name, meta in header.items():
                match = EXPERT.match(name)
                if match and name.endswith(".trellis") and match[3] in ("w1", "w3"):
                    widths[(int(match[1]), int(match[2]), match[3])] = meta["shape"][-1]
        return self.unequal(widths)

    @staticmethod
    def unequal(widths: dict[tuple[int, int, str], int]) -> set[tuple[int, int]]:
        return {(layer, e) for (layer, e, w), w1 in widths.items() if w == "w1" and widths[(layer, e, "w3")] != w1}

    def _routed_map(self) -> dict[str, str]:
        index = json.loads((self.experts.snapshot / INDEX).read_text())["weight_map"]
        return {name: shard for name, shard in index.items() if ROUTED.match(name)}

    def _dense_records(self, routed: set[str], kept: set[str]) -> list[TensorRecord]:
        shards = [name for name in self.dense.files if name.endswith(".safetensors")]
        with ThreadPoolExecutor(self.workers) as pool:
            headers = list(pool.map(lambda name: self.dense.remote(name).tensors, shards))
        everything = {name: record for header in headers for name, record in header.items()}
        dense_routed = {name for name in everything if ROUTED.match(name)}
        if dense_routed != routed:
            raise ValueError(f"routed experts differ: {len(dense_routed ^ routed)} names are in one pack only")
        return [record for name, record in everything.items() if name not in kept]

    def _link_shards(self, shards: set[str]) -> None:
        for shard in sorted(shards):
            target = self.out / shard
            source = (self.experts.snapshot / shard).resolve()
            if target.exists() and target.stat().st_ino == source.stat().st_ino:
                continue
            target.unlink(missing_ok=True)
            os.link(source, target)

    def _copy_side_files(self) -> None:
        for name in SIDE_FILES:
            if name in self.dense.files:
                shutil.copyfile(self.dense.download(name), self.out / name)

    def _write_json(self, name: str, payload: Mapping[str, object]) -> None:
        (self.out / name).write_text(json.dumps(payload, indent=1) + "\n")
