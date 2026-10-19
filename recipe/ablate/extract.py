"""Refusal directions behind an abliterated pack, from its edited residual writers (run in the toolbox image: torch).

An abliteration edits a writer W (rows = residual dimensions) to W' = W - R diag(lam) R^T W. Given both packs'
native FP8 copies of the writers, the edits' column space is recovered from the Gram matrix of the differences as an
orthonormal Q (any basis of it), and each layer's edit as the k x k operator A with W' - W = -Q A Q^T W (least
squares).
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import torch

from recipe.pack.hub import HubRepo, TensorRecord

BLOCK = 32


class SafetensorsFile:
    """A local safetensors file: header records and tensor bytes."""

    def __init__(self, path: Path):
        self.path = path
        with open(path, "rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(size))
        header.pop("__metadata__", None)
        base = 8 + size
        self.tensors = {
            name: TensorRecord(
                name, path.name, m["dtype"], tuple(m["shape"]), base + m["data_offsets"][0], base + m["data_offsets"][1]
            )
            for name, m in header.items()
        }

    def read(self, name: str) -> bytes:
        record = self.tensors[name]
        with open(self.path, "rb") as handle:
            handle.seek(record.start)
            return handle.read(record.nbytes)


class Fp8Writer:
    """A block-32 FP8 E4M3 weight with its E8M0 scales, dequantized to float32."""

    @staticmethod
    def dequantize(weight: bytes, scale: bytes, shape: tuple[int, ...]) -> torch.Tensor:
        rows, cols = shape
        w = torch.frombuffer(bytearray(weight), dtype=torch.float8_e4m3fn).reshape(rows, cols).float()
        exponents = torch.frombuffer(bytearray(scale), dtype=torch.uint8).reshape(rows // BLOCK, cols // BLOCK)
        factors = torch.pow(2.0, exponents.float() - 127.0)
        return w * factors.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)


@dataclass
class Directions:
    directions: torch.Tensor
    mix_attn: torch.Tensor
    captured: list[float]
    spectrum: list[float]

    def save(self, path: Path, first_layer: int) -> None:
        tensors = {
            "directions": self.directions.contiguous(),
            "mix_attn": self.mix_attn.contiguous(),
            "first_layer": torch.tensor([first_layer], dtype=torch.int64),
        }
        meta, blobs, cursor = {}, [], 0
        for name, tensor in tensors.items():
            data = tensor.numpy().tobytes()
            dtype = {torch.float32: "F32", torch.int64: "I64"}[tensor.dtype]
            meta[name] = {"dtype": dtype, "shape": list(tensor.shape), "data_offsets": [cursor, cursor + len(data)]}
            blobs.append(data)
            cursor += len(data)
        header = json.dumps(meta).encode()
        header += b" " * (-len(header) % 8)
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"".join(blobs))


class DirectionExtractor:
    """Recovers R and each layer's strengths from the `attn.wo_b` writers that differ between two packs."""

    def __init__(self, base: Path, edited: HubRepo, layers: list[int], rank: int):
        self.base_index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
        self.base = base
        self.edited = edited
        self.edited_index = edited.read_json("model.safetensors.index.json")
        self.layers = layers
        self.rank = rank

    def writer(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = f"layers.{layer}.attn.wo_b"
        local = SafetensorsFile(self.base / self.base_index[f"{name}.weight"])
        shape = local.tensors[f"{name}.weight"].shape
        original = Fp8Writer.dequantize(local.read(f"{name}.weight"), local.read(f"{name}.scale"), shape)
        remote = self.edited.remote(self.edited_index["weight_map"][f"{name}.weight"])
        w, s = remote.tensors[f"{name}.weight"], remote.tensors[f"{name}.scale"]
        edited = Fp8Writer.dequantize(remote.read(w.start, w.end), remote.read(s.start, s.end), shape)
        return original, edited - original

    def run(self) -> Directions:
        gram: torch.Tensor | None = None
        for layer in self.layers:
            _, delta = self.writer(layer)
            gram = delta @ delta.T if gram is None else gram + delta @ delta.T
            print(f"layer {layer}: |dW| {delta.norm():.4f}", flush=True)
        assert gram is not None
        values, vectors = torch.linalg.eigh(gram.double())
        order = torch.argsort(values, descending=True)
        top = vectors[:, order[: self.rank]].float()
        mix = torch.zeros(max(self.layers) + 1, self.rank, self.rank)
        captured = []
        for layer in self.layers:
            original, delta = self.writer(layer)
            along, moved = top.T @ original, top.T @ delta
            mix[layer] = -moved @ along.T @ torch.linalg.inv(along @ along.T)
            captured.append(float(moved.norm() ** 2 / delta.norm() ** 2))
        spectrum = [float(v) for v in values[order[: 2 * self.rank + 2]]]
        return Directions(top.T.contiguous(), mix, captured, spectrum)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--edited", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--layers", default="14-39")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lo, hi = (int(x) for x in args.layers.split("-"))
    extractor = DirectionExtractor(
        args.base, HubRepo(args.edited, args.revision, Path("/tmp/hf")), list(range(lo, hi + 1)), args.rank
    )
    found = extractor.run()
    found.save(args.out, lo)
    print("eigenvalues:", ", ".join(f"{v:.3e}" for v in found.spectrum))
    print("captured by span(R) per layer:", ", ".join(f"{c:.3f}" for c in found.captured))
    strengths = torch.linalg.eigvalsh((found.mix_attn[lo:] + found.mix_attn[lo:].transpose(1, 2)) / 2)
    per_layer = "; ".join(", ".join(f"{x:.3f}" for x in row) for row in strengths)
    print("edit strengths per layer (eigenvalues of A):", per_layer)


if __name__ == "__main__":
    main()
