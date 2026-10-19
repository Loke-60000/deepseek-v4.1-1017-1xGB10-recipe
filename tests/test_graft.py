import json
import struct
import tempfile
import unittest
from pathlib import Path

from recipe.pack.graft import DenseWriter, GraftBuilder, expert_of
from recipe.pack.hub import TensorRecord


class FakeRemote:
    def __init__(self, blob: bytes):
        self.blob = blob
        self.reads = 0

    def read(self, start: int, end: int) -> bytes:
        self.reads += 1
        return self.blob[start:end]


def safetensors(tensors: dict[str, bytes]) -> tuple[bytes, dict[str, TensorRecord]]:
    meta, cursor = {}, 0
    for name, data in tensors.items():
        meta[name] = {"dtype": "U8", "shape": [len(data)], "data_offsets": [cursor, cursor + len(data)]}
        cursor += len(data)
    header = json.dumps(meta).encode()
    base = 8 + len(header)
    blob = struct.pack("<Q", len(header)) + header + b"".join(tensors.values())
    records = {
        n: TensorRecord(n, "src", "U8", (len(d),), base + m["data_offsets"][0], base + m["data_offsets"][1])
        for (n, d), m in zip(tensors.items(), meta.values(), strict=True)
    }
    return blob, records


def read_back(path: Path) -> dict[str, bytes]:
    raw = path.read_bytes()
    size = struct.unpack("<Q", raw[:8])[0]
    meta = json.loads(raw[8 : 8 + size])
    return {n: raw[8 + size + m["data_offsets"][0] : 8 + size + m["data_offsets"][1]] for n, m in meta.items()}


class DenseWriterTest(unittest.TestCase):
    def test_copies_selected_tensors_across_chunks(self):
        tensors = {"a": bytes(range(200)) * 3, "skip": b"x" * 4000, "b": b"\x07" * 1500, "c": b"\x09" * 10}
        blob, records = safetensors(tensors)
        remote = FakeRemote(blob)
        wanted = [records["c"], records["a"], records["b"]]
        writer = DenseWriter(wanted, {"src": remote}, workers=3)
        writer.MAX_GAP, writer.CHUNK = 64, 256
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dense.safetensors"
            writer.write(out)
            self.assertEqual(read_back(out), {n: tensors[n] for n in ("a", "b", "c")})
        self.assertGreater(remote.reads, 3)


class SwapTest(unittest.TestCase):
    def test_unequal_gate_up_widths(self):
        widths = {(0, 1, "w1"): 16, (0, 1, "w3"): 32, (0, 2, "w1"): 16, (0, 2, "w3"): 16, (0, 1, "w2"): 48}
        self.assertEqual(GraftBuilder.unequal(widths), {(0, 1)})

    def test_expert_of(self):
        self.assertEqual(expert_of("layers.12.ffn.experts.300.w2.svh"), (12, 300))
        self.assertIsNone(expert_of("layers.12.ffn.shared_experts.w2.weight"))


if __name__ == "__main__":
    unittest.main()
