import struct
import tempfile
import unittest
from pathlib import Path

from recipe.pack.warm import ENTRY, MAGIC, ColdWarmer, PreparedFile


def prepared_file(path: Path, tensors: dict[str, int]) -> None:
    entries, names = b"", list(tensors)
    for name in names:
        entries += ENTRY.pack(3, 3, len(name), 0, 1, 1, 1, 1, tensors[name], 1, bytes(32)) + name.encode()
    path.write_bytes(MAGIC + struct.pack("<II", 1, len(names)) + entries)


class WarmTest(unittest.TestCase):
    def test_prepared_header_and_expert_spans(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared_file(root / "L3.bin", {"L3.moe.w1.data": 4096, "L3.moe.w2.data": 8192, "L3.moe.w3.trellis": 16384})
            self.assertEqual(PreparedFile(root / "L3.bin").offsets["L3.moe.w2.data"], 8192)
            sizes = {(3, e, w): 10 * (e + 1) for e in range(3) for w in ("w1", "w2", "w3")}
            spans = ColdWarmer(sizes, root).spans([(3, 2)])
        self.assertEqual([(s.start, s.length) for s in spans], [(4096 + 30, 30), (8192 + 30, 30), (16384 + 30, 30)])

    def test_warm_reads_the_spans_within_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared_file(root / "L0.bin", {"L0.moe.w1.data": 4096, "L0.moe.w2.data": 8192, "L0.moe.w3.data": 12288})
            with open(root / "L0.bin", "r+b") as handle:
                handle.truncate(16384)
            sizes = {(0, e, w): 100 for e in range(2) for w in ("w1", "w2", "w3")}
            warmer = ColdWarmer(sizes, root)
            self.assertEqual(warmer.warm([(0, 1), (0, 0)], budget=450), 400)
            self.assertEqual(warmer.warm([(0, 1), (0, 0)], budget=10_000), 600)

    def test_hottest_cold_experts_first(self):
        self.assertEqual(ColdWarmer.order([(0, 1), (0, 2), (1, 0)], {(0, 2): 9, (1, 0): 4}), [(0, 2), (1, 0), (0, 1)])


if __name__ == "__main__":
    unittest.main()
