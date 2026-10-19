import json
import struct
import tempfile
import unittest
from pathlib import Path

from recipe.pack.cold import ColdPlanner, ExpertSizes


def write_pack(directory: Path, sizes: dict[tuple[int, int], int]) -> None:
    meta, weight_map, cursor = {}, {}, 0
    for (layer, expert), size in sizes.items():
        for w in ("w1", "w2", "w3"):
            name = f"layers.{layer}.ffn.experts.{expert}.{w}.trellis"
            meta[name] = {"dtype": "I16", "shape": [size // 2], "data_offsets": [cursor, cursor + size]}
            weight_map[name] = "a.safetensors"
            cursor += size
    weight_map["layers.0.attn.wq_a.trellis"] = "a.safetensors"
    meta["layers.0.attn.wq_a.trellis"] = {"dtype": "I16", "shape": [1], "data_offsets": [cursor, cursor + 2]}
    header = json.dumps(meta).encode()
    (directory / "a.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


class ColdPlannerTest(unittest.TestCase):
    def setUp(self):
        self.sizes = ExpertSizes({(0, 0): 100, (0, 1): 100, (1, 0): 300, (1, 1): 50})

    def test_pack_sizes_sum_projections(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_pack(Path(tmp), {(0, 0): 10, (3, 7): 4})
            sizes = ExpertSizes.of_pack(Path(tmp))
        self.assertEqual(sizes.sizes, {(0, 0): 30, (3, 7): 12})

    def test_fewest_hits_per_byte_first(self):
        hits = {(0, 0): 50, (0, 1): 1, (1, 0): 3, (1, 1): 40}
        cold = ColdPlanner(self.sizes, hits).plan(budget=350)
        self.assertEqual(cold.experts, {(1, 0), (0, 1)})
        self.assertEqual(cold.bytes, 400)

    def test_unseen_experts_go_first_largest_first(self):
        cold = ColdPlanner(self.sizes, {(0, 0): 5, (0, 1): 5}).plan(budget=1)
        self.assertEqual(cold.experts, {(1, 0)})

    def test_share_and_json(self):
        hits = {(0, 0): 9, (0, 1): 1, (1, 0): 0, (1, 1): 0}
        planner = ColdPlanner(self.sizes, hits)
        cold = planner.plan(budget=400)
        self.assertAlmostEqual(planner.share_of_picks(cold), 0.1)
        self.assertEqual(cold.to_json(), {"layers.0": [1], "layers.1": [0, 1]})

    def test_hits_file_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hits.json"
            path.write_text(json.dumps({"layers.2": [4, 0, 7]}))
            self.assertEqual(ColdPlanner.read_hits(path), {(2, 0): 4, (2, 1): 0, (2, 2): 7})


if __name__ == "__main__":
    unittest.main()
