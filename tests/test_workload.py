import tempfile
import unittest
from collections import Counter
from pathlib import Path

from recipe.bench.workload import Workload


class WorkloadTest(unittest.TestCase):
    def test_mix_is_balanced_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "notes.md").write_text("Memory pressure notes. " * 400)
            first = list(Workload(prose_roots=(Path(tmp),)).requests(3))
            second = list(Workload(prose_roots=(Path(tmp),)).requests(3))
        kinds = ("code", "prose", "math", "tools", "multilingual")
        self.assertEqual(Counter(r.kind for r in first), Counter(dict.fromkeys(kinds, 3)))
        self.assertEqual([r.messages for r in first], [r.messages for r in second])
        self.assertTrue(all(r.extra.get("tools") for r in first if r.kind == "tools"))


if __name__ == "__main__":
    unittest.main()
