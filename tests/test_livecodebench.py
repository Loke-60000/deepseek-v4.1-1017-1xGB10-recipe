import base64
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

from recipe.bench.livecodebench import GRADER, LiveCodeBench


def hidden(tests):
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode()


STDIN = {
    "public": json.dumps([{"input": "2 3\n", "output": "5\n", "testtype": "stdin"}]),
    "private": hidden([{"input": "10 -4\n", "output": "6", "testtype": "stdin"}]),
    "func": "",
}
CALL = {
    "public": json.dumps([{"input": "[1, 2, 3]\n2", "output": "[2, 4, 6]", "testtype": "functional"}]),
    "private": hidden([{"input": "[]\n5", "output": "[]", "testtype": "functional"}]),
    "func": "scale",
}
FLOAT = {
    "public": json.dumps([{"input": "", "output": "0.50000\n", "testtype": "stdin"}]),
    "private": hidden([]),
    "func": "",
}


class LiveCodeBenchTest(unittest.TestCase):
    def grade(self, cases):
        with tempfile.TemporaryDirectory() as tmp:
            for name, (task, code) in cases.items():
                (Path(tmp) / f"{name}.json").write_text(json.dumps(task))
                if code is not None:
                    (Path(tmp) / f"{name}.py").write_text(code)
            (Path(tmp) / "grader.py").write_text(GRADER)
            run = subprocess.run([sys.executable, f"{tmp}/grader.py", tmp], capture_output=True, text=True, check=True)
        return dict(line.split() for line in run.stdout.splitlines())

    def test_hidden_and_public_tests_both_count(self):
        verdicts = self.grade(
            {
                "t0": (STDIN, "a, b = map(int, input().split())\nprint(a + b)\n"),
                "t1": (STDIN, "a, b = map(int, input().split())\nprint(5)\n"),
                "t2": (CALL, "class Solution:\n    def scale(self, xs, k):\n        return [x * k for x in xs]\n"),
                "t3": (CALL, "class Solution:\n    def scale(self, xs, k):\n        return xs\n"),
                "t4": (FLOAT, "print(0.5)\n"),
                "t5": (STDIN, None),
            }
        )
        self.assertEqual(verdicts, {"t0": "1", "t1": "0", "t2": "1", "t3": "0", "t4": "1", "t5": "0"})

    def test_code_is_taken_from_the_last_python_block(self):
        bench = LiveCodeBench(Path("."), "image", Path("."), "2025-02-01", "2025-05-01")
        self.assertEqual(bench.parse("```python\nx = 1\n```\nbetter:\n```python\nx = 2\n```"), "x = 2\n")


if __name__ == "__main__":
    unittest.main()
