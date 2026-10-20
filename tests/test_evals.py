import json
import tempfile
import unittest
from pathlib import Path

from recipe.bench.evals import AIME2025, Benchmark, EvalRun, HumanEvalPlus, Item
from recipe.probe import Reply


class Fixed(Benchmark):
    name = "fixed"

    def items(self):
        return [Item(f"q{i}", f"question {i}", "B", {}) for i in range(3)]

    def parse(self, text):
        return self.letter(text)


class FakeProbe:
    def __init__(self):
        self.asked = []

    def complete(self, messages, max_tokens, extra=None, model=None):
        self.asked.append(messages[0]["content"])
        return Reply("thinking...\nAnswer: B", 10, 0.5)

    def wait_ready(self):
        return []


class EvalsTest(unittest.TestCase):
    def test_answers_are_parsed_from_the_last_marker(self):
        self.assertEqual(Benchmark.letter("Answer: A is wrong.\n**Answer: (c)**"), "C")
        self.assertEqual(Benchmark.letter("Answer：D"), "D")
        self.assertIsNone(Benchmark.letter("I think it is B"))
        aime = AIME2025(Path("."))
        self.assertEqual(aime.parse("so \\boxed{12} then \\boxed{ 070 }"), "70")
        self.assertIsNone(aime.parse("70"))
        code = HumanEvalPlus(Path("."), "image", Path(".")).parse("x\n```python\ndef f():\n    return 1\n```\n")
        self.assertEqual(code, "def f():\n    return 1\n")

    def test_options_are_lettered(self):
        prompt = Benchmark.mcq("Q?", ["one", "two"])
        self.assertIn("A) one\nB) two", prompt)
        self.assertIn("one of AB.", prompt)

    def test_a_stopped_run_resumes_and_grades_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "fixed.replies.jsonl").write_text(
                json.dumps({"id": "q0", "text": "Answer: A", "tokens": 5, "seconds": 1.0}) + "\n"
            )
            probe = FakeProbe()
            summary = EvalRun(probe, out, 2, 100, "high").run(Fixed(out))
        self.assertEqual(sorted(probe.asked), ["question 1", "question 2"])
        self.assertEqual(summary["questions"], 3)
        self.assertEqual(summary["score"], 66.7)


if __name__ == "__main__":
    unittest.main()
