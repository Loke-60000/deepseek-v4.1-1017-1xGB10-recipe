import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from recipe.bench.evals import AIME2026, Benchmark, EvalRun, Item, Sandbox
from recipe.bench.scicode import SciCode
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


class CodingProbe(FakeProbe):
    """Answers each SciCode step with a function named as the step's header asks."""

    def complete(self, messages, max_tokens, extra=None, model=None):
        prompt = messages[0]["content"]
        self.asked.append(prompt)
        name = re.findall(r"def (\w+)\(", prompt)[-1]
        return Reply(f"Here.\n```python\nimport numpy as np\ndef {name}():\n    return 0\n```\n", 10, 0.5)


def scicode(root):
    def step(number, background="", tests=()):
        return {
            "step_number": number,
            "step_description_prompt": f"Describe {number}.",
            "step_background": background,
            "function_header": f"def f{number.replace('.', '_')}(x):\n    '''Doc.'''",
            "return_line": "    return y",
            "test_cases": list(tests),
        }

    bench = SciCode(root, Sandbox("image", root), root / "test_data.h5")
    bench.problems = {
        "62": {
            "problem_id": "62",
            "required_dependencies": "import numpy as np",
            "sub_steps": [
                step("62.1"),
                step("62.2", "Background: two."),
                step("62.3", "Background: three.", ["t = 1"]),
            ],
        },
        "9": {"problem_id": "9", "required_dependencies": "import math", "sub_steps": [step("9.1"), step("9.2")]},
    }
    bench.given = {"62.1": "class Block:\n    pass"}
    return bench


class EvalsTest(unittest.TestCase):
    def test_answers_are_parsed_from_the_last_marker(self):
        self.assertEqual(Benchmark.letter("Answer: A is wrong.\n**Answer: (c)**"), "C")
        self.assertEqual(Benchmark.letter("Answer：D"), "D")
        self.assertIsNone(Benchmark.letter("I think it is B"))
        aime = AIME2026(Path("."))
        self.assertEqual(aime.parse("so \\boxed{12} then \\boxed{ 070 }"), "70")
        self.assertIsNone(aime.parse("70"))

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

    def test_scicode_code_is_the_first_python_block_without_imports(self):
        text = (
            "x\n```python\nimport numpy as np\nfrom scipy import linalg\ndef f():\n    return 1\n```\n```python\nz\n```"
        )
        self.assertEqual(SciCode.code(text), "\n\ndef f():\n    return 1\n")
        self.assertEqual(SciCode.code("```\ny = 2\n```"), "\ny = 2\n")
        self.assertEqual(SciCode.code("y = 2"), "y = 2")
        self.assertIsNone(scicode(Path(".")).parse("y = 2"))

    def test_scicode_finds_given_code_as_the_harness_does(self):
        source = "class Block:\n    def __init__(self, n):\n        self.n = n\n\ndef other():\n    pass\n"
        header = "class Block:\n    def __init__(self, n):\n        '''Doc.'''"
        self.assertEqual(SciCode.definition(source, header), "def __init__(self, n):\n    self.n = n")
        self.assertEqual(SciCode.definition(source, "def other():"), "def other():\n    pass")

    def test_scicode_prompt_has_background_and_earlier_code(self):
        bench = scicode(Path("."))
        items = {item.id: item for item in bench.items()}
        self.assertEqual(sorted(items), ["62.2", "62.3", "9.1", "9.2"])
        replies = {"62.2": "```python\nimport numpy as np\ndef f62_2(x):\n    return 2\n```"}
        prompt = bench.prompt(items["62.3"], replies)
        earlier = (
            "Describe 62.1.\n\n\nclass Block:\n    pass\n\n------\n\n"
            "Describe 62.2.\nBackground: two.\n\n\ndef f62_2(x):\n    return 2\n\n\n"
            "NEXT STEP - PROBLEM STEP AND FUNCTION HEADER:"
        )
        self.assertIn(earlier, prompt)
        self.assertIn("Describe 62.3.\nBackground: three.\n\ndef f62_3(x):\n    '''Doc.'''\n\n    return y\n", prompt)
        self.assertIn("DEPENDENCIES:\nUse only", prompt)
        self.assertIn("\n\nimport numpy as np\n\nRESPONSE GUIDELINES:", prompt)
        program = bench.program(items["62.3"], {**replies, "62.3": "```python\ndef f62_3(x):\n    return 3\n```"})
        self.assertTrue(program.startswith("import numpy as np\nclass Block:\n    pass\n\ndef f62_2(x):"))
        self.assertIn(
            "targets = process_hdf5_to_tuple('62.3', 1, '/data/test_data.h5')\ntarget = targets[0]\n\nt = 1\n", program
        )

    def test_scicode_steps_are_asked_in_order_and_a_stopped_run_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            bench = scicode(out)
            first = {
                "id": "62.2",
                "text": "```python\ndef f62_2(x):\n    return 'kept'\n```",
                "tokens": 5,
                "seconds": 1.0,
            }
            (out / "scicode.replies.jsonl").write_text(json.dumps(first) + "\n")
            probe = CodingProbe()
            with mock.patch.object(SciCode, "grade", lambda self, items, texts: {i.id: i.id != "9.1" for i in items}):
                summary = EvalRun(probe, out, 2, 100, "high").run(bench)
            replies = [json.loads(line)["id"] for line in (out / "scicode.replies.jsonl").read_text().splitlines()]
        asked = {re.findall(r"def (\w+)\(", prompt)[-1]: prompt for prompt in probe.asked}
        self.assertEqual(sorted(asked), ["f62_3", "f9_1", "f9_2"])
        self.assertIn("return 'kept'", asked["f62_3"])
        self.assertIn("def f9_1():\n    return 0", asked["f9_2"])
        self.assertLess(replies.index("9.1"), replies.index("9.2"))
        self.assertEqual(summary["questions"], 4)
        self.assertEqual(summary["score"], 75.0)


if __name__ == "__main__":
    unittest.main()
