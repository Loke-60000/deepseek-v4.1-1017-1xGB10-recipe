from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import urllib.request
from functools import cached_property
from pathlib import Path
from typing import Any

from recipe.bench.evals import Benchmark, Item, Sandbox

TEMPLATE = """PROBLEM DESCRIPTION:
You will be provided with problem steps along with background knowledge necessary for solving the problem. Your task \
will be to develop a Python solution focused on the next step of the problem-solving process.

PROBLEM STEPS AND FUNCTION CODE:
Here, you'll find the Python code for the initial steps of the problem-solving process. This code is integral to \
building the solution.

{problem_steps_str}

NEXT STEP - PROBLEM STEP AND FUNCTION HEADER:
This part will describe the next step in the problem-solving process. A function header will be provided, and your \
task is to develop the Python code for this next step based on the provided description and function header.

{next_step_str}

DEPENDENCIES:
Use only the following dependencies in your solution. Do not include these dependencies at the beginning of your code.

{dependencies}

RESPONSE GUIDELINES:
Now, based on the instructions and information provided above, write the complete and executable Python program for \
the next step in a single block.
Your response should focus exclusively on implementing the solution for the next step, adhering closely to the \
specified function header and the context provided by the initial steps.
Your response should NOT include the dependencies and functions of all previous steps. If your next step function \
calls functions from previous steps, please make sure it uses the headers provided without modification.
DO NOT generate EXAMPLE USAGE OR TEST CODE in your response. Please make sure your response python code in format of \
```python```."""


class SciCode(Benchmark):
    """SciCode's test problems with the scientists' background, scored per subproblem as Artificial Analysis does.
    Asked and tested the way the official harness (github.com/scicode-bench/SciCode, eval/inspect_ai/scicode.py) does
    it: a problem's steps in order, each prompt and test program carrying the model's own code for the steps before
    it, each step's tests checked against the targets in the harness's test_data.h5."""

    name = "scicode"
    published = 55.9
    HARNESS = "https://raw.githubusercontent.com/scicode-bench/SciCode/69a8cfc829fe8788a426ce8b5de6292366dce7ef"
    GIVEN = frozenset({"13.6", "62.1", "76.3"})  # the harness writes these steps itself: never asked, never scored
    # The file in the README's Google Drive folder (drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR),
    # with the id and checksum inspect_evals pins.
    TEST_DATA = (
        "https://drive.usercontent.google.com/download?id=17G_k65N_6yFFZ2O-jQH00Lh6iaw3z-AW&export=download&confirm=t"
    )
    TEST_DATA_SHA256 = "48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890"

    def __init__(self, cache: Path, sandbox: Sandbox, test_data: Path):
        super().__init__(cache)
        self.sandbox = sandbox
        self.test_data = test_data

    @cached_property
    def problems(self) -> dict[str, dict[str, Any]]:
        lines = self.fetch("SciCode1/SciCode", "problems_test.jsonl").read_text().splitlines()
        return {p["problem_id"]: p for p in (json.loads(line) for line in lines if line.strip())}

    def items(self) -> list[Item]:
        return [
            Item(step["step_number"], self.describe(step, code=True), "pass", {"problem": pid, "step": str(k)})
            for pid, problem in self.problems.items()
            for k, step in enumerate(problem["sub_steps"])
            if step["step_number"] not in self.GIVEN
        ]

    def chains(self, items: list[Item]) -> list[list[Item]]:
        problems: dict[str, list[Item]] = {}
        for item in items:
            problems.setdefault(item.extra["problem"], []).append(item)
        return list(problems.values())

    def prompt(self, item: Item, replies: dict[str, str]) -> str:
        problem, k = self.problems[item.extra["problem"]], int(item.extra["step"])
        earlier = zip(problem["sub_steps"][:k], self.previous(problem, k, replies), strict=True)
        lines = [line for step, code in earlier for line in (self.describe(step), code, "------")]
        return TEMPLATE.format(
            problem_steps_str="\n\n".join(lines[:-1]),
            next_step_str=item.prompt,
            dependencies=problem["required_dependencies"],
        )

    def parse(self, text: str) -> str | None:
        return self.code(text) if "```" in text else None

    @staticmethod
    def code(text: str) -> str:
        """The harness's extraction: the first python block (or any block, or the whole reply), imports dropped."""
        if "```" in text:
            text = text.split("```python")[1] if "```python" in text else text.split("```")[1]
            text = text.split("```")[0]
        return re.sub(r"^\s*(import .*|from .*\s+import\s+.*)", "", text, flags=re.MULTILINE)

    @staticmethod
    def describe(step: dict[str, Any], code: bool = False) -> str:
        text = f"{step['step_description_prompt']}\n{step['step_background']}"
        return f"{text}\n\n{step['function_header']}\n\n{step['return_line']}" if code else text

    def previous(self, problem: dict[str, Any], k: int, replies: dict[str, str]) -> list[str]:
        """The code of the steps before step k: the model's replies, the harness's own for the given steps."""
        numbers = [step["step_number"] for step in problem["sub_steps"][:k]]
        return [self.given[n] if n in self.GIVEN else self.code(replies[n]) for n in numbers]

    @cached_property
    def given(self) -> dict[str, str]:
        headers = {s["step_number"]: s["function_header"] for p in self.problems.values() for s in p["sub_steps"]}
        code = {}
        for number in self.GIVEN:
            with urllib.request.urlopen(f"{self.HARNESS}/eval/data/{number}.txt", timeout=60) as response:
                code[number] = self.definition(response.read().decode(), headers[number])
        return code

    @staticmethod
    def definition(source: str, header: str) -> str:
        """The harness's lookup: the header's first `def` name, else its `class` name, then the first function or
        class of that name in the source. For a class header that finds the class's `__init__`, as the harness does."""
        found = re.search(r"\bdef\s+(\w+)\s*\(", header) or re.search(r"\bclass\s+(\w+)\s*\(", header)
        if found:
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.FunctionDef | ast.ClassDef) and node.name == found.group(1):
                    return ast.unparse(node)
        raise ValueError(f"no definition for {header.splitlines()[0]!r}")

    def program(self, item: Item, texts: dict[str, str]) -> str:
        problem, k = self.problems[item.extra["problem"]], int(item.extra["step"])
        cases = problem["sub_steps"][k]["test_cases"]
        previous = "\n".join(self.previous(problem, k, texts))
        tests = "".join(f"target = targets[{i}]\n\n{case}\n" for i, case in enumerate(cases))
        return (
            f"{problem['required_dependencies']}\n{previous}\n\n{self.code(texts[item.id])}\n\n"
            "from scicode.compare.cmp import process_hdf5_to_tuple\n\n"
            f"targets = process_hdf5_to_tuple('{item.id}', {len(cases)}, '/data/test_data.h5')\n{tests}"
        )

    def grade(self, items: list[Item], texts: dict[str, str]) -> dict[str, bool]:
        test_data = self.download_test_data()
        with self.sandbox.folder() as tmp:
            (tmp / "scicode" / "compare").mkdir(parents=True)
            shutil.copy(Path(__file__).with_name("scicode_cmp.py"), tmp / "scicode" / "compare" / "cmp.py")
            files = {}
            for n, item in enumerate(items):
                files[f"/w/t{n}.py"] = item.id
                (tmp / f"t{n}.py").write_text(self.program(item, texts))
            script = (
                "python3 -c 'import h5py, scipy, sympy' || exit 3; "
                'cd /tmp && for f in /w/t*.py; do timeout 1800 python3 "$f" >/dev/null 2>&1 && echo "$f"; done; true'
            )
            out = self.sandbox.run(tmp, ["bash", "-c", script], {test_data: "/data/test_data.h5"})
        passed = {files[name] for name in out.split() if name in files}
        return {item.id: item.id in passed for item in items}

    def download_test_data(self) -> Path:
        if not self.test_data.exists():
            self.test_data.parent.mkdir(parents=True, exist_ok=True)
            part = self.test_data.with_suffix(".part")
            urllib.request.urlretrieve(self.TEST_DATA, part)
            with part.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != self.TEST_DATA_SHA256:
                raise ValueError(f"test_data.h5 from {self.TEST_DATA} has sha256 {digest}")
            part.rename(self.test_data)
        return self.test_data
