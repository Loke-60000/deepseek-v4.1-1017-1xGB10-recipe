from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

from recipe.bench.evals import Benchmark, Item

SYSTEM = (
    "You are an expert Python programmer. You will be given a question (problem specification) and will generate a "
    "correct Python program that matches the specification and passes all tests."
)
WITH_STARTER = (
    "### Question:\n{question}\n\n### Format: You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters.\n```python\n{starter}\n```\n\n"
    "### Answer: (use the provided format with backticks)\n\n"
)
FROM_STDIN = (
    "### Question:\n{question}\n\n### Format: Read the inputs from stdin solve the problem and write the answer to "
    "stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that "
    "when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
    "```python\n# YOUR CODE HERE\n```\n\n### Answer: (use the provided format with backticks)\n\n"
)

# Runs in the sandbox (no network): the hidden tests are pickled, so they are unpacked there, never on the host.
GRADER = r'''
import base64, decimal, json, pickle, subprocess, sys, zlib
from pathlib import Path

CALL = """
import json, sys
sys.setrecursionlimit(100000)
exec(open(sys.argv[1]).read())
args = [json.loads(line) for line in sys.stdin.read().split("\\n") if line.strip()]
print(json.dumps(getattr(Solution(), sys.argv[2])(*args)))
"""

def same(got, want):
    a = [line.strip() for line in got.strip().splitlines()]
    b = [line.strip() for line in want.strip().splitlines()]
    if a == b:
        return True
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        try:
            if [decimal.Decimal(t) for t in x.split()] != [decimal.Decimal(t) for t in y.split()]:
                return False
        except decimal.InvalidOperation:
            return False
    return True

def passes(task, code):
    tests = json.loads(task["public"]) + json.loads(pickle.loads(zlib.decompress(base64.b64decode(task["private"]))))
    for test in tests:
        if task["func"]:
            argv = [sys.executable, "/tmp/call.py", code, task["func"]]
        else:
            argv = [sys.executable, code]
        try:
            run = subprocess.run(argv, input=test["input"], capture_output=True, text=True, timeout=6)
        except subprocess.TimeoutExpired:
            return False
        if run.returncode != 0:
            return False
        if task["func"]:
            try:
                if json.loads(run.stdout.strip().splitlines()[-1]) != json.loads(test["output"]):
                    return False
            except (ValueError, IndexError):
                return False
        elif not same(run.stdout, test["output"]):
            return False
    return True

root = Path(sys.argv[1])
Path("/tmp/call.py").write_text(CALL)
for task_file in sorted(root.glob("t*.json")):
    task = json.loads(task_file.read_text())
    code = root / f"{task_file.stem}.py"
    print(task_file.stem, int(code.exists() and passes(task, str(code))), flush=True)
'''


class LiveCodeBench(Benchmark):
    """LiveCodeBench v6 code generation: LiveCodeBench's own prompts and its pass@1 grading (every public and hidden
    test, 6 s each), run with no network in a throwaway container."""

    name = "livecodebench-v6"
    REPO = "livecodebench/code_generation_lite"

    def __init__(self, cache: Path, sandbox_image: str, scratch: Path, start: str, end: str):
        super().__init__(cache)
        self.sandbox_image = sandbox_image
        self.scratch = scratch
        self.window = (start, end)

    def items(self) -> list[Item]:
        rows = [json.loads(line) for line in self.fetch(self.REPO, "test6.jsonl").read_text().splitlines() if line]
        start, end = self.window
        out = []
        for row in rows:
            if not start <= row["contest_date"][:10] < end:
                continue
            starter = row["starter_code"].strip()
            task = (
                WITH_STARTER.format(question=row["question_content"], starter=starter)
                if starter
                else FROM_STDIN.format(question=row["question_content"])
            )
            prompt = f"{SYSTEM}\n\n{task}"
            func = json.loads(row["metadata"] or "{}").get("func_name", "") if starter else ""
            extra = {"public": row["public_test_cases"], "private": row["private_test_cases"], "func": func}
            out.append(Item(f"lcb-{row['question_id']}", prompt, "pass", extra))
        return out

    def parse(self, text: str) -> str | None:
        blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text, flags=re.S)
        return blocks[-1] if blocks else None

    def grade(self, items: list[Item], texts: dict[str, str]) -> dict[str, bool]:
        self.scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            Path(tmp).chmod(0o755)
            names = {}
            for n, item in enumerate(items):
                names[f"t{n}"] = item.id
                (Path(tmp) / f"t{n}.json").write_text(json.dumps(item.extra))
                code = self.parse(texts[item.id])
                if code is not None:
                    (Path(tmp) / f"t{n}.py").write_text(code)
            (Path(tmp) / "grader.py").write_text(GRADER)
            run = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "--user",
                    "65534:65534",
                    "-v",
                    f"{tmp}:/w:ro",
                    "--entrypoint",
                    "python3",
                    self.sandbox_image,
                    "/w/grader.py",
                    "/w",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        verdicts = dict(line.split() for line in run.stdout.splitlines() if re.fullmatch(r"t\d+ [01]", line))
        return {item_id: verdicts.get(name) == "1" for name, item_id in names.items()}
