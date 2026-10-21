from __future__ import annotations

import json
import random
import re
import subprocess
import tempfile
import urllib.error
from abc import ABC, abstractmethod
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from recipe.probe import Probe

LETTERS = "ABCDEFGHIJ"
MCQ = (
    "Answer the following multiple choice question. The last line of your response should be of the following "
    "format: 'Answer: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before "
    "answering.\n\n{question}\n\n{options}"
)


@dataclass(frozen=True)
class Item:
    id: str
    prompt: str
    answer: str
    extra: dict[str, str]


@dataclass(frozen=True)
class Outcome:
    id: str
    correct: bool
    parsed: str | None
    tokens: int
    seconds: float
    error: str | None = None


class Benchmark(ABC):
    """A question set, its prompts and its grading. Datasets come from the Hub into the local cache."""

    name: str
    published: float | None = None

    def __init__(self, cache: Path):
        self.cache = cache

    def fetch(self, repo: str, filename: str) -> Path:
        return Path(hf_hub_download(repo, filename, repo_type="dataset", cache_dir=self.cache))

    @abstractmethod
    def items(self) -> list[Item]: ...

    @abstractmethod
    def parse(self, text: str) -> str | None: ...

    def grade(self, items: list[Item], texts: dict[str, str]) -> dict[str, bool]:
        return {item.id: self.parse(texts[item.id]) == item.answer for item in items}

    @staticmethod
    def mcq(question: str, options: list[str]) -> str:
        letters = LETTERS[: len(options)]
        lines = "\n".join(f"{letter}) {option}" for letter, option in zip(letters, options, strict=True))
        return MCQ.format(letters=letters, question=question, options=lines)

    @staticmethod
    def letter(text: str) -> str | None:
        found = re.findall(r"(?i)answer\s*[:：]\s*\**\s*\(?([A-J])\b", text)
        return found[-1].upper() if found else None


class GPQADiamond(Benchmark):
    name = "gpqa-diamond"
    published = 90.9

    def items(self) -> list[Item]:
        import pandas as pd

        rows = pd.read_csv(self.fetch("Idavidrein/gpqa", "gpqa_diamond.csv"))
        out = []
        for i, row in rows.iterrows():
            options = [row["Correct Answer"], *(row[f"Incorrect Answer {k}"] for k in (1, 2, 3))]
            order = random.Random(f"gpqa-{i}").sample(range(4), 4)
            shuffled = [str(options[k]).strip() for k in order]
            answer = LETTERS[order.index(0)]
            out.append(Item(f"gpqa-{i}", self.mcq(str(row["Question"]).strip(), shuffled), answer, {}))
        return out

    def parse(self, text: str) -> str | None:
        return self.letter(text)


class MMLUPro(Benchmark):
    name = "mmlu-pro"

    def __init__(self, cache: Path, sample: int = 1000):
        super().__init__(cache)
        self.sample = sample

    def items(self) -> list[Item]:
        import pandas as pd

        rows = pd.read_parquet(self.fetch("TIGER-Lab/MMLU-Pro", "data/test-00000-of-00001.parquet"))
        picked = rows.sample(n=self.sample, random_state=0).sort_values("question_id")
        return [
            Item(
                f"mmlu-pro-{row['question_id']}",
                self.mcq(row["question"], list(row["options"])),
                row["answer"],
                {"category": row["category"]},
            )
            for row in picked.to_dict("records")
        ]

    def parse(self, text: str) -> str | None:
        return self.letter(text)


class AIME2025(Benchmark):
    name = "aime-2025"
    FILE = ("MathArena/aime_2025", "data/train-00000-of-00001.parquet")

    def items(self) -> list[Item]:
        import pandas as pd

        rows = pd.read_parquet(self.fetch(*self.FILE))
        prompt = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        return [
            Item(f"aime-2025-{row['problem_idx']}", prompt.format(problem=row["problem"]), str(int(row["answer"])), {})
            for row in rows.to_dict("records")
        ]

    def parse(self, text: str) -> str | None:
        boxed = re.findall(r"\\boxed\{\s*([^{}]*?)\s*\}", text)
        if not boxed:
            return None
        digits = re.sub(r"[^\d]", "", boxed[-1])
        return str(int(digits)) if digits else None


class AIME2026(AIME2025):
    name = "aime-2026"
    FILE = ("MathArena/aime_2026", "data/train-00000-of-00001.parquet")


class HumanEvalPlus(Benchmark):
    """HumanEval with EvalPlus's extended tests, run with no network in a throwaway container."""

    name = "humaneval-plus"
    PROMPT = (
        "Complete the following Python function. Reply with the complete function, including the signature and any "
        "imports it needs, in one ```python code block.\n\n```python\n{prompt}```"
    )
    BROKEN = frozenset({"HumanEval/32"})  # its test fails the dataset's own reference solution

    def __init__(self, cache: Path, sandbox_image: str, scratch: Path):
        super().__init__(cache)
        self.sandbox_image = sandbox_image
        self.scratch = scratch

    def items(self) -> list[Item]:
        lines = self.fetch("evalplus/humanevalplus", "test.jsonl").read_text().splitlines()
        tasks = [t for t in (json.loads(line) for line in lines if line.strip()) if t["task_id"] not in self.BROKEN]
        return [
            Item(
                t["task_id"],
                self.PROMPT.format(prompt=t["prompt"]),
                "pass",
                {"test": t["test"], "entry": t["entry_point"]},
            )
            for t in tasks
        ]

    def parse(self, text: str) -> str | None:
        blocks = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.S)
        return blocks[-1] if blocks else None

    def grade(self, items: list[Item], texts: dict[str, str]) -> dict[str, bool]:
        self.scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            Path(tmp).chmod(0o755)
            files = {}
            for n, item in enumerate(items):
                code = self.parse(texts[item.id])
                if code is None:
                    continue
                files[f"t{n}.py"] = item.id
                program = f"{code}\n\n{item.extra['test']}\n\ncheck({item.extra['entry']})\n"
                (Path(tmp) / f"t{n}.py").write_text(program)
            script = 'cd /w && for f in t*.py; do timeout 30 python3 "$f" >/dev/null 2>&1 && echo "$f"; done; true'
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
                    "bash",
                    self.sandbox_image,
                    "-c",
                    script,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            passed = {files[name] for name in run.stdout.split() if name in files}
        return {item.id: item.id in passed for item in items}


class EvalRun:
    """Sends a benchmark's prompts to the local server, a few at a time, and keeps every reply on disk: a run that
    stops (a restart, a lost connection) picks up where it left off."""

    def __init__(self, probe: Probe, out: Path, concurrency: int, max_tokens: int, effort: str):
        self.probe = probe
        self.out = out
        self.concurrency = concurrency
        self.max_tokens = max_tokens
        self.effort = effort

    def run(self, bench: Benchmark) -> dict[str, Any]:
        self.out.mkdir(parents=True, exist_ok=True)
        replies_path = self.out / f"{bench.name}.replies.jsonl"
        items = bench.items()
        done = {r["id"]: r for r in self._read(replies_path)}
        todo = [item for item in items if item.id not in done]
        print(f"{bench.name}: {len(items)} questions, {len(done)} already answered", flush=True)
        with ThreadPoolExecutor(self.concurrency) as pool, replies_path.open("a") as log:
            for future in as_completed([pool.submit(self._ask, item) for item in todo]):
                record = future.result()
                log.write(json.dumps(record) + "\n")
                log.flush()
                done[record["id"]] = record
                print(f"{bench.name}: {len(done)}/{len(items)}", flush=True)
        texts = {item.id: done[item.id]["text"] for item in items}
        verdicts = bench.grade(items, texts)
        outcomes = [
            Outcome(
                item.id,
                verdicts[item.id],
                bench.parse(texts[item.id]),
                done[item.id]["tokens"],
                done[item.id]["seconds"],
                done[item.id].get("error"),
            )
            for item in items
        ]
        (self.out / f"{bench.name}.graded.jsonl").write_text("".join(json.dumps(asdict(o)) + "\n" for o in outcomes))
        summary = self.summary(bench, outcomes)
        (self.out / f"{bench.name}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    def _ask(self, item: Item) -> dict[str, Any]:
        extra = {"reasoning_effort": self.effort}
        for attempt in range(3):
            try:
                reply = self.probe.complete([{"role": "user", "content": item.prompt}], self.max_tokens, extra)
                return {"id": item.id, "text": reply.text, "tokens": reply.completion_tokens, "seconds": reply.seconds}
            except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
                if attempt == 2:
                    return {"id": item.id, "text": "", "tokens": 0, "seconds": 0.0, "error": str(error)}
                self.probe.wait_ready()
        raise AssertionError("unreachable")

    def summary(self, bench: Benchmark, outcomes: list[Outcome]) -> dict[str, Any]:
        n = len(outcomes)
        return {
            "benchmark": bench.name,
            "questions": n,
            "score": round(100 * sum(o.correct for o in outcomes) / n, 1),
            "published_full_model": bench.published,
            "unparsed": sum(o.parsed is None for o in outcomes),
            "errors": sum(o.error is not None for o in outcomes),
            "reasoning_effort": self.effort,
            "max_tokens": self.max_tokens,
            "mean_tokens": round(sum(o.tokens for o in outcomes) / n),
        }

    @staticmethod
    def _read(path: Path) -> Iterator[dict[str, Any]]:
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    yield json.loads(line)
