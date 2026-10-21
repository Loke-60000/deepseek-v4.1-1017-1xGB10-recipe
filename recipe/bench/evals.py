from __future__ import annotations

import json
import random
import re
import subprocess
import tempfile
import threading
import urllib.error
from abc import ABC, abstractmethod
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
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

    def chains(self, items: list[Item]) -> list[list[Item]]:
        """Items asked one after another, in order; different chains are asked in parallel."""
        return [[item] for item in items]

    def prompt(self, item: Item, replies: dict[str, str]) -> str:
        """The prompt for an item, given the replies so far: those to the items before it in its chain are there."""
        return item.prompt

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


class AIME2026(Benchmark):
    name = "aime-2026"

    def items(self) -> list[Item]:
        import pandas as pd

        rows = pd.read_parquet(self.fetch("MathArena/aime_2026", "data/train-00000-of-00001.parquet"))
        prompt = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        return [
            Item(f"aime-2026-{row['problem_idx']}", prompt.format(problem=row["problem"]), str(int(row["answer"])), {})
            for row in rows.to_dict("records")
        ]

    def parse(self, text: str) -> str | None:
        boxed = re.findall(r"\\boxed\{\s*([^{}]*?)\s*\}", text)
        if not boxed:
            return None
        digits = re.sub(r"[^\d]", "", boxed[-1])
        return str(int(digits)) if digits else None


class Sandbox:
    """A throwaway container in the toolbox image, with no network and as nobody, for code the model wrote: the files go
    in a world-readable scratch folder mounted read-only at /w."""

    def __init__(self, image: str, scratch: Path):
        self.image = image
        self.scratch = scratch

    @contextmanager
    def folder(self) -> Iterator[Path]:
        self.scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            Path(tmp).chmod(0o755)
            yield Path(tmp)

    def run(self, folder: Path, command: list[str], mounts: dict[Path, str] | None = None) -> str:
        volumes = [f"{folder}:/w:ro", *(f"{host}:{inside}:ro" for host, inside in (mounts or {}).items())]
        argv = ["docker", "run", "--rm", "--network", "none", "--user", "65534:65534"]
        argv += [flag for volume in volumes for flag in ("-v", volume)]
        argv += ["--entrypoint", command[0], self.image, *command[1:]]
        return subprocess.run(argv, capture_output=True, text=True, check=True).stdout


class Replies:
    """A benchmark's replies so far, each appended to a JSONL file as it arrives."""

    def __init__(self, path: Path):
        self.path = path
        self.done = {r["id"]: r for r in self._read(path)}
        self.lock = threading.Lock()

    def texts(self) -> dict[str, str]:
        with self.lock:
            return {key: record["text"] for key, record in self.done.items()}

    def add(self, record: dict[str, Any]) -> int:
        with self.lock, self.path.open("a") as log:
            log.write(json.dumps(record) + "\n")
            self.done[record["id"]] = record
            return len(self.done)

    @staticmethod
    def _read(path: Path) -> Iterator[dict[str, Any]]:
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    yield json.loads(line)


class EvalRun:
    """Sends a benchmark's prompts to the local server, a few chains at a time, and keeps every reply on disk: a run
    that stops (a restart, a lost connection) picks up where it left off."""

    def __init__(self, probe: Probe, out: Path, concurrency: int, max_tokens: int, effort: str):
        self.probe = probe
        self.out = out
        self.concurrency = concurrency
        self.max_tokens = max_tokens
        self.effort = effort

    def run(self, bench: Benchmark) -> dict[str, Any]:
        self.out.mkdir(parents=True, exist_ok=True)
        replies = Replies(self.out / f"{bench.name}.replies.jsonl")
        items = bench.items()
        done = replies.done
        todo = [chain for chain in bench.chains(items) if any(item.id not in done for item in chain)]
        print(f"{bench.name}: {len(items)} questions, {len(done)} already answered", flush=True)
        with ThreadPoolExecutor(self.concurrency) as pool:
            futures = [pool.submit(self._ask_chain, bench, chain, replies, len(items)) for chain in todo]
            for future in as_completed(futures):
                future.result()
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

    def _ask_chain(self, bench: Benchmark, chain: list[Item], replies: Replies, total: int) -> None:
        for item in chain:
            if item.id not in replies.done:
                answered = replies.add(self._ask(item.id, bench.prompt(item, replies.texts())))
                print(f"{bench.name}: {answered}/{total}", flush=True)

    def _ask(self, item_id: str, prompt: str) -> dict[str, Any]:
        extra = {"reasoning_effort": self.effort}
        for attempt in range(3):
            try:
                reply = self.probe.complete([{"role": "user", "content": prompt}], self.max_tokens, extra)
                return {"id": item_id, "text": reply.text, "tokens": reply.completion_tokens, "seconds": reply.seconds}
            except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
                if attempt == 2:
                    return {"id": item_id, "text": "", "tokens": 0, "seconds": 0.0, "error": str(error)}
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
