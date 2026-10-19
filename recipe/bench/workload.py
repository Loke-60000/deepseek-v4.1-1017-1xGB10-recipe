from __future__ import annotations

import json
import random
import sysconfig
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search a repository for files whose contents match a regular expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression"},
                    "path": {"type": "string", "description": "Directory to search"},
                    "max_results": {"type": "integer"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    },
]

TOOL_ASKS = [
    "Find every place in ./src that calls subprocess.run without check=True, then summarise the risk.",
    "What's the weather in Lyon and in Berlin right now? Compare them.",
    "Search the repo under ./engine for TODO comments mentioning memory and list the files.",
    "Is it warmer in Tokyo or in Sydney today, in fahrenheit?",
]

MULTILINGUAL = [
    "Explique en détail comment fonctionne un mélange d'experts dans un modèle de langage, avec un exemple chiffré.",
    "Erkläre ausführlich, wie ein Speicherverwaltungssystem mit virtueller Adressierung funktioniert.",
    "Describe con detalle la historia de la imprenta y su impacto en Europa.",
    "请详细解释量子纠缠的基本原理，并举一个通俗的例子。",
    "日本の四季の行事について、それぞれ具体例を挙げて詳しく説明してください。",
    "اشرح بالتفصيل كيف تعمل محركات البحث على الإنترنت في فهرسة الصفحات.",
]


@dataclass(frozen=True)
class Request:
    kind: str
    messages: list[dict[str, Any]]
    max_tokens: int
    extra: dict[str, Any] = field(default_factory=dict)


class Workload:
    """A deterministic mix of code, prose, math, tool calls and multilingual chat for routing calibration."""

    def __init__(self, seed: int = 1017, prose_roots: tuple[Path, ...] = (Path("/usr/share/doc"),)):
        self.random = random.Random(seed)
        self.prose_roots = prose_roots

    def requests(self, per_kind: int) -> Iterator[Request]:
        makers = [self._code, self._prose, self._math, self._tools, self._multilingual]
        for i in range(per_kind):
            for make in makers:
                yield make(i)

    def _code(self, i: int) -> Request:
        files = sorted(Path(sysconfig.get_paths()["stdlib"]).glob("*.py"))
        source = self.random.choice(files).read_text(errors="replace")[:24000]
        ask = [
            "Review this module for bugs and explain the riskiest part.",
            "Summarise what this code does, then write a unit test for its most complex function.",
            "Refactor the longest function for clarity.",
        ][i % 3]
        return Request("code", [{"role": "user", "content": f"{ask}\n\n```python\n{source}\n```"}], 256)

    def _prose(self, i: int) -> Request:
        texts = [
            p for root in self.prose_roots for p in root.rglob("*") if p.suffix in {".md", ".txt", ""} and p.is_file()
        ]
        text = ""
        while len(text) < 8000 and texts:
            text += self.random.choice(texts).read_text(errors="replace")[:6000] + "\n\n"
        content = f"Summarise the following notes for a newcomer:\n\n{text}"
        return Request("prose", [{"role": "user", "content": content}], 256)

    def _math(self, i: int) -> Request:
        a, b, c = (self.random.randint(12, 999) for _ in range(3))
        problem = (
            f"A warehouse ships {a} boxes on Monday, {b} more than Monday on Tuesday, and on Wednesday a third "
            f"of the two days' total plus {c}. How many boxes ship over the three days? Show your reasoning."
        )
        return Request("math", [{"role": "user", "content": problem}], 384)

    def _tools(self, i: int) -> Request:
        ask = TOOL_ASKS[i % len(TOOL_ASKS)]
        return Request("tools", [{"role": "user", "content": ask}], 192, {"tools": TOOLS, "tool_choice": "auto"})

    def _multilingual(self, i: int) -> Request:
        return Request("multilingual", [{"role": "user", "content": MULTILINGUAL[i % len(MULTILINGUAL)]}], 256)

    @staticmethod
    def describe(requests: list[Request]) -> str:
        chars = sum(len(json.dumps(r.messages)) for r in requests)
        return f"{len(requests)} requests, ~{chars // 4:,} prompt tokens"
