import json
import unittest

from recipe.bench.golden import Answer, Golden
from recipe.probe import Reply


class GoldenTest(unittest.TestCase):
    def test_shared_prefix(self):
        reference = [Answer("code", "abcdef"), Answer("math", "391"), Answer("tools", "")]
        candidate = [Answer("code", "abcxyz"), Answer("math", "391"), Answer("tools", "")]
        self.assertEqual([a.shared_prefix for a in Golden.compare(reference, candidate)], [0.5, 1.0, 1.0])

    def test_tool_call_ids_are_ignored(self):
        a = [Answer("tools", '[{"id": "call_8101bc66", "function": {"name": "f"}}]')]
        b = [Answer("tools", '[{"id": "call_6755fe7e", "function": {"name": "f"}}]')]
        self.assertEqual(Golden.compare(a, b)[0].shared_prefix, 1.0)

    def test_concurrent_recording_keeps_request_order(self):
        class Echo:
            def complete(self, messages, max_tokens, extra=None, model=None):
                return Reply(json.dumps(messages)[-40:], 1, 0.0)

        golden = Golden(Echo(), per_kind=1)
        self.assertEqual(golden.record(), golden.record(concurrency=3, stagger=0.01))


if __name__ == "__main__":
    unittest.main()
