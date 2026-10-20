import unittest

from recipe.bench.golden import Answer, Golden


class GoldenTest(unittest.TestCase):
    def test_shared_prefix(self):
        reference = [Answer("code", "abcdef"), Answer("math", "391"), Answer("tools", "")]
        candidate = [Answer("code", "abcxyz"), Answer("math", "391"), Answer("tools", "")]
        self.assertEqual([a.shared_prefix for a in Golden.compare(reference, candidate)], [0.5, 1.0, 1.0])

    def test_tool_call_ids_are_ignored(self):
        a = [Answer("tools", '[{"id": "call_8101bc66", "function": {"name": "f"}}]')]
        b = [Answer("tools", '[{"id": "call_6755fe7e", "function": {"name": "f"}}]')]
        self.assertEqual(Golden.compare(a, b)[0].shared_prefix, 1.0)


if __name__ == "__main__":
    unittest.main()
