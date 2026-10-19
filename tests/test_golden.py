import unittest

from recipe.bench.golden import Answer, Golden


class GoldenTest(unittest.TestCase):
    def test_shared_prefix(self):
        reference = [Answer("code", "abcdef"), Answer("math", "391"), Answer("tools", "")]
        candidate = [Answer("code", "abcxyz"), Answer("math", "391"), Answer("tools", "")]
        self.assertEqual([a.shared_prefix for a in Golden.compare(reference, candidate)], [0.5, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
