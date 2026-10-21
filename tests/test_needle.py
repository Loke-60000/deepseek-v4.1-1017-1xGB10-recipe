import unittest

from recipe.bench.needle import NeedleBench
from recipe.probe import Reply


class Reader:
    """Answers with whatever passphrase the prompt holds."""

    def complete(self, messages, max_tokens, extra=None, model=None):
        prompt = messages[0]["content"]
        start = prompt.index("vault is ") + len("vault is ")
        return Reply(prompt[start : prompt.index(".", start)], 3, 0.1, prompt_tokens=len(prompt) // 4)


class NeedleTest(unittest.TestCase):
    def test_the_passphrase_sits_at_the_depth_and_is_checked(self):
        result = NeedleBench(Reader()).run(tokens=20_000, depth=0.3, salt=7)
        self.assertTrue(result.found)
        self.assertGreater(result.tokens, 10_000)
        self.assertGreater(result.lowest_available_gib, 0)


if __name__ == "__main__":
    unittest.main()
