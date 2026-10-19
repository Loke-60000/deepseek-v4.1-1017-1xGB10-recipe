import tempfile
import unittest
from pathlib import Path

from recipe.settings import Environment, Settings


class SettingsTest(unittest.TestCase):
    def test_dotenv_is_layered_under_the_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            dotenv = Path(tmp) / ".env"
            dotenv.write_text("# comment\nDSV41_PATHS_ROOT='~/a'\nDSV41_SERVE_PORT=1\n\n")
            env = Environment.load(dotenv, {"DSV41_SERVE_PORT": "2"})
        self.assertEqual(env, {"DSV41_PATHS_ROOT": "~/a", "DSV41_SERVE_PORT": "2"})

    def test_paths_expand_home(self):
        settings = Settings.load(env={"DSV41_PATHS_ROOT": "~/x"})
        self.assertEqual(settings.paths.root, Path.home() / "x")
        self.assertFalse(str(settings.paths.hf_cache).startswith("~"))


if __name__ == "__main__":
    unittest.main()
