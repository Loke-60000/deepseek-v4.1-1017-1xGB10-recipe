import json
import tempfile
import unittest
from pathlib import Path

from recipe.profile import Profile


class ProfileTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.write(
            "base",
            {
                "serve": {"model_id": "a", "aliases": [], "port": 1, "max_tokens": 2},
                "env": {"TF_DSV41_CONTEXT": "10", "TF_DSV41_SLOTS": "4", "TF_DSV41_IMAGES": "native"},
            },
        )
        self.write(
            "child", {"extends": "base", "serve": {"port": 9}, "env": {"TF_DSV41_SLOTS": "1", "TF_DSV41_IMAGES": None}}
        )

    def write(self, name, data):
        (self.dir / f"{name}.json").write_text(json.dumps(data))

    def test_extends_overrides_and_removes(self):
        profile = Profile.load("child", self.dir, environ={})
        self.assertEqual((profile.slots, profile.context, profile.serve.port), (1, 10, 9))
        self.assertNotIn("TF_DSV41_IMAGES", profile.env)

    def test_environment_wins(self):
        environ = {"TF_DSV41_SLOTS": "8", "DSV41_SERVE_MODEL_ID": "gl-zero", "DSV41_SERVE_ALIASES": "x, y", "HOME": "/"}
        profile = Profile.load("child", self.dir, environ=environ)
        self.assertEqual((profile.slots, profile.serve.model_id, profile.serve.aliases), (8, "gl-zero", ("x", "y")))
        self.assertNotIn("HOME", profile.env)

    def test_cycle_is_refused(self):
        self.write("base", {"extends": "child"})
        with self.assertRaises(ValueError):
            Profile.load("child", self.dir, environ={})

    def test_shipped_profiles_load(self):
        for path in Path("config/profiles").glob("*.json"):
            Profile.load(path.stem, environ={})


if __name__ == "__main__":
    unittest.main()
