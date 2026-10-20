import tempfile
import unittest
from pathlib import Path
from unittest import mock

from recipe.guard import UNITS, MemoryGuard, Units


class GuardTest(unittest.TestCase):
    def test_kills_only_below_the_floor_and_when_running(self):
        guard = MemoryGuard("dsv41", floor_bytes=1000)
        with (
            mock.patch.object(MemoryGuard, "available", side_effect=[2000, 500, KeyboardInterrupt]),
            mock.patch.object(MemoryGuard, "running", return_value=True),
            mock.patch("recipe.guard.subprocess.run") as run,
            mock.patch("recipe.guard.time.sleep"),
            self.assertRaises(KeyboardInterrupt),
        ):
            guard.watch()
        run.assert_called_once_with(["docker", "kill", "dsv41"], check=False)

    def test_units_render_with_this_machine(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch("recipe.guard.subprocess.run"):
            names = Units("/usr/bin/python3", "serve", 2.5, Path(tmp)).install()
            texts = {name: (Path(tmp) / name).read_text() for name in names}
        self.assertEqual(set(names), set(UNITS))
        self.assertIn("-m recipe.cli guard --floor-gib 2.5", texts["dsv41-guard.service"])
        self.assertIn("serve warm --profile serve", texts["dsv41-warm.service"])


if __name__ == "__main__":
    unittest.main()
