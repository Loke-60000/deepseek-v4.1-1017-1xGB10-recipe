import tempfile
import unittest
from pathlib import Path
from unittest import mock

from recipe.keeper import Keeper
from recipe.profile import Profile
from recipe.server import SHIPPED_COLD, Server
from recipe.settings import Settings


class KeeperTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        env = {"DSV41_PATHS_ROOT": str(self.root), "DSV41_PATHS_HF_CACHE": str(self.root / "hub")}
        self.server = Server(Settings.load(env=env), Profile.load("serve", environ={}))

    def tearDown(self):
        self.tmp.cleanup()

    def step(self, keeper, started, available):
        with (
            mock.patch.object(Keeper, "started", return_value=started),
            mock.patch.object(Keeper, "available", return_value=available),
            mock.patch("recipe.keeper.subprocess.run") as run,
            mock.patch.object(Keeper, "_warm") as warm,
        ):
            keeper.step()
            if keeper.warming:
                keeper.warming.join()
        return run, warm

    def test_kills_the_server_under_the_floor(self):
        run, warm = self.step(Keeper(self.server, floor=1000), "t1", 500)
        run.assert_called_once_with(["docker", "kill", "dsv41"], check=False)
        warm.assert_not_called()

    def test_warms_once_per_start_and_leaves_a_stopped_server_alone(self):
        keeper = Keeper(self.server, floor=1000)
        self.assertEqual(self.step(keeper, None, 500)[1].call_count, 0)
        self.assertEqual(self.step(keeper, "t1", 5000)[1].call_count, 1)
        self.assertEqual(self.step(keeper, "t1", 5000)[1].call_count, 0)
        self.assertEqual(self.step(keeper, "t2", 5000)[1].call_count, 1)

    def test_shipped_cold_sets_are_staged_without_touching_local_ones(self):
        mine = self.root / "cold" / "mine.json"
        mine.parent.mkdir()
        mine.write_text("{}")
        self.server.stage_cold()
        for shipped in SHIPPED_COLD.glob("*.json"):
            self.assertEqual((self.root / "cold" / shipped.name).read_bytes(), shipped.read_bytes())
        self.assertEqual(mine.read_text(), "{}")

    def test_keeper_container_sees_the_host_paths(self):
        socket = self.root / "docker.sock"
        socket.touch()
        with mock.patch("recipe.server.DOCKER_SOCKET", socket):
            argv = self.server._keeper().argv()
        self.assertIn(f"{self.root}:{self.root}", argv)
        self.assertIn(f"DSV41_PATHS_ROOT={self.root}", argv)
        self.assertIn(f"{socket}:{socket}", argv)
        self.assertEqual(argv[argv.index("--group-add") + 1], str(socket.stat().st_gid))
        self.assertEqual(argv[-4:], ["dsv41", "keep", "--profile", "serve"])


if __name__ == "__main__":
    unittest.main()
