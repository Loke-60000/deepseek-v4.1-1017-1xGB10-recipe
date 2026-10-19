from __future__ import annotations

import argparse

from recipe.assets import AssetBuilder, EngramPacker
from recipe.image import Images
from recipe.pack.graft import GraftBuilder
from recipe.profile import Profile
from recipe.server import Server
from recipe.settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser(prog="dsv41")
    commands = parser.add_subparsers(dest="command", required=True)
    graft = commands.add_parser("graft", help="build the pack folder from the experts and dense packs")
    graft.add_argument("--workers", type=int, default=8)
    image = commands.add_parser("image", help="build the engine and toolbox images")
    image.add_argument("which", choices=["engine", "toolbox", "all"], nargs="?", default="all")
    assets = commands.add_parser("assets", help="build a profile's Triton, RoPE and Engram host assets")
    assets.add_argument("--profile", default="base")
    commands.add_parser("engram", help="pack the Engram tables for one rank")
    serve = commands.add_parser("serve", help="run the server for a profile")
    serve.add_argument("action", choices=["start", "stop", "check", "dry-run"])
    serve.add_argument("--profile", default="base")
    args = parser.parse_args()

    settings = Settings.load()
    if args.command == "graft":
        print(GraftBuilder(settings, workers=args.workers).build())
    elif args.command == "image":
        images = Images(settings.image)
        if args.which in ("engine", "all"):
            print(images.engine())
        if args.which in ("toolbox", "all"):
            print(images.toolbox())
    elif args.command == "assets":
        print(AssetBuilder(settings, Profile.load(args.profile)).build())
    elif args.command == "engram":
        print(EngramPacker(settings).build())
    elif args.command == "serve":
        server = Server(settings, Profile.load(args.profile))
        actions = {"start": server.start, "stop": server.stop, "check": server.preflight}
        if args.action == "dry-run":
            print(server.dry_run())
        else:
            actions[args.action]()


if __name__ == "__main__":
    main()
