from __future__ import annotations

import argparse
from pathlib import Path

from recipe.ablate.job import DirectionJob
from recipe.assets import AssetBuilder, EngramPacker
from recipe.bench.golden import Golden
from recipe.bench.runner import WorkloadRunner
from recipe.bench.workload import Workload
from recipe.image import Images
from recipe.pack.cold import GIB, ColdPlanner, ExpertSizes
from recipe.pack.graft import GraftBuilder
from recipe.probe import Probe
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
    cold = commands.add_parser("cold", help="plan a cold expert set from routing hits")
    cold.add_argument("--gib", type=float, required=True, help="trellis bytes to leave in mappings")
    cold.add_argument("--hits", type=Path, help="TF_DSV41_HITS output; without it, --prior's experts go first")
    cold.add_argument("--prior", type=Path, default=Path("config/cold/vllm-recipe-1380.json"))
    cold.add_argument("--out", type=Path, required=True)
    calibrate = commands.add_parser("calibrate", help="send the routing calibration workload to a running server")
    calibrate.add_argument("--profile", default="base")
    calibrate.add_argument("--per-kind", type=int, default=12)
    ablate = commands.add_parser("ablate", help="extract the derisked pack's refusal directions")
    ablate.add_argument("--rank", type=int, default=1)
    golden = commands.add_parser("golden", help="record or check greedy answers against a reference")
    golden.add_argument("action", choices=["record", "check"])
    golden.add_argument("--profile", default="base")
    golden.add_argument("--file", type=Path, required=True)
    serve = commands.add_parser("serve", help="run the server for a profile")
    serve.add_argument("action", choices=["start", "stop", "check", "dry-run", "smoke"])
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
    elif args.command == "cold":
        sizes = ExpertSizes.of_pack(settings.paths.pack)
        if args.hits:
            hits = ColdPlanner.read_hits(args.hits)
        else:
            prior = ColdPlanner.read_set(args.prior)
            hits = {expert: int(expert not in prior) for expert in sizes.sizes}
        planner = ColdPlanner(sizes, hits)
        plan = planner.plan(int(args.gib * GIB))
        plan.write(args.out)
        share = f", {planner.share_of_picks(plan):.2%} of measured picks" if args.hits else ""
        print(f"{args.out}: {len(plan.experts)} experts, {plan.bytes / GIB:.2f} GiB of {sizes.total / GIB:.2f}{share}")
    elif args.command == "calibrate":
        serve_options = Profile.load(args.profile).serve
        requests = list(Workload().requests(args.per_kind))
        print(Workload.describe(requests))
        outcomes = WorkloadRunner(Probe(serve_options.port, serve_options.model_id)).run(requests)
        print(WorkloadRunner.summary(outcomes))
    elif args.command == "ablate":
        print(DirectionJob(settings, rank=args.rank).build())
    elif args.command == "golden":
        serve_options = Profile.load(args.profile).serve
        probe = Probe(serve_options.port, serve_options.model_id)
        probe.wait_ready()
        answers = Golden(probe).record()
        if args.action == "record":
            Golden.save(answers, args.file)
            print(f"{args.file}: {len(answers)} answers")
        else:
            for agreement in Golden.compare(Golden.load(args.file), answers):
                print(f"{agreement.kind:13s} shared prefix {agreement.shared_prefix:6.1%}")
    elif args.command == "serve":
        server = Server(settings, Profile.load(args.profile))
        actions = {"start": server.start, "stop": server.stop, "check": server.preflight}
        if args.action == "dry-run":
            print(server.dry_run())
        elif args.action == "smoke":
            probe = Probe(server.profile.serve.port, server.profile.serve.model_id)
            print("models:", ", ".join(probe.wait_ready()))
            reply = probe.smoke()
            print(f"17 * 23 -> {reply.text.strip()!r} ({reply.completion_tokens} tokens, {reply.seconds:.1f} s)")
        else:
            actions[args.action]()


if __name__ == "__main__":
    main()
