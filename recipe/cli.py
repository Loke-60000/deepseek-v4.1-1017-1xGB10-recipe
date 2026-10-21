from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from recipe.ablate.job import DirectionJob
from recipe.assets import AssetBuilder, EngramPacker
from recipe.bench.evals import AIME2025, Benchmark, EvalRun, GPQADiamond, HumanEvalPlus, MMLUPro
from recipe.bench.golden import Golden
from recipe.bench.needle import NeedleBench
from recipe.bench.runner import WorkloadRunner
from recipe.bench.speed import PROMPTS, PrefillBench, SpeedBench
from recipe.bench.workload import Workload
from recipe.image import Images
from recipe.keeper import Keeper, Warmup
from recipe.pack.cold import GIB, ColdPlanner, ExpertSizes
from recipe.pack.graft import GraftBuilder
from recipe.probe import Probe
from recipe.profile import Profile
from recipe.server import Server
from recipe.settings import REPO_ROOT, Settings


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
    cold.add_argument("--prior", type=Path, default=REPO_ROOT / "config" / "cold" / "vllm-recipe-1380.json")
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
    golden.add_argument("--model", help="model id to ask (default: the profile's)")
    golden.add_argument("--max-tokens", type=int, default=64)
    golden.add_argument("--concurrency", type=int, default=1)
    golden.add_argument("--stagger", type=float, default=0.0, help="seconds between concurrent starts")
    speed = commands.add_parser("speed", help="decode rate, aggregate throughput and first-token time")
    speed.add_argument("--profile", default="base")
    speed.add_argument("--kinds", default=",".join(PROMPTS))
    speed.add_argument("--concurrency", default="1,2,4")
    speed.add_argument("--max-tokens", type=int, default=512)
    prefill = commands.add_parser("prefill", help="first-token time of long prompts")
    prefill.add_argument("--profile", default="base")
    prefill.add_argument("--tokens", default="8192,32768,131072")
    evals = commands.add_parser("eval", help="accuracy on GPQA Diamond, AIME 2025, MMLU-Pro and HumanEval+")
    evals.add_argument("--profile", default="base")
    evals.add_argument("--bench", default="gpqa-diamond,aime-2025,humaneval-plus,mmlu-pro")
    evals.add_argument("--effort", default="max")
    evals.add_argument("--max-tokens", type=int, default=32768)
    evals.add_argument("--concurrency", type=int, default=4)
    needle = commands.add_parser("needle", help="retrieval from a long prompt, with the lowest free memory on the way")
    needle.add_argument("--profile", default="base")
    needle.add_argument("--tokens", type=int, required=True)
    needle.add_argument("--depth", default="0.5")
    keep = commands.add_parser("keep", help="the keeper container's loop: memory guard and warm-up after each start")
    keep.add_argument("--profile", default="base")
    serve = commands.add_parser("serve", help="run the server for a profile")
    serve.add_argument("action", choices=["start", "stop", "check", "dry-run", "smoke", "warm"])
    serve.add_argument("--keep-free-gib", type=float, default=6, help="warm: free memory left untouched")
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
        probe = Probe(serve_options.port, args.model or serve_options.model_id)
        probe.wait_ready()
        answers = Golden(probe, max_tokens=args.max_tokens).record(args.concurrency, args.stagger)
        if args.action == "record":
            Golden.save(answers, args.file)
            print(f"{args.file}: {len(answers)} answers")
        else:
            for agreement in Golden.compare(Golden.load(args.file), answers):
                print(f"{agreement.kind:13s} shared prefix {agreement.shared_prefix:6.1%}")
    elif args.command == "speed":
        serve_options = Profile.load(args.profile).serve
        Probe(serve_options.port, serve_options.model_id).wait_ready()
        bench = SpeedBench(serve_options.port, serve_options.model_id, args.max_tokens)
        for level in (int(c) for c in args.concurrency.split(",")):
            for kind in args.kinds.split(","):
                print(bench.run(kind, level).line(), flush=True)
    elif args.command == "prefill":
        serve_options = Profile.load(args.profile).serve
        Probe(serve_options.port, serve_options.model_id).wait_ready()
        bench_prefill = PrefillBench(serve_options.port, serve_options.model_id)
        salt = time.time_ns() % 100_000
        for i, n in enumerate(int(t) for t in args.tokens.split(",")):
            print(bench_prefill.run(n, salt=salt + i).line(), flush=True)
    elif args.command == "eval":
        serve_options = Profile.load(args.profile).serve
        probe = Probe(serve_options.port, serve_options.model_id, timeout=7200)
        probe.wait_ready()
        out, cache = settings.paths.state / "evals", settings.paths.hf_cache
        benches: dict[str, Benchmark] = {
            "gpqa-diamond": GPQADiamond(cache),
            "aime-2025": AIME2025(cache),
            "humaneval-plus": HumanEvalPlus(cache, settings.image.toolbox_tag, out / "sandbox"),
            "mmlu-pro": MMLUPro(cache),
        }
        run = EvalRun(probe, out, args.concurrency, args.max_tokens, args.effort)
        for name in args.bench.split(","):
            print(json.dumps(run.run(benches[name])), flush=True)
    elif args.command == "needle":
        serve_options = Profile.load(args.profile).serve
        probe = Probe(serve_options.port, serve_options.model_id, timeout=7200)
        probe.wait_ready()
        for i, depth in enumerate(float(d) for d in args.depth.split(",")):
            print(NeedleBench(probe).run(args.tokens, depth, salt=time.time_ns() % 100_000 + i).line(), flush=True)
    elif args.command == "keep":
        Keeper(Server(settings, Profile.load(args.profile))).watch()
    elif args.command == "serve":
        server = Server(settings, Profile.load(args.profile))
        actions = {"start": server.start, "stop": server.stop, "check": server.preflight}
        if args.action == "dry-run":
            print(server.dry_run())
        elif args.action == "warm":
            print(Warmup(server, int(args.keep_free_gib * GIB)).run())
        elif args.action == "smoke":
            probe = Probe(server.profile.serve.port, server.profile.serve.model_id)
            print("models:", ", ".join(probe.wait_ready()))
            reply = probe.smoke()
            print(f"17 * 23 -> {reply.text.strip()!r} ({reply.completion_tokens} tokens, {reply.seconds:.1f} s)")
        else:
            actions[args.action]()


if __name__ == "__main__":
    main()
