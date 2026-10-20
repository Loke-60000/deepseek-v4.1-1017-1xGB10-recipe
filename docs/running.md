# Running it

The engine and the recipe's tools ship as two images, built by GitHub Actions on every push to `main` and on every
release tag:

| Image | Holds |
|---|---|
| `ghcr.io/loke-60000/dsv41-1xgb10` | the patched TensorFold engine that serves the model |
| `ghcr.io/loke-60000/dsv41-1xgb10-toolbox` | the recipe's CLI, its configs and the build tools |

Tags: `latest` and `vX.Y.Z` for releases, `main` for the newest commit. A toolbox image always starts the engine image
of its own tag. You do not need to clone this repo or build anything.

## Install the launcher

```bash
mkdir -p ~/.local/bin ~/.config
docker run --rm --entrypoint cat ghcr.io/loke-60000/dsv41-1xgb10-toolbox:latest /recipe/bin/dsv41 > ~/.local/bin/dsv41
chmod +x ~/.local/bin/dsv41
```

`dsv41` runs the CLI inside the toolbox image. It talks to the host's Docker and mounts the model folder and the
Hugging Face cache at their own paths, so everything it starts sees the same files.

Settings live in `~/.config/dsv41.env`, with the keys from [.env.example](../.env.example). To pin a release, set
`DSV41_TOOLBOX=ghcr.io/loke-60000/dsv41-1xgb10-toolbox:<tag>` there.

## First start

```bash
dsv41 graft                          # build the pack from the two Hub repos
dsv41 engram                         # pack the Engram tables
dsv41 assets --profile serve         # the profile's kernels, RoPE and Engram tables
dsv41 serve start --profile serve
dsv41 serve smoke --profile serve
```

| Step | Does |
|---|---|
| `graft` | assembles the pack: SAGE routed experts and flash17 for the rest, hardlinked from the Hub cache |
| `engram` | packs the 189 GiB Engram tables for a single GPU |
| `assets` | compiles the profile's kernels and builds its RoPE and Engram tables |
| `serve start` | checks the assets, then starts the `dsv41` server and the `dsv41-keeper` containers |
| `serve smoke` | asks one question and checks the answer |

Budget about 650 GB of disk: the SAGE download (~305 GB), the Engram tables (189 GB), the engine's prepared files,
which the first boot writes (~111 GB), and the dense part of the pack. Later boots read the prepared files back.

The API listens on `http://<host>:8888/v1`. `dsv41 serve stop` lets open requests finish, then stops both containers.

## The keeper

`dsv41-keeper` runs next to the server, and Docker restarts both after a reboot. It:

- stops the server if available memory falls under 2.5 GiB, because on unified memory the whole box freezes before
  the OOM killer reacts;
- warms the server each time it starts: it reads the most-used cold experts into memory and sends a few requests.
  Without that, the first replies run at 8-17 tok/s while experts load from disk.

`docker logs dsv41-keeper` shows what it did. `dsv41 serve warm` runs the same warm-up by hand.

## Configuration

Settings come from `config/recipe.json` and `config/profiles/*.json` inside the image. Values in
`~/.config/dsv41.env` or the environment win: `DSV41_SERVE_*` for the API, any `TF_DSV41_*` for the engine.

| Profile | For |
|---|---|
| `serve` | production: 4 slots x 256K, vision, prompt caching, 45 GiB of experts on disk |
| `serve8` | 8 slots in the same KV pool: more total throughput, slower per stream |
| `calibrate` | counts which experts get routed, to choose what stays on disk |
| `native`, `bringup` | small single-slot boots for testing |

To log served tokens for training a draft model later, set `TF_DSV41_TRACE_TOKENS=/state/tokens.jsonl`.

## Derisked model id

```bash
dsv41 ablate                         # once: extracts the edit from the DERISKED pack
```

Then set `DSV41_SERVE_ABLATE_MODEL=<id>` and run `serve start` again. Requests for that id get the DERISKED pack's
edit, applied at run time to the same weights; other requests in the same batch are not affected.

## Choosing the experts kept on disk

The shipped `measured-45g.json` comes from routing counts over code, prose, math, tool calls and six languages.
Cold sets live in `<model root>/cold`, where `serve start` copies the shipped ones. To redo it for your own traffic:

```bash
dsv41 serve start --profile calibrate
dsv41 calibrate --profile calibrate
dsv41 serve stop
dsv41 cold --gib 45 --hits <model root>/state/hits.json --out <model root>/cold/mine-45g.json
```

Then set `TF_DSV41_COLD=/cold/mine-45g.json`. 45 GiB was the best size on one Spark: smaller sets read more from disk
at 4 streams, larger ones leave the page cache too little room.

## Measuring

| Command | Measures |
|---|---|
| `dsv41 speed --concurrency 1,4` | decode speed per stream, total throughput and time to first token |
| `dsv41 prefill --tokens 28000` | prompt processing speed on a fresh prompt |
| `dsv41 golden record` / `check` | saves greedy answers, then checks a later build gives the same ones |

## Working from a clone

To change the recipe or the engine patches, clone the repo and use `python3 -m recipe.cli` in place of `dsv41`, with
`.env` in the repo root for settings. `python3 -m recipe.cli image` builds both images locally under the same names;
the keeper always runs from the toolbox image, so rebuild it after changing the Python code.
