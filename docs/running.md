# Running it

Everything goes through `python3 -m recipe.cli` (`dsv41` once the package is installed). Run the commands from the
repo root on the Spark; the engine and the build tools run in Docker.

## First start

```bash
cp .env.example .env                     # paths, model id, aliases, derisked id
python3 -m recipe.cli image              # engine and toolbox images, patches applied
python3 -m recipe.cli graft              # build the pack from the two Hub repos
python3 -m recipe.cli engram             # pack the Engram tables
python3 -m recipe.cli assets --profile serve
python3 -m recipe.cli serve start --profile serve
python3 -m recipe.cli serve warm --profile serve
python3 -m recipe.cli serve smoke --profile serve
```

| Step | Does |
|---|---|
| `image` | builds the engine from the pinned kit plus `patches/`, and the toolbox used by the other steps |
| `graft` | assembles the pack: SAGE routed experts and flash17 for the rest, hardlinked from the Hub cache |
| `engram` | packs the 189 GiB Engram tables for a single GPU |
| `assets` | compiles the profile's kernels and builds its RoPE and Engram tables |
| `serve start` | checks the assets, then starts the `dsv41` container (restarts with Docker) |
| `serve warm` | reads the most-used cold experts into memory, then sends a few warm-up requests |
| `serve smoke` | asks one question and checks the answer |

The first boot also writes the engine's prepared files (about 111 GiB); later boots read them back.

The API listens on `http://<host>:8888/v1`. `serve stop` lets open requests finish first.

## After every start

Run `serve warm`. Without it, the first replies run at 8-17 tok/s while experts load from disk.

To have this happen on its own, install the systemd user units once (needs `loginctl enable-linger`):

```bash
python3 -m recipe.cli units install      # --profile serve --floor-gib 2.5 by default
```

- `dsv41-guard` stops the server if available memory falls under the floor. On unified memory the whole box freezes
  before the OOM killer reacts.
- `dsv41-warm` runs `serve warm` after each boot.

## Configuration

Settings come from `config/recipe.json` and `config/profiles/*.json`. Profiles build on each other with `extends`.
Values in `.env` or the environment win: `DSV41_SERVE_*` for the API, any `TF_DSV41_*` for the engine.

| Profile | For |
|---|---|
| `serve` | production: 4 slots x 256K, vision, prompt caching, 45 GiB of experts on disk |
| `serve8` | 8 slots in the same KV pool: more total throughput, slower per stream |
| `calibrate` | counts which experts get routed, to choose what stays on disk |
| `native`, `bringup` | small single-slot boots for testing |

To log served tokens for training a draft model later, set `TF_DSV41_TRACE_TOKENS=/state/tokens.jsonl` in `.env`.

## Derisked model id

```bash
python3 -m recipe.cli ablate             # once: extracts the edit from the DERISKED pack
```

Then set `DSV41_SERVE_ABLATE_MODEL=<id>` in `.env` and restart. Requests for that id get the DERISKED pack's edit,
applied at run time to the same weights; other requests in the same batch are not affected.

## Choosing the experts kept on disk

`config/cold/measured-45g.json` comes from routing counts over code, prose, math, tool calls and six languages. To
redo it for your own traffic:

```bash
python3 -m recipe.cli serve start --profile calibrate
python3 -m recipe.cli calibrate --profile calibrate
python3 -m recipe.cli serve stop
python3 -m recipe.cli cold --gib 45 --hits <state dir>/hits.json --out config/cold/mine-45g.json
```

Point `TF_DSV41_COLD` at `/cold/mine-45g.json`. 45 GiB was the best size on one Spark: smaller sets read more from
disk at 4 streams, larger ones leave the page cache too little room.

## Measuring

| Command | Measures |
|---|---|
| `speed --concurrency 1,4` | decode speed per stream, total throughput and time to first token |
| `prefill --tokens 28000` | prompt processing speed on a fresh prompt |
| `golden record` / `golden check` | saves greedy answers, then checks a later build gives the same ones |
