# Deploying this model: notes for agents

You are deploying DeepSeek-V4.1-Flash-1017 behind an OpenAI-compatible API on one NVIDIA DGX Spark. Use the
published images. Do not clone the repo or build images unless you are changing the recipe itself.

## Check the machine first

Stop and report if any of these fails:

| Check | Expect |
|---|---|
| `uname -m` | `aarch64` |
| `nvidia-smi --query-gpu=name --format=csv,noheader` | `NVIDIA GB10` |
| `grep MemTotal /proc/meminfo` | about 128 GB |
| `docker info --format '{{.Runtimes}}'` | includes `nvidia` |
| `df -h` on the model folder's disk | about 650 GB free |
| `ss -ltn` | nothing on port 8888 |
| other GPU services | none running; the model needs nearly all of the 128 GB |

If the images are private, `docker login ghcr.io` with a token that can read packages.

## Install the launcher

```bash
mkdir -p ~/.local/bin ~/.config
docker run --rm --entrypoint cat ghcr.io/loke-60000/dsv41-1xgb10-toolbox:latest /recipe/bin/dsv41 > ~/.local/bin/dsv41
chmod +x ~/.local/bin/dsv41
```

`dsv41` runs the recipe's CLI from the toolbox image and drives Docker on the host. Settings go in
`~/.config/dsv41.env` (same keys as `.env.example` in the repo). Set at least `DSV41_PATHS_ROOT` if the default
`~/models/dsv41-1xgb10` is not on the big disk, and `DSV41_SERVE_MODEL_ID` if clients expect another model id.
Pin a release with `DSV41_TOOLBOX=ghcr.io/loke-60000/dsv41-1xgb10-toolbox:<tag>`; that toolbox uses the engine image
of the same tag.

## Deploy

Run these in order. The first three are long (downloads of about 300 GB and hours of packing); run them in the
background, write their output to a log, and poll the log instead of waiting on the command.

| Step | Command | Done when |
|---|---|---|
| 1 | `dsv41 graft` | prints the pack folder |
| 2 | `dsv41 engram` | prints the Engram folder, after its own check |
| 3 | `dsv41 assets --profile serve` | prints the assets folder |
| 4 | `dsv41 serve start --profile serve` | returns; containers `dsv41` and `dsv41-keeper` are up |
| 5 | wait for `docker logs dsv41` to show `serving:` | about 10 minutes; the first boot writes ~111 GB of prepared files |
| 6 | wait for `docker logs dsv41-keeper` to show `warmed:` | a few minutes after step 5 |
| 7 | `dsv41 serve smoke --profile serve` | the answer contains `391` |

`HF_TOKEN` in the environment is passed through if the Hub asks for one.

The keeper container runs beside the server and Docker restarts both after a reboot. It stops the server if available
memory drops under 2.5 GiB (on unified memory the box freezes before the OOM killer acts), and it warms the server
every time it starts. Leave it running.

## Verify

- `curl -s localhost:8888/v1/models` lists the model id.
- `dsv41 speed --profile serve --concurrency 1,4` gives about 29-30 tok/s at 1 stream and 90-97 tok/s in total at 4,
  once warm. The first replies after a start or after a long new prompt are slower; that is expected.

## Do not

- Run other GPU workloads on the box, or raise `TF_DSV41_SLOTS` / `TF_DSV41_CONTEXT` without checking memory.
- Enable DSpark drafts (`TF_DSV41_DRAFTS`) or unset `TF_DSV41_EXPERT_TOPP`: both are slower on this memory budget.
- Delete `prepared/` or `engram/` under the model root; they take hours to rebuild.
- Stop the server with `docker stop`/`docker kill`; use `dsv41 serve stop`, which drains requests and removes the
  keeper.

## When something fails

| Symptom | Cause and fix |
|---|---|
| `dsv41 serve start` says assets are missing | run step 3 for that profile |
| boot ends with `PoolDoesNotFit` or `failed in boot` | not enough free memory: stop other GPU services |
| `dsv41` exits with a Docker permission error | the user is not in the `docker` group |
| the server stopped and the keeper logged `killing dsv41` | memory ran out; find what else used it, then `serve start` again |
| replies at 8-17 tok/s right after a start | the warm-up has not finished; check the keeper's log |

More detail for humans: [docs/running.md](docs/running.md).
