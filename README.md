<p align="center"><img src="docs/assets/banner.png" alt="DeepSeek V4.1-1017 on 1x DGX Spark" width="100%"></p>

# DeepSeek-V4.1-Flash-1017 on one DGX Spark

Runs DeepSeek-V4.1-Flash-1017 on a single DGX Spark with [TensorFold](https://github.com/ashhart/TensorFold)'s
DeepSeek engine, behind an OpenAI-compatible API: four 256K-context slots, image input, prompt caching, and an
optional second model id that answers as the derisked variant.

<p align="center"><img src="docs/assets/overview.png" alt="Unified memory holds 78 GiB of weights and buffers plus a 36 GiB page cache; the SSD holds 45 GiB of cold experts and the 189 GiB Engram tables. 8 % of routed picks go to the 43 % of experts on SSD. 92-97 tok/s at 4 streams, 29-30 at one, 0.3-0.4 s to the first token." width="100%"></p>

The model is bigger than the Spark's memory, so it is assembled from two quantized packs and the rarely used experts
stay on the SSD:

- routed experts from [vcruz305's SAGE EXL3 1.62 bpw](https://huggingface.co/vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw),
  everything else from [jayleaton's flash17 EXL3 2.8 bpw](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw);
- the least-routed experts are read by the GPU straight from the engine's files on disk, and the page cache keeps the
  ones in use;
- the 189 GiB Engram tables are read from the SSD a row at a time.

## Speed

Greedy decoding, 512 tokens per request, warm cache:

| Streams | Per stream | Total | First token |
|---:|---:|---:|---:|
| 1 | 29-30 tok/s | 29-30 tok/s | 0.3 s |
| 4 | 24-25 tok/s | 92-97 tok/s | 0.5-1.0 s |
| 8 (`serve8`) | 14-19 tok/s | 100-136 tok/s | 1.6-3.1 s |

New prompts of 25-37K tokens prefill at 850-1,370 tok/s.

Known limits:

- A long *new* prompt runs every expert and pushes the cached ones out, so the next few replies drop to 10-19 tok/s
  until the cache settles. Follow-up turns reuse the cached prompt and are not affected.
- DSpark speculative decoding is off: on this memory budget it reads far more experts from disk and halves 4-stream
  throughput.
- Each token runs the fewest of its 6 routed experts that carry 85 % of the router weight
  (`TF_DSV41_EXPERT_TOPP=0.85`). Running all 6 is a third slower here.

## Setup

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

The API listens on `http://<host>:8888/v1`; `serve stop` drains open requests first.

Run `serve warm` after every start. It reads the most-used cold experts into memory and sends a few warm-up
requests; without it, the first replies run at 8-17 tok/s while experts load from disk.

`python3 -m recipe.cli units install` sets up two systemd user units (requires `loginctl enable-linger`):

- `dsv41-guard` stops the server if available memory falls under 2.5 GiB. On unified memory the whole box freezes
  before the OOM killer reacts.
- `dsv41-warm` runs `serve warm` after each boot.

## Configuration

Settings come from `config/recipe.json` and `config/profiles/*.json`. Values in `.env` or the environment win:
`DSV41_SERVE_*` for the API, any `TF_DSV41_*` for the engine.

| Profile | For |
|---|---|
| `serve` | production: 4 slots x 256K, vision, prompt caching, 45 GiB of experts on disk |
| `serve8` | 8 slots in the same KV pool: more total throughput, slower per stream |
| `calibrate` | counts which experts get routed, to choose what stays on disk |
| `native`, `bringup` | small single-slot boots for testing |

To re-pick the experts that stay on disk: run the `calibrate` profile, send `dsv41 calibrate`, then
`dsv41 cold --gib 45 --hits <hits file> --out config/cold/<name>.json`.

## Derisked model id

Set `DSV41_SERVE_ABLATE_MODEL=<id>` and run `dsv41 ablate` once. Requests for that id get the edit from
[vcruz305's DERISKED pack](https://huggingface.co/vcruz305/DSV4.1-Flash-1017-DERISKED-EXL3-1.61bpw), applied at run
time to the same weights. Other requests in the same batch are not affected.

## Engine patches

`patches/` applies to jayleaton's kit at the commit pinned in `config/recipe.json`.

| Patch | Change |
|---|---|
| 0001 | AOT kernel list for a single GPU |
| 0002 | cold experts read from the prepared files on disk |
| 0003 | FP8 tensors in a grafted pack loaded as bytes |
| 0004 | per-expert routing counters (`TF_DSV41_HITS`) |
| 0005 | 1-bit experts in the prefill kernels |
| 0006 | the derisked second model id |
| 0007 | token log for training a draft model (`TF_DSV41_TRACE_TOKENS`) |
| 0008 | sampling on the GPU for a single rank |
| 0009 | fixes prompts of 7.2K-8.2K tokens failing with `Unbound` |

## Credits

DeepSeek for the model. ashhart for [TensorFold](https://github.com/ashhart/TensorFold). jayleaton for the
[DeepSeek-V4.1 TensorFold kit](https://github.com/jayleaton/deepseek-v41-tensorfold-spark) and the flash17 pack.
vcruz305 for the SAGE and DERISKED packs, the cold-expert approach, and the starting expert list in
`config/cold/vllm-recipe-1380.json`, taken from vcruz305's
[Spark recipe](https://github.com/vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe) (MIT). turboderp for EXL3.
