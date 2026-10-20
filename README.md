<p align="center"><img src="docs/assets/banner.png" alt="DeepSeek V4.1-1017 on 1x DGX Spark" width="100%"></p>

# DeepSeek-V4.1-Flash-1017 on one DGX Spark

DeepSeek-V4.1-Flash-1017 (552B MoE, Engram, DSpark drafts, vision) served from a single NVIDIA DGX Spark (GB10,
128 GB unified memory) with [TensorFold](https://github.com/ashhart/TensorFold)'s DeepSeek-V4.1 engine, through an
OpenAI-compatible API. Four slots of 256K context, image input, prompt sessions, and an optional second model id that
serves a derisked (abliterated) variant from the same weights.

## How it fits

The model does not fit in 128 GB, so the recipe combines three things:

| Piece | What it is |
|---|---|
| Routed experts | [vcruz305's SAGE EXL3 1.62 bpw](https://huggingface.co/vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw), DSpark draft experts too |
| Everything else | [jayleaton's flash17 EXL3 2.8 bpw](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw) dense tensors, and the 820 experts whose SAGE gate/up widths differ |
| Engram tables | 189 GiB of FP8 n-gram memory, packed for one rank, read from NVMe row by row |
| Cold experts | the least-routed experts stay in the engine's prepared files and the GPU reads them through a read-only mapping (GB10's ATS); the page cache keeps the ones in use |

`dsv41 graft` builds the pack folder from the two Hub repos (hardlinks plus ~18 GB of range reads), `dsv41 engram`
packs the Engram shards, and the engine image is TensorFold's DeepSeek-V4.1 kit with this repo's `patches/`.

## Measured (one GB10, `serve` profile)

Greedy, 512 tokens a request, warm page cache, mixed kinds back to back (`dsv41 speed`); prompts of 28K tokens
prefill at ~960 tok/s warm (`dsv41 prefill`):

| Streams | Prose per stream | Code per stream | Structured per stream | Aggregate | First token |
|---:|---:|---:|---:|---:|---:|
| 1 | 26.7 tok/s | 27.0 tok/s | 28.0 tok/s | 26-28 tok/s | 0.3-0.4 s |
| 4 | 24.0 | 24.8 | 24.6 | 92-97 | 0.5-1.0 s |
| 8 (`serve8`) | 14 | 18 | 19 | 100-136 | 1.6-3.1 s |

Native prefill matches the decode-window path answer for answer on the golden set (`dsv41 golden`), and normal
requests are byte-identical with the derisked id loaded and running in the same batches.

## What decides the speed

- **The working set must stay in RAM.** Weights plus buffers need ~123 GiB of a box that leaves ~100 for the model.
  The experts the page cache holds cost nothing; the ones it misses are faulted in from NVMe by the GPU, slowly. The
  routing is skewed (the coldest 30 % of experts take 2.8 % of picks), so the cold set is chosen from measured hits,
  and it is larger than the shortfall: 45 GiB mapped lets the page cache follow the traffic (0.5 GB of NVMe reads a
  sweep, against 8 GB at 25 GiB).
- **DSpark is off.** Verifying drafted rows touches several times more distinct experts a step; on this memory budget
  that multiplied NVMe reads ~8x and cut 4-stream throughput from ~90 to ~50 tok/s. With the cache warm on one kind
  of text it reaches 35-45 tok/s single-stream on code and JSON, so it is worth revisiting with more RAM.
- **Memory levers already used:** DSpark experts from the SAGE pack (2.8 GiB instead of 6.3), `vm.min_free_kbytes`
  1 GiB instead of 4, the other GPU services on the box stopped.
- **Expert top-p stays at 0.85.** Each token runs the fewest of its 6 routed experts that carry 85 % of the router
  weight. Exact top-6 (`TF_DSV41_EXPERT_TOPP` unset, as DeepSeek serves it) read 24 % more from NVMe and cut
  4-stream throughput by a third.
- **Tried and dropped:** NVMe read-ahead (GPU faults do not take that path) and `MADV_HUGEPAGE` on the mappings (2 MiB
  faults pull neighbouring experts in and cost 25-45 % at 4 streams).

## Quick start

```bash
cp .env.example .env                 # paths, served model id, aliases, derisked id
python3 -m recipe.cli image          # engine + toolbox images (patches applied)
python3 -m recipe.cli graft          # the pack folder
python3 -m recipe.cli engram         # Engram shards for one rank
python3 -m recipe.cli assets --profile serve
python3 -m recipe.cli serve start --profile serve
python3 -m recipe.cli serve warm --profile serve    # after every start: see below
python3 -m recipe.cli serve smoke --profile serve
```

The API is at `http://<host>:8888/v1`. `dsv41 serve stop` drains requests first.

`serve warm` reads the most-routed cold experts into the page cache at disk speed, then sends a short reply of each
kind and one long prompt. A fresh start without it serves the first long prompt at ~90 tok/s and the first replies
at 8-17 tok/s while experts fault in; with it, ~440 tok/s, and full speed after the first few requests.

`python3 -m recipe.cli units install` adds two systemd user units (needs `loginctl enable-linger`):
`dsv41-guard` kills the container when MemAvailable drops under `--floor-gib` (2.5 by default), because unified
memory locks the whole box up before the OOM killer acts, and `dsv41-warm` runs `serve warm` after every boot.

## Profiles

`config/profiles/*.json`, layered with `extends`; any `TF_DSV41_*` variable in the environment or `.env` overrides
the profile, and `DSV41_SERVE_*` overrides the serve options.

| Profile | Use |
|---|---|
| `serve` | production: 4 slots x 256K, vision, 4K native prefill, sessions, measured 45 GiB cold set, drafts off |
| `serve8` | `serve` with 8 slots sharing the same 1M-token pool: more aggregate, slower per stream |
| `native` | native prefill check, 1 slot, no drafts |
| `calibrate` | decode-window prefill and routing hit counters, for choosing the cold set |
| `bringup` | the smallest boot |

## Derisked model id

`DSV41_SERVE_ABLATE_MODEL=<id>` serves a second model id from the same weights: requests naming it get the
[DERISKED](https://huggingface.co/vcruz305/DSV4.1-Flash-1017-DERISKED-EXL3-1.61bpw) pack's edit applied at run time.
`dsv41 ablate` recovers that edit from the derisked pack's FP8 attention writers: one direction per layer, layers
14-39, strength about 3.3, applied as `y -= Q (A (Q^T y))` to the attention output of the masked rows only. Other
requests in the same batch are untouched, bit for bit.

## Tools

| Command | Does |
|---|---|
| `dsv41 calibrate` | sends a mixed workload (code, prose, math, tool calls, six languages) for routing statistics |
| `dsv41 cold --gib N --hits FILE` | the cold set: fewest routed picks per byte first |
| `dsv41 golden record/check` | greedy answers under one profile, compared under another |
| `dsv41 speed` | decode rate, aggregate throughput and first-token time under concurrency |

## Engine patches

| Patch | Change |
|---|---|
| 0001 | single-rank output groups in the AOT enumeration |
| 0002 | cold experts read through prepared-file mappings |
| 0003 | FP8 safetensors entries read as bytes (a grafted pack's unused native tensors) |
| 0004 | routing hit counters (`TF_DSV41_HITS`) |
| 0005 | 1-bit experts in the x3gm prefill kernels |
| 0006 | run-time ablation for a second model id |
| 0007 | appending token log for drafter distillation; prefill calls under `TF_DSV41_SYNC_EACH` |
| 0008 | single-rank sampling on the device, in vocabulary slices |
| 0009 | the 4K prefill workspace holds every index row block's position: prompts of 7.2K-8.2K tokens failed with `Unbound` |

## Credits

DeepSeek for the model; ashhart's [TensorFold](https://github.com/ashhart/TensorFold) and jayleaton's
[DeepSeek-V4.1 TensorFold kit](https://github.com/jayleaton/deepseek-v41-tensorfold-spark) for the engine and the
flash17 pack; vcruz305 for the SAGE and DERISKED packs and the cold-expert idea; turboderp for EXL3.
