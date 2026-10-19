# TensorFold dsv41 (jayleaton 2x Spark TP2) digest for a 1x Spark port

Sources (B = https://github.com/jayleaton/deepseek-v41-tensorfold-spark/blob/main/):
[FL] B+docs/FLASH-1017.md  [OP] B+docs/OPERATIONS.md  [ZS] B+docs/ZIG-SERVE.md  [ZR] B+docs/ZIG-RESULTS.md
[KI] B+docs/KNOWN-ISSUES.md  [DE] B+docs/DECODE.md  [AR] B+docs/ARCHITECTURE.md  [RS] B+docs/RESULTS.md
[RST] B+docs/README-STATS.md  [ENV] B+config/prod-zig-flash17.env.example  [LA] B+docs/campaign/LANDSCAPE.md
[AL] B+docs/campaign/ARCH-LEVERAGE.md  [DA] B+docs/campaign/DRAFT-ACCEPTANCE.md  [DR] B+docs/campaign/DECODE-ROOFLINE.md
[PS] B+docs/campaign/PARKED-SAGE-1.59.md  [EP] B+docs/campaign/ENGINE-PLAN.md  [G17L] B+docs/campaign/G17-LEVERS.md
[G16] B+docs/campaign/G16-RELEASE.md  [ER] B+engine/zig/zig/src/families/deepseek_v41/engram_rows.zig
[KVL] B+engine/zig/zig/src/families/deepseek_v41/kv/layout.zig

Caveat: G1-G19 campaign numbers are the older Python engine; Zig numbers are in [ZR]/[FL]. Everything is TP2.

## Levers and measured effects
- `TF_DSV41_POOL_TOKENS` (shared paged pool, not per slot; unset = 4 x CONTEXT). flash17 head low point: 3.0M 3.56-3.98 GiB (under the 4 GiB floor), 2.4M 3.96, 2.2M 4.15, **2.1M 4.23 (recommended)** [FL]. The boot check refuses a pool whose worst dip (`TF_DSV41_SERVE_DIP_GIB`=3.0) would breach the hard floor [ZS].
- Floors: `FLOOR_GIB`=5 target, `FLOOR_HARD_GIB`=4 admission; MemFree >=104 GiB gate before load; earlyoom `-m 2,1` is the backstop [OP].
- Chunk: profile uses 4K (`PF_4K=1`, `_MEMORY_OK=1`, `PF_WS_FLOOR_GIB=3.5`). Each long prompt allocates a transient **2,268 MiB** workspace per rank, which sets the memory low point (not the weights) [FL]. Falls back to 2K if memory is short. Python: 2,048 adaptive rows gave replay 32K 1,996 -> 2,310 tok/s; `PREFILL_ADAPT_GIB`=4.5 (the engine default of 5 flaps); rows step 2048/1024/512 [OP]. Do not combine PF_TBO with 4K: it ran out of memory [ZS].
- `INDEX_BUDGET_MIB` 64 (vs 256) avoids +1.6 GiB of cached allocator segments in long prefills [OP][G16].
- Slots 4, `ROWS_CAP` 64, `SLOT_DRAFTS=1`. Clock cap 2100-2200 MHz is "for temperature". Unmatched: 88.84 tok/s (2250 MHz, 3.0M) vs 87.9 (2100, 2.1M) [FL].
- `EXPERT_TOPP`=0.85, min 3: lossy, x1.05 decode. `L2PF` 12 MiB: +2.5-3.5%. `MHC_FN=bf16`: prose +2.4% [DE].
- `MIMALLOC_ALLOW_THP=0` (otherwise long prefills stall in compaction). Never build CUDA extensions next to loaded weights: one build took MemAvailable to 1.26 GiB [OP].

## "Carveout"
The repo does not use one. [LA] mentions it only: "display-carve-out memory for KV or embeddings (**coolbho3k +1.8 GiB a GPU**)". coolbho3k stored **NVFP4 main+indexer KV (FP8 SWA) in 1,792 MiB of display memory** for 1,048,576 tokens/request, 3.31M aggregate, lowest floor 0.887 GiB. Reads are ~160 vs 250 GB/s. Verdict: "Consider for cold data (embedding, KV) only; requires `nvidia_drm modeset=1` (a GX10 report of a 700 MHz latch)". **The mechanism is not documented here (UNCERTAIN)**; see coolbho3k's repo.

## KV and prompt cache
- Zig rows: comp 584 B FP8 (448 e4m3 + 64 bf16 RoPE + scales), index key 132 B FP8. Sources are layers 2/8/14 (ratio 2) and 20 (ratio 1). **1,790 B/token replicated**, 1,060 B/rank split [KVL]. So **one rank: 1M tokens = ~1.67 GiB**, 2.1M = ~3.5 GiB.
- `KV_SPLIT` needs TP>=2, so it does not apply on one Spark. FP4 comp rows (384 B, about 1,290 B/token replicated by my arithmetic) exist in the layout enum but are **not wired at runtime** (kv_state passes only `split`). `INDEX_KV` is "FP8 fixed in this port". FP4 is the model's QAT format and was estimated exact if used throughout [AL].
- Per-slot rings/carries are ~3.3 MB [AR].
- Prefix/session cache: a prefix trie over chained page digests (`sessions/prefix.zig`), a RAM tier of 256 MiB, and an NVMe tier (`SESSION_DISK_GIB`=128, `_MIN`=1024 tokens). Idle sessions are parked to NVMe when the pool is needed. The NVMe resume restored 996,000 of 996,011 tokens in **1.3 s** [RST]. Hits are reported as `cached_tokens` [B+docs/BENCHMARKS.md]. CED replay makes resumed == fresh.

## Engram IO
- Shards `engram-l{1,14}-r{rank}of{world}.bin` (264 B rows, O_DIRECT, 512 B sectors, holes <=4 KiB merged). `pack_engram.py --world 1` should produce a single-rank shard. Reads are 12.7 KB/token across both modules; decode does ~150-300 reads/round/rank (TP2; double that on one rank); prefill ~49K reads per 2K chunk before dedupe [EP][AL].
- Reader: io_uring if allowed. **Docker's default seccomp blocks io_uring**, so prod uses `ENGRAM_AIO=1` (native AIO), else a thread pool `ENGRAM_QD`=64. LRU cache `TF_DSV41_ENGRAM_CACHE`=65,536 records/layer (17 MB) [ER]. Prefetch: the bonus row at verify end, draft rows at drafter end, the next prefill segment under the current one (`PREFETCH_AHEAD`, `ENGRAM_PREFETCH`); the GPU waits on an event before layers 1/14 (`ENGRAM_GATE`).
- Measured: synchronous reads cost 19% of prefill (~1.06 ms/token, G4). The native reader plus bulk prefetch brought this to 3.2 s at 32K. Decode host Engram took 5.75 ms/step before prefetch (G2). The GPU gate gave C2 +3.5%, C4 +2.6% [DE].
- Hot-row cache: designed, not built. Third parties report a 4 GiB cache at 67-76% hit (knapcio) and top 100M rows covering 92.7% (bot-lab-21) [LA][AL].

## DSpark / concurrency
- Conditional acceptance per position: prose T0 0.58/0.45; code 0.92/0.81/0.80/0.89/0.86; structured ~1.0 [DA]. On 34,753 passes: 0.506/0.415/0.359/0.349/0.385. The 2nd-choice rate s2=0.174 rules out trees, and 28% of first-position targets fall outside the 128 candidates. Tokens/round: code 3.88, prose 1.57-1.68, structured 5.91, edit/tweet ~6.2 with lookup [RS][G17L].
- k: a cost-priced depth policy from boot calibration, **capped at 5** (block = 5; others found k=10 -17 to -38%). Python verify costs were 29.5/36.3/40.25/44.2/48.15 ms for 1-5 rows, plus a ~3.5-5.9 ms draft pass.
- "2.5x multi-stream" = G10 Python C4 93.5 vs Mia kit 37.6. [RST] says not to promote it as a current Zig claim. Current Zig: code x4 156.4, prose x4 104.7 [ZR]; flash17 x4 153.76 [FL]. Python G19: C2 74.07, C4 101.09; steady C4 136, code-only 192.5. A 4-live window costs ~114 ms because ~57 distinct experts/layer are hit (U(16)=50.3) [G17L]. Slot-agnostic row graphs: C2 +34%, C4 +12%.

## Decode roofline (TP2, per rank, 2.9 bpw)
1-row floor ~3.79 GB -> ~17 ms @220 GB/s: experts 1.61 GB, attention dense 1.76 GB (K5), router 0.17, head 0.25. Each extra row adds ~5-6 new experts/layer (~3.5-5 ms). Measured 1-row window: 35 ms (G4) -> **21.8 ms** (G14). Dense EXL3 runs at 155-200 GB/s because of SM imbalance. A 1-row window at ds4-level efficiency would be ~21-22 ms [DE][DR]. Single rank: "every token would read all of the dense weights (~3.8 GB with no TP split)" [PS].

## Single-Spark-relevant statements
- [PS]: **SAGE 1.59 at TP1 "does not fit at a useful context"**: experts 101.4 GiB + non-routed 13.1 GiB (attention 3.25, embedding 1.23, shared 0.75, head 0.46, MTP 6.75) = ~114.5 GiB vs 119-121 usable. "Dropping DSpark's 6.3 GiB makes it fit, but it removes speculation". Two TP1 replicas: no.
- SAGE quality: top-1 91.2% / KL 0.189 vs FP4 (general 82%, code 97.9%). Estimated ~88-92% vs the kit, which **fails their >=96% gate**. That is why it was parked (user decision). It also needs x3ld `RANGES` (2,16) for K6, and K1/K5/mixed-width experts fall back to the slow upstream grouped kernel in prefill. Lossless alternative they suggest: SAGE experts as a draft-only model.
- On GB10 the page cache is GPU memory: drop caches before load. Engram O_DIRECT exists to avoid page-cache growth [OP][EP].
- Python budget model: graphs 1.2 + prefill 1.3 + select 0.25 + drafting 0.35 + slack 0.9 + runtime 1.0 GiB (B+engine/serving/memory.py). That is more than a 118 GiB-weights single rank has free. Expect `PF_4K=0`, 1K/512-row chunks, a small pool, and `GRAPHS_MAX` pressure (~15 MiB host memory per graph).

## Known issues / negative results
Unsolved [KI]:
- C4 mixed ~101 (<120), bounded by the prose stream.
- Prose drafting at ~1.7 tokens/round.
- Graph LRU shrinks under memory dips (48 -> 11).
- In-stream 503 when streaming.
- `/health` drafted counters are 0.
- `MOE_FUSED` is +0.7 ms (off).
- No trimmed draft vocabulary is shipped.

Negative [DE][RS]:
- Lossy verify budget: breaks tool calls (1-4/12).
- Draft trees +0.1%; PDL +-0.1 ms; x3dn -9-24% cold; joint depth C2 -2.1%; XMAP +0.22 ms; SPEC_NUCLEUS +1.3/-0.6%.
- Drafter LoRA self-distillation: prose +5.5% but code -4.4%.

Avoid (third-party evidence) [LA]: NVFP4 experts ("no bandwidth saved on GB10"), REAP/top-k cuts.
