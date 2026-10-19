# DeepSeek V4.1-Flash-1017 deployment research (2026-10-20)

Scope: how DeepSeek serves V3 -> V4.1 in production, their recommended settings, which techniques transfer
to ONE DGX Spark (GB10, 128 GB unified, ~119-121 GiB to CUDA) running TensorFold `tensorfold-dsv41`
with EXL3 (~1.62 bpw routed experts), and what public Spark deployments measured.

Legend: [P] = primary source (DeepSeek paper / repo / HF card / API docs). [C] = community repo.
[D] = my derivation from primary numbers. [?] = uncertain / not verified.


---

## 0. Model facts that matter for serving (from config.json + V4.1 report) [P]

Sources: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/config.json ,
V4.1 report https://arxiv.org/abs/2609.19969 (also https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)

- 40 layers = **Causal Encoder-Decoder (CED)**: layers 0-19 "encoder", 20-39 "decoder". Decoder global KV is
  *projected from the layer-19 hidden state*, so prefill of a prompt only needs the 20 encoder layers for
  all tokens; decoder layers only run over the last `n_win`=128 tokens ("Decoder SWA Bounded Replay").
  Report: "nearly halves prefill computation"; 8B active/token at prefill, 16B at decode.
- Attention: layers 0-1 pure SWA (window 128). Encoder layers 2-19: CSA2 compress ratio m=2, 3 groups of 6
  (1 Full + 5 Reuse). Decoder 20-39: m=1, 5 groups of 4 (first group Full+3 Reuse; others Reindex+3 Reuse).
  `kv_source_layer_ids=[2,8,14,20]` (only 4 layers own main KV + indexer K),
  `index_source_layer_ids=[2,8,14,20,24,28,32,36]`, index top-k 512, indexer 32 heads x 128 dim.
  Hierarchical Sparse Indexer: layer 20 picks 2048 blocks x 8 = 16,384 candidates; later Reindex layers
  score only that pool (decode indexer cost ~constant in context length beyond layer 20's full scan).
  Report: decode FLOPs rise only ~1/4 going 4K -> 1M context.
- KV precision: main KV **FP4 (E2M1, one E4M3 scale per 16 ch, no global scale) = 288 B/entry**; SWA KV stays
  **FP8 = 528 B/entry** (FlashMLA format: 512 B e4m3 + 16 B e8m0 scales). Indexer Q/K FP4 (MXFP4) via QAT.
  FlashMLA README: "528 (V4.1) or 288 (V4.1 fp4) bytes per token"
  https://github.com/deepseek-ai/FlashMLA (2026.09.30 release dropped Hopper; SM100/SM103 only -> nothing
  runs as-is on sm_121).
- **[D] KV bytes per token (persistent/global part)**: main KV 288 B x (3 encoder sources / 2 + 1 decoder
  source) = 720 B; indexer K ~68 B x 2.5 = ~170 B -> **~0.9 KB/token -> ~0.85 GiB per 1M-token sequence
  with FP4 main KV** (~1.6 GiB/1M with FP8 main KV). SWA state is per-sequence constant:
  40 x 128 x 528 B = ~2.7 MB/seq. Cross-check [C]: vLLM single-Spark recipe reports 1 GiB KV pool =
  760,070 tokens (~1.41 KB/token incl. overhead, not FP4 main KV).
  => On one Spark, **context length is cheap; the KV pool is ~1 GiB per 1M tokens of *total* live context**.
- Engram: `engram_layer_ids=[1,14]`, per position 3 n-gram orders (2,3,4) x 8 hash heads = 24 rows per module,
  row = `engram_head_dim` 256 x FP8 = 256 B (+scale). `engram_num_embeddings` 384,006,168 + 384,016,682 rows
  -> [D] 196.6e9 B = ~183 GiB (+scales; matches "196B Engram params", the ~189 GiB on disk).
  **[D] Per token: 48 random 256-B row reads (12 KiB useful; ~192 KiB if every read is a 4 KiB NVMe page)**.
  Hash ids depend only on token ids (compressed vocab 99,092, NFKC/lowercase-normalized) and the previous 3
  tokens -> fully known before the forward pass (see `inference/engram.py`).
- DSpark drafter: 3 MoE blocks (128 routed experts, top-3), SWA 128, `dspark_block_size=5`, Markov head
  rank 256, confidence head, conditioned on target layers 37/38/39. Lossless (rejection sampling).
- Routing: 384 routed + 1 shared, top-6, `sqrtsoftplus` scoring, noaux_tc, routed scale 1.5; expert
  intermediate 2304. Released weights: FP8 (32x32 blocks, ue8m0 scales), routed experts FP4.
- YaRN factor 16 over 65,536 original -> `max_position_embeddings` 1,048,576.

---

## 1. How DeepSeek deploys (V3/R1 -> V4 -> V4.1)

### 1.1 V3/R1 (Feb 2025, Open Source Week day 6) [P]
https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md
- Prefill/decode disaggregation. Prefill unit: 4 nodes, routed experts EP32, MLA/shared expert DP32,
  32 redundant experts, 9 routed + 1 shared per GPU. Decode unit: 18 nodes, EP144/DP144, 2 routed + 1 shared/GPU.
- Dual-batch overlap in prefill (two micro-batches alternate so one's all-to-all hides behind the other's
  compute); decode splits attention into two steps and runs a 5-stage pipeline.
- Load balancers: prefill (attention compute / tokens per GPU), decode (KV usage / requests per GPU), EP
  (max dispatch receive load) -> EPLB https://github.com/deepseek-ai/EPLB (redundant hot experts).
- FP8 for matmuls and dispatch; BF16 for core MLA and combine.
- 24 h stats: 608B input tokens, **56.3% (342B) hit the on-disk KV cache**; ~73.7k input tok/s per H800
  node at prefill (incl. hits), ~14.8k output tok/s per node at decode; 20-22 tok/s per user; average KV
  length per output token 4,989.
- MTP (V3 report https://arxiv.org/abs/2412.19437, sec. 5.4.3): 1 extra token, 2nd-token acceptance
  85-90%, ~1.8x TPS. MTP-1 stayed the production setting until DSpark (V4 preview) because static
  MTP-3/5 "strictly degrades aggregate throughput under high concurrency" (DSpark paper).
- On-disk KV ("Context Caching on Disk"), 3FS as backend: https://github.com/deepseek-ai/3FS (KVCache read
  throughput peaks ~40 GiB/s aggregate across clients). API doc https://api-docs.deepseek.com/guides/kv_cache:
  on by default, hit requires full match of a "prefix unit"; units persisted at end of user input, end of
  model output, when a common prefix is detected, and at fixed token intervals for long inputs; best effort;
  cleared after "a few hours to a few days". (The old 64-token granularity is no longer stated. [?])

### 1.2 V4 (V4-Pro / V4-Flash, 2026) [P]
V4 report https://arxiv.org/abs/2606.19348
- Fine-grained EP: experts split into "waves"; dispatch/L1/L2/combine of different waves overlap in one
  mega-kernel (MegaMoE in DeepGEMM). 1.50-1.73x for general inference, up to 1.96x for small-batch
  latency-sensitive serving (RL rollouts, agents).
- Routed experts FP4 (QAT; FP4->FP8 dequant is lossless), indexer QK in FP4, KV mostly FP8.
- Batch-invariant + deterministic kernels (attention dual-kernel, DeepGEMM end-to-end) -> cached prefix KV is
  bitwise the same as recomputed. Relevant if you want prefix-cache hits to be numerically invisible.
- KV manager: SWA + uncompressed tail treated as fixed-size per-sequence "state cache" pool; compressed KV
  in paged blocks (block = multiple of lcm of compression ratios).
- On-disk KV cache: compressed CSA/HCA entries stored in full; on hit reuse up to the last complete
  compression block; tail recomputed. SWA KV ~8x the volume of compressed KV, so three options:
  Full SWA caching / Periodic checkpointing (last n_win every p tokens) / Zero SWA caching (recompute
  n_win x L tokens).
- DSpark (paper https://arxiv.org/abs/2607.05147, code https://github.com/deepseek-ai/DeepSpec):
  replaced MTP-1 two weeks after V4-preview launch. gamma=5, Markov head, calibrated confidence head,
  hardware-aware scheduler choosing per-request verify length from a profiled SPS(batch) table.
  Live traffic: +60-85% per-user tok/s (V4-Flash) at matched throughput; +51% throughput at 80 tok/s/user
  SLA. Offline (Qwen3 targets, gamma=7, no scheduler): accepted length ~6 math, ~5 code, ~3.3-3.7 chat;
  static-threshold sweep raises chat acceptance 45.7% -> 95.7%. Scheduler runs asynchronously, using
  confidences from two steps earlier to size the next batch so CUDA graphs / zero-overhead scheduling
  are kept.

### 1.3 V4.1-Flash (Sept 2026, 1017 refresh Oct 17) [P]
V4.1 report sec. 3.2 (pdf pages 18-20)
- **Encoder-Prefill-Decode (EPD) disaggregation**: vision encoder, prefill, decode scale separately.
- Kernels: FlashMLA (fused RoPE-attn-RoPE-cast), DeepGEMM Mega-Gate / **Mega-mHC** / Mega-MoE,
  TileKernels, DeepSelect TopK. A Reuse-mode CSA2 layer is 15 kernels at prefill, 11 at decode.
  Mega-mHC fuses residual update + input mixing + coefficient prediction + pre-norm + FP8 cast: residual
  read once/written once (2n+2)d (halves activation traffic). Single-Pass mHC uses A_{l-1} for input mixing
  so the mixing doesn't wait for the reduction.
- **Persistent KV cache** = global KV only (main + indexer K), on SSD, LRU, kept >= 72 h. V4.1 persistent
  footprint is 1/8 of V4 (1/2 from dropping SWA, 1/4 from CSA2 sharing + FP4).
  SWA KV lives in a host-DRAM pool (10% of host DRAM per machine), TTL of minutes, stored at two points:
  end of prompt and end of output (for regenerate / next turn).
- **SWA Bounded Replay**: on SWA miss, replay only the last n_win=128 tokens of the cached prefix (not
  L x n_win) together with the new suffix; also used for every prefill to build decoder SWA KV.
  Trained-for (simulated during post-training). Result depends slightly on hit position (not bitwise
  equal to a cold prefill) - "barely compromises response quality".
- Engram: 196B params in two modules (layers 1, 14), FP8 tables, "deterministic addressing enables
  embeddings to be prefetched from host memory via background RDMA transfers, with prefetching for the first
  module overlapping computation in the first Transformer block". Inference uses "sharded Engram embedding
  tables". During RL rollouts tables stay in GPU memory to avoid host-memory fragmentation OOMs.
  Engram paper (https://github.com/deepseek-ai/Engram/blob/main/Engram_paper.pdf): 100B table fully in host
  DRAM via PCIe prefetch costs 2.8% throughput on an 8B dense model on H800; N-grams are Zipfian ->
  proposes a multi-level cache: hot rows in HBM/DRAM, long tail on NVMe. No NVMe numbers published.
- DSpark in V4.1: 3 blocks, SWA 128, 5 draft positions, scheduler maximizes system throughput given load.
- No public per-GPU batch sizes, EP degree, or cache hit rates for V4/V4.1 [?]. API pricing shows cache-hit
  input at 1/50 of miss ($0.003 vs $0.15 /1M off-peak) and a 2,500 concurrency limit for deepseek-flash
  (https://api-docs.deepseek.com/quick_start/pricing).

---

## 2. Recommended inference settings (V4.1-Flash-1017) [P]

- HF card: temperature 1.0, top_p 1.0, context 1M, max_tokens >= 256K. (inference/README's interactive
  example uses `--temperature 0.6`; treat as an example, not a recommendation.)
- Report evals: reasoning temp 1.0/top_p 1.0; agent harness temp 1.0/top_p 0.95 with 1M context.
- API (https://api-docs.deepseek.com/guides/thinking_mode): thinking on by default; in thinking mode
  temperature/penalties are ignored; for deepseek-flash top_p effective range 0.95-1.0 (values <0.95
  clamped to 0.95), fixed 1.0 outside thinking. Max output up to 384K.
- deepseek-recipe 0.2.0 (2026-10-20, https://github.com/deepseek-ai/deepseek-recipe/blob/main/CHANGELOG.md):
  temperature forced 1.0 in thinking mode; V4.1 thinking top_p >= 0.95; non-thinking top_p 1.0; default
  max_tokens 8192 (chat) / 65536 (thinking) / 131072 (thinking+max); upper bound 393,216.
- Reasoning effort: numeric 1-100 prefix `Reasoning Effort: {b} (range 1-100, ...)` in a `<｜System｜>`
  block at conversation start, only in thinking mode. Aliases low=50, high=75 (default), max=100. API maps
  minimal/low->low, medium/high/xhigh->high, max/ultra->max.
- Prompt format: no Jinja template in 1017; use `encoding/encoding.py` or deepseek-recipe. V4.1 changes vs
  V4: DSML tags with leading space (`<｜DSML｜ calls>`), numeric effort, mid-conversation `<｜System｜>`.
  Chat mode = `<｜Assistant｜></think>`; thinking mode = `<｜Assistant｜><think>`.
- **2026-10-16 encoder fix** (commit 314113f on deepseek-ai/DeepSeek-V4.1-Flash, "fix(encoding): preserve
  tool image order and current-turn thinking"; already included in the 1017 repo):
  1) user messages that contain only tool results no longer count as a new user turn, so `drop_thinking`
     keeps reasoning inside the current turn across tool calls (previously it was stripped after each tool
     result when no tools were declared);
  2) tool-result content blocks may carry images; tool messages are merged/sorted by call order *before*
     images are collected, so `media["images"]` matches prompt order;
  3) image-placeholder validation now also applies to tool messages.
  Related inference fixes on the same repo: 2026-10-12 "Restore each indexer's K cache on incomplete
  compression steps" (indexer K pointer must be published even when no new compressed entry completed this
  step - affects decode with m=2 layers) and FP8 act-quant of attention output before the grouped `wo_a`
  einsum (main + DSpark attention). Worth checking TensorFold matches both.
- Prompt-cache consequence: `drop_thinking` strips earlier-turn reasoning when no tools are declared, so the
  next turn's prompt diverges from the KV generated during the previous turn right where `<think>` began.
  With tools declared, all reasoning is kept, so agent loops are append-only and prefix-cache friendly.

---

## 3. What transfers to one unified-memory Spark, and the levers (ranked by expected impact)

### 3.0 The budget that drives everything
- Usable ~119-121 GiB; weights ~118 GiB (TensorFold's SAGE-1.59 TP1 breakdown: experts 101.4 + non-routed
  13.1 GiB [attn 3.25, embed 1.23, shared 0.75, head 0.46, DSpark/MTP 6.75], verdict "does not fit at a
  useful context" - https://github.com/jayleaton/deepseek-v41-tensorfold-spark/blob/main/docs/campaign/PARKED-SAGE-1.59.md).
  So free memory is **1-3 GiB** before any lever. Every GiB freed is worth ~1.1M tokens of FP4 KV.
- What the free GiB must hold [C]: KV pool, per-chunk prefill workspace (TensorFold: 2,268 MiB per 4K
  chunk; 0xBakeer: 7.2 GB + 15.1 KB/token-of-context per 2K chunk in PyTorch), CUDA graphs (~15 MiB host
  each), Engram staging/row cache, and **page cache** (on GB10 page cache is the same DRAM; `cuMemGetInfo`
  counts it as used; MemAvailable is the real signal; box hard-hangs near 0 instead of OOM-killing).
- **Context is not the constraint; memory headroom for KV + prefill scratch is.** At FP4 the whole
  persistent KV of a 1M-token request is ~0.85 GiB [D]; per-sequence SWA/compressor state ~3 MB.

### 3.1 Ranked levers

| # | Lever | Goal hit | Expected impact | Evidence |
|---|---|---|---|---|
| 1 | **Cold-expert tier via read-only mmap side file accessed by the GPU over ATS** (least-routed experts by per-layer routing frequency, 64-B aligned, stays in page cache) | frees ~8-9 GiB -> slots, context, prefill chunk | Largest single memory lever. vLLM single-Spark: 8.59 GiB side file of 1,380 least-routed experts + embedding; 13.5 -> 24.2 tok/s vs whole-layer aliasing (whole-layer alias touched ~10%/step at ~6.6 GB/s = ~130 ms of a 292 ms step). Caveats: only works while the side file stays page-cache resident (MTP made resident -> page cache collapsed -> 9.5 tok/s); true NVMe streaming loses badly (0xBakeer: 2.6 tok/s, 0.92 GB/token from NVMe); prefill touches 370-381/384 experts per layer per chunk, and each extra live slot adds ~5-6 distinct experts/layer, so the cold set must be truly cold *under the target concurrency* (profile routing at C4/C8, not C1) | vcruz README/ENGINEERING_NOTES; 0xBakeer RESULTS/LIMITATIONS; jayleaton G17-LEVERS |
| 2 | **Use the model's native FP4 main KV (288 B, E2M1 + E4M3/16) and FP4/NVFP4 indexer K; keep SWA FP8** | context x concurrency | TensorFold today: 584 B FP8 comp rows + 132 B FP8 index = **1,790 B/token** (FP4 rows exist in the layout enum but are not wired). FP4 => **~890 B/token** (halves pool: 1M tokens 1.67 -> ~0.83 GiB). Model was QAT-trained for FP4 main KV + FP4 indexer, so this is the deployment format, not a lossy hack. coolbho3k: scale search per 16-group (amax/6.5..2.5) NMSE 0.911% -> 0.668%; NVFP4 indexer raises top-512 recall vs MXFP4. Do NOT FP4 the SWA KV (DeepSeek: "sensitive to quantization") | V4.1 report 2.4.4; FlashMLA README; jayleaton kv/layout.zig; coolbho3k nvfp4_kv |
| 3 | **Display-memory "carveout"** (headless, `nvidia_drm modeset=1 fbdev=0`, DRM dumb buffer -> mmap -> `cuMemHostRegister(DEVICEMAP|IOMEMORY)` -> device ptr) | +1.75 GiB | 1,879,048,192 B usable that CUDA otherwise can't touch; MemAvailable drops only ~18 MiB. Reads ~169 GB/s vs ~262 GB/s, so put cold-ish data there: KV pool (~2M FP4 tokens), embedding, Engram row cache. Driver-dependent (failed on an ASUS GX10 with 595.84) | coolbho3k docs/display-memory.md, release/runtime/sources/display_kv.c |
| 4 | **Disk-backed prefix/session cache, DeepSeek-style** | prompt caching, concurrency (idle sessions off-pool) | DeepSeek V3: 56.3% of input tokens hit disk cache; V4.1 persists only global KV (main + indexer, ~0.9 KB/token -> a 1M session is ~0.85 GiB on NVMe), keeps SWA only at end-of-prompt and end-of-output in a short-TTL RAM pool, and on SWA miss replays the last 128 tokens (Encoder SWA Bounded Replay; trained for). TensorFold already has a prefix trie + NVMe tier (1M-token resume in 1.3 s) - align it: (a) store FP4 global KV only, (b) SWA snapshots only at turn boundaries, (c) bounded 128-token replay instead of exact replay, (d) treat slots as *active* decoders and park idle sessions to NVMe aggressively so "concurrent users" >> slots. justincasey: 133,855-token prompt TTFT 20.7 s -> 0.81 s; coolbho3k: SWA checkpoint every 4096 tokens, 11.4K fork 10.76 -> 3.36 s. Cache-key note: without tools `drop_thinking` removes prior-turn reasoning, so key the reusable prefix at the end of the previous *prompt*, not the end of output; with tools all reasoning is kept (append-only) | V4.1 report 3.2.1-3.2.2; V4 report 3.5.2; open-infra-index day 6; justincasey README; coolbho3k upstream-api-cache.md; jayleaton README-STATS |
| 5 | **DSpark verify length scheduled by load** (DeepSeek's hardware-aware prefix scheduler: global greedy over calibrated prefix-survival probs a_{r,j} against a boot-profiled SPS(B) table; run it asynchronously with confidences from 2 steps back so graphs stay fixed) | decode at C>1 | Single stream: k=5 is the knee (vLLM: k=7 +17 ms for +0.55 tok/step). Under concurrency deeper k hurts: coolbho3k C6 aggregate K3 62.05 / K4 54.99 / K5 50.74. Acceptance is content-driven: prose ~1.6-2.5 tok/step, code 3.3-4.7, structured/markup ~5-6. A per-request budget lets code streams keep k=5 while prose drops to 1-2 instead of a global k | DSpark paper 3.2.2, 5.2; vcruz; coolbho3k dspark/RESULTS.md; jayleaton DRAFT-ACCEPTANCE |
| 6 | **Prompt-lookup n-gram drafts on top of DSpark** (replace drafts with the continuation of the latest earlier match of the last 3 tokens in the last 64K; point-mass draft keeps it exact) | decode on agent/code-edit traffic | 2.4-3.5 tok/step on copy-heavy work, no prose regression; edit/tweet ~6.2 tok/round with lookup | coolbho3k ngram_draft/RESULTS.md; jayleaton RESULTS |
| 7 | **Prefill: CED decoder bounded replay + 2K chunks** | prefill speed, memory | Run decoder layers 20-39 only on the last 128 prompt tokens (0xBakeer -41% prefill). Chunk 2K on one Spark: vLLM 4096 halves varied-text prefill (446 vs 718 tok/s) and 8192 OOMs; coolbho3k 3072 is 2% slower and +0.45 GiB. Each chunk re-reads every routed expert (~370-381/384 touched), so larger chunks only help if memory allows; size the chunk from MemAvailable at admission (TensorFold already falls back 4K -> 2K). `INDEX_BUDGET_MIB=64` avoids +1.6 GiB of cached segments in long prefills | V4.1 report 2.2/3.2.2; vcruz; coolbho3k; jayleaton OPERATIONS |
| 8 | **Engram: page-aligned O_DIRECT repack + small hot-row cache + prefetch on drafts** | protects memory, a few ms/step | Per token 48 rows x 256 B (+scale). Repack 15 rows per 4 KiB page so one read = one page (coolbho3k "page15"); O_DIRECT so the 183 GiB table never grows the page cache (vLLM recipe saw allocator stalls when page cache grew 6 -> 10 GiB); hash ids depend only on token ids, so prefetch the bonus/anchor row at verify end and draft rows as soon as the drafter samples (TensorFold has `ENGRAM_PREFETCH`); hot-row LRU in RAM (Zipfian per Engram paper; third-party: 4 GiB cache 67-76% hit, top 100M rows cover 92.7% [?]) - but on one Spark every MiB of row cache competes with KV, so keep it small (tens-hundreds of MiB) or put it in the carveout. Expected gain only ~1-3 ms/step (coolbho3k +1.4% decode) | Engram paper; V4.1 report 2.4.2; coolbho3k engram_io; tonyd2wild RECIPE; jayleaton engram_rows.zig |
| 9 | **Drop / move small resident blobs** | +0.5-2 GiB | RoPE cache trimmed to max position (-0.484 GiB in vLLM); input embedding (1.23 GiB) to SSD/side file with a 40 MiB direct-mapped vocab row cache (coolbho3k: +4% decode with deferred graph validation); vision tower ~0.8-0.9 GiB bf16 [D] if serving text-only. Dropping the 6.3-6.75 GiB drafter frees the most but costs ~2x decode - don't | vcruz; coolbho3k host_roundtrips; jayleaton PARKED-SAGE-1.59 |
| 10 | **Host hygiene on unified memory** | stability = usable headroom | `vm.compaction_proactiveness=0`, `vm.watermark_boost_factor=0`, `vm.swappiness=0`, drop caches before load, MemAvailable guard (0.5 GiB kill floor sampled at 50 ms in vLLM; TensorFold floors 4/5 GiB), `MIMALLOC_ALLOW_THP=0`, keep NVMe quiet (a 97%-full disk boot ran 509 ms/step for its whole life), never build/run a second CUDA process beside the server, check for the GB10 clock latch (631-949 MHz stuck; power-cycle) | vcruz scripts; jayleaton OPERATIONS; tonyd2wild gpu-clock-latch.md |
| 11 | **Long-context decode: score indexer straight from KV pages; Reindex layers 24/28/32/36 score only layer 20's 16,384-candidate pool; length-aware radix top-k** | decode at 256K-1M | coolbho3k: decode at 1M 81.7 -> 67.5 ms/step (vs 63-65 short ctx); top-k 0.69 -> 0.03 ms. This is DeepSeek's Hierarchical Sparse Indexer as designed - check TensorFold isn't scanning full context in every Reindex layer | V4.1 report 2.3.2; coolbho3k README / indexer notes |

### 3.2 Correctness audits worth doing (cheap, can silently cost speed or quality)
- Drafter context-KV writer must use the same SWA FP8 group format as the target (coolbho3k: mismatch kept
  output exact but cut acceptance 13-16%; fix gave +11-13% decode).
- Indexer K publish on non-completing compression steps (DeepSeek fix 2026-10-12) and FP8 act-quant of attention
  output before grouped `wo_a` (main + DSpark).
- Router gate must stay fp32 (bf16 picked wrong experts for 11-31% of tokens - 0xBakeer).
- Encoder: use the post-10-16 encoding (1017 repo or deepseek-recipe 0.2.0).
- Thinking-mode flat-distribution bug reported on 4x Spark/vLLM (justincasey, open; suspect top-k padding read
  as real entries): run "Say hi.", effort high, T=0 and check top-1 prob.

### 3.3 What does NOT transfer
- EP / DeepEP / MegaMoE wave overlap, dual-batch overlap (coolbho3k: micro-batching lost on unified memory),
  PD/EPD disaggregation across machines, 3FS. Analog on one box: prefill chunks interleaved with decode steps
  and a per-step token budget.
- Static k>3 at high concurrency (DeepSeek kept MTP-1 for that reason until the scheduler existed).
- Lossy shortcuts tried by others: expert top-k cuts / REAP pruning (0xBakeer keep-set: 17-37 tok/s but coverage
  failures), lossy verify budget (broke tool calls), NVFP4 experts ("no bandwidth saved on GB10").

---

## 4. Public Spark deployments of DeepSeek V4/V4.1 (measured, as published)

| Deployment | HW | Engine / quant | Decode c1 | Concurrency | Prefill | Context | Key tricks |
|---|---|---|---|---|---|---|---|
| vcruz305 DSV4.1-Flash-1017-EXL3-Spark-recipe | **1x Spark** | vLLM fork, EXL3 SAGE 1.62 bpw | 29.5-34.0 tok/s (140-146 ms/step, 4.1-4.7 tok/step, DSpark k=5) | 1 seq | ~720 tok/s varied text; 1,005-1,179 long | 229,376 (needle OK at 225K; TTFT 196 s) | cold-expert ATS side file 8.59 GiB; tile-K 16; KV 1 GiB; prefix cache off; chunk 2112; guard 0.5 GiB |
| 0xBakeer deepseek-v41-flash-spark | 1x Spark | PyTorch+Triton, FP4/CB3 keep-set | 17-37 tok/s (pruned router, lossy) / 2.6 tok/s true NVMe streaming | 1 | 337 tok/s @5K | 131K validated, 256K at keep 0.36 | per-layer saliency keep-sets; Engram pread + 200k-row cache; decoder bounded replay |
| llama.cpp V4-Flash-0731 (not 4.1) | 1x Spark | UD-IQ2_M | 19.3-19.7 | 4 slots x 131K, "52 tok/s @4" claimed | 419-462 | 524,288 total | split ctx across slots; q8_0 KV broke output |
| jayleaton deepseek-v41-tensorfold-spark | 2x Spark TP2 | TensorFold Zig, EXL3 2.8 bpw | 87.9 (64 @32K-131K) | 4 slots, x4 ~154 aggregate | 2,198-2,349 | 1M, 2.1M-token shared pool | pool sizing vs 4 GiB floor; NVMe session tier; Engram AIO + prefetch; clock cap 2100 |
| coolbho3k DeepSeek-v4.1-Flash-2x-DGX-Spark | 2x TP2/DCP2 | EXL3 3 bpw | 30.78 pooled | C6 53-65 | ~1,078-1,219 @32K | 1M/request, 6 slots, 3.31M KV tokens | display-memory KV, FP4 KV, page15 Engram, n-gram drafts, K3 |
| justincasey ...-1017-4x-DGX-Spark | 4x TP4 | vLLM, MXFP4 | 65.1 prose / 93.9 code | c16 552-639 | 6,622 @32K, 4,160 @1M | 1M, 16 seqs | persistent KV snapshots (TTFT 20.7 -> 0.81 s) |
| tonyd2wild DeepSeek-V4.1-Flash-vLLM-DGX-Spark | 3-4x | vLLM / TensorFold TP4 | 98.6 (TF TP4) | C16 270.8 | 2,144-2,266 | 1M, 16 | Engram rows staged in prepare_inputs (2-3 ms/step); page-cache flusher |
| 0xSero DeepSeek-V4.1-Flash-Two-Sparks | 2x | EXL3 2.77 bpw, k=7 | 29-41 | c2 60 | 2,050 | 262K, 2 seqs | - |

Repos: https://github.com/vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe ,
https://github.com/0xBakeer/deepseek-v41-flash-spark ,
https://forums.developer.nvidia.com/t/1x-dgx-spark-deepseek-v4-flash-0731-on-llama-cpp-131k-ctx-slot-19-7-tok-s-single-stream-52-tok-s-4-concurrent/379129 ,
https://github.com/jayleaton/deepseek-v41-tensorfold-spark , https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark ,
https://github.com/justincasey/DeepSeek-V4.1-Flash-1017-4x-DGX-Spark , https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark ,
https://github.com/0xSero/DeepSeek-V4.1-Flash-Two-Sparks .
Detailed per-repo notes: `digest-single-spark.md`, `digest-tensorfold.md`, `digest-multispark.md` in this directory.

No public single-Spark deployment measured more than 1 concurrent request on V4.1, prefix-cache benefit, or 1M
context on one box. A ~20.85 tok/s single-Spark V4.1 claim and an Oct 8 NVIDIA forum thread ("113.6 GB base
model + 40 MB domain sidecars") were seen in search results but not verified [?].

## 5. Open questions / uncertain
- Exact DeepSeek prefix-unit interval and hash keying for V4.1 (API doc no longer states 64 tokens).
- DeepSeek per-GPU batch sizes / EP degree for V4.1: not published.
- Engram hot-row hit rates: only third-party numbers.
- Whether the display carveout works on our driver version.
- My KV byte math assumes indexer K stored only at the 4 `kv_source_layer_ids` (consistent with the report's
  "indexer K projected from main KV" and with the reference implementation's 3,200 B/token in bf16).
