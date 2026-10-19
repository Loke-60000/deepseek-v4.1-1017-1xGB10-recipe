# Multi-Spark DSV4.1-Flash digest (2026-10-20)

R1=https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark/blob/main/ · R2=https://github.com/justincasey/DeepSeek-V4.1-Flash-1017-4x-DGX-Spark/blob/main/ · R3=https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/main/ · R4=https://github.com/0xSero/DeepSeek-V4.1-Flash-Two-Sparks/blob/main/
1017 checkpoint: R2 and R3's `1017*` runs. R1, R4 and R3 boot3–10 used the September checkpoint.

| Setup | c1 decode | Concurrent | Prefill | Ctx/slots |
|---|---|---|---|---|
| R1 2x TP2/DCP2, EXL3 3bpw, DSpark K3 | 30.8 pooled (23–37) | C6 53–65 agg | 32K 1,078–1,219; 1.03M prompt 25.6–31.5 min | 1M/req, 6 slots, 3,313,955 tok |
| R2 4x TP4 MXFP4, k=6 | 65 prose / 94 code | c8 ~410, c16 552–639 (34–43/stream) | 6,622@32K, 5,507@512K, 4,160@1M | 1M, 16 |
| R3 TensorFold TP4 1017 (3.5bpw) | 98.6 | C16 270.8 | 2,144–2,266 | 1M, 16 |
| R3 TensorFold TP2 1017 (2.9bpw) | 73.1 (code 105.6) | C4 135.8 | ~2,044–2,200 to 187K; 1M needle 641 s | 1M pool / 4 |
| R4 TP2 EXL3 2.77bpw, k=7 | code 39–41, prose ~29 | c2 60 | 2,050 | 262K, 2 |

## 1. Display-memory carveout
Sources: R1 docs/display-memory.md, README "How the display-reserved memory becomes KV", release/runtime/sources/display_kv.c.
- **What it is:** firmware reserves about 2 GB of RAM for display (UEFI setting, keep it at 2 GB). Ordinary CUDA allocations cannot use it.
- **Prerequisites:** run headless, with `nvidia_drm modeset=1 fbdev=0`.
- **How it works:**
  1. Create a DRM dumb buffer (`MODE_CREATE_DUMB`, 4096 × size/16384, 32 bpp).
  2. `MAP_DUMB` and `mmap` it.
  3. Register it with `cuMemHostRegister(DEVICEMAP|IOMEMORY)`.
  4. Get a device pointer with `cuMemHostGetDevicePointer` and use it as KV storage.
- **Gain:** 1,879,048,192 B (1.75 GiB) per GPU. MemAvailable dropped only about 18 MiB. R1 puts all of its KV there.
- **Cost:** about 169 GB/s reads, vs 262 GB/s for ordinary memory.
- **Risk:** it failed on one ASUS GX10 with driver 595.84 (`CUDA_ERROR_INVALID_VALUE` at registration). It works on 580.173.02.

## 2. KV bytes
- **Size:** main + indexer KV is (288 + 68) × (3/2 + 1) = **890 B per token in total**, excluding SWA, the compressor ring and padding (R1 nvfp4_kv/README.md).
  - That is about 0.83 GiB per 1M tokens.
  - R1 fits 3.31M tokens into 2 × 1.75 GiB, about 567 B/token/rank effective.
  - For comparison, R2 and R4 come to about 1.3 KB/token/rank (derived).
- **Main KV (`nvfp4_search`):** E2M1 values with an E4M3 scale per 16. For each group it tries every scale in [amax/6.5, amax/2.5] and keeps the lowest error. NMSE: 0.911% → 0.668%.
- **Indexer:** NVFP4 keys (72 B rows) with FP8 decode queries. Top-512 recall: 0.861–0.921 with MXFP4 → 0.894–0.941.
- **SWA:** FP8 group 32 with BF16 RoPE, 19,008 B per 32-token page.

## 3. Engram IO
- **R1 (release/experimental/engram_io/README.md, docs/engram-io-performance.md):**
  - Layout: "page15" repack, 15 whole rows per 4 KiB page so no row straddles a page.
  - Read path: O_DIRECT through a 96-thread pool, plus a 64 MiB RAM row cache per table.
  - Staging and compute: rows land in GPU-readable host staging and are dequantized on the GPU. Retrieval runs on a separate stream, with an event before each Engram layer.
  - Ownership: each rank owns half the tables (97.66 GiB per host).
  - Cold 256-token batch: 21 → 11 ms.
  - End to end it is only **+1.4% decode** (30.36 → 30.78), with flat prefill.
  - CPU-side lookup is negligible (<1 µs to 150 µs per batch, engram_cpu/README.md).
  - No GPUDirect Storage on Spark.
- **R3 (docs/RECIPE.md):**
  - Rows are staged in `prepare_inputs` before the forward pass, which keeps full CUDA graphs. Eager mode was host-bound at about 200 ms/step.
  - Cost: 2–3 ms/step from local NVMe, 6–8 ms over NFS.
  - Page-cache growth to 10 GiB starved the GB10 allocator once MemFree fell below about 4 GiB. Fix: a cache flusher or O_DIRECT.
  - No speculative prefetch, because the row hashes depend on the drafts.
- **R2:** io_uring reader (needs `seccomp=unconfined`); files must be local and defragmented.
- **Offset bug (R3 docs/boot3-wedge-and-engram-offset.md):** rank-local row ids were read from offset 0 of the full tensor, so ranks 1–3 read rank 0's rows. Output stayed fluent. Fix: add `row_start` to the bases.

## 4. DSpark
- **R1 K3 / K4 / K5 (dspark/RESULTS.md):**
  - Serial geomean 1.000 / 1.001 / 0.951.
  - C6 aggregate 62.05 / 54.99 / 50.74 tok/s.
  - Adaptive EMA came out at 0.977, and its 2.27 GiB graph pool broke the headroom limit.
- **Verify cap (R1 verify_cap/RESULTS.md):** keep the leading drafts while their cumulative confidence stays ≥ 0.1.
  - K5 + cap: −1.8% overall (code +25%, prose −11%). The drafter and attention rows dominate the cost, not MoE.
  - EMA {3,5} + cap: +0.7%.
- **Tokens/step:**
  - R1 at K3: prose 1.7–2.4, python 2.5–3.1, code edit 3.4, JSON 3.0.
  - R2 at k=6 (1017): prose 1.94, json 3.20, code 3.29. k=3 was 14–18% slower at c16.
- **Prompt-lookup on top of DSpark (R1 ngram_draft/RESULTS.md):**
  - Find the last 3 tokens earlier in the last 64K tokens and replace the drafts with what followed (point mass, so it stays exact).
  - Gives 2.4–3.5 tokens/step on copy-heavy work, with no prose loss.
- **Other sampling options:** probabilistic drafting +5.5% at T=1; block verification flat.

## 5. Prefix cache and chunking
- **R1 checkpoints:** an SWA checkpoint every 4096 tokens, stored in the evictable pool (docs/upstream-api-cache.md). TTFT for an earlier 11.4K-token fork: 10.76 → 3.36 s.
- **R2 persistent KV snapshot:** survives restarts. TTFT for 134K tokens: 20.7 → 0.81 s.
- **R1 chunk size:** 2048-token chunks. 3072 was 2% slower and cost 0.45 GiB. The MoE streams all 384 experts once per chunk, a floor of about 390 ms per chunk.
- **R2 chunk size:** 24,576 tokens.

## 6. Bugs
- **SWA draft bug (R1 drafter/SWA_DRAFT_BUG.md, swa_kv/CONTEXT_FIX.md):**
  - Cause: the target moved to FP8 group 32, but DSpark's `_insert_context_kv` still wrote group-64 pages. The reader applied group-32 scales.
  - Effect: outputs stayed exact; only acceptance fell, by 13–16%.
  - How it was found: an offline reference drafter accepted 40–50% more drafts.
  - Fix: route the drafter through the shared writer, worth +11–13% decode.
- **Thinking-mode corruption (R2 docs/findings/2026-10-18-thinking-mode-output-corruption.md): open, no root cause.**
  - Symptom: with thinking on, short answers get a flat distribution (top-1 0.10).
  - It reproduces in prefill and depends on context offset.
  - Suspects: top-k padding treated as real entries in sparse attention or the indexer, collectives, Engram.
  - Related: R3 found masked slots reading row 0 of KV block 0 (vllm#59689).
- **R3 boot4/5:** off-Hopper the indexer needs block size 128, and DeepGEMM pages need 64 states.
- **GPU clock latch (R3 docs/gpu-clock-latch.md):** the GPU sticks at 631–949 MHz and `nvidia-smi` doesn't show it. Fix: unplug for 30–60 s. Healthy check: an fp16 burn reaches 2.2–2.4 GHz.

## 7. What transfers to one Spark
1. **Display-memory KV:** +1.75 GiB.
2. **FP4 main KV, NVFP4 indexer, FP8 group-32 SWA:** at about 890 B/token, 4–8 slots of 1M fit in a few GiB.
3. **Embedding rows served from SSD** with a 40 MiB, 4096-slot row cache (R1 host_roundtrips/RESULTS.md). Cold lookup 300 → 165 µs; +4% together with deferred graph-error checks.
4. **Engram path:** page-aligned repack, O_DIRECT, small hot-row cache, pre-forward staging, GPU dequant.
5. **Indexer:**
   - Score straight from the KV pages, and have the indexers on layers 24/28/32/36 score only layer 20's candidates. At 1M context this took decode from 81.7 to 67.5 ms/step.
   - Length-aware radix top-k: 0.69 → 0.03 ms.
6. **Speculation:** prompt-lookup n-gram drafting, then a verify cap or content-adaptive k.
7. **Prompt caching:** SWA checkpoints plus persistent snapshots.
8. **Host settings:** `vm.compaction_proactiveness=0`, pin to the X925 cores, a MemAvailable watchdog, clock check.
9. **Does not transfer:** DCP / BF16 exchange, NCCL/RoCE tuning.

**Uncertain:**
- How much the slower display region hurts our access pattern.
- Whether TensorFold has the drafter SWA-format mismatch or the thinking-mode bug. Audit both.
