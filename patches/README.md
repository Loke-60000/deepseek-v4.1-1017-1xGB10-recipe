# Engine patches

Applied in order to jayleaton's [DeepSeek-V4.1 TensorFold kit](https://github.com/jayleaton/deepseek-v41-tensorfold-spark)
at the commit pinned in `config/recipe.json`, under `engine/zig`. `python3 -m recipe.cli image` applies them.

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
| 0010 | `TF_DSV41_SLOT_DRAFT_MAX_LIVE`: DSpark drafts only while at most that many requests are live (unset: unchanged) |
