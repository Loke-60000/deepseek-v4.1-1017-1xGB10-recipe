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

<p align="center"><img src="docs/assets/speed.png" alt="1 stream: 29-30 tok/s, first token 0.3 s; 4 streams: 92-97 tok/s in total, 24-25 per stream, first token 0.5-1.0 s; 8 streams: 100-136 tok/s in total, 14-19 per stream, first token 1.6-3.1 s. Prefill 850-1,370 tok/s. Accuracy on GPQA Diamond, AIME 2025, HumanEval+ and MMLU-Pro is being measured; the full model scores 90.9 % on GPQA Diamond." width="100%"></p>

Greedy decoding, 512 tokens per request, warm cache. Accuracy runs on the same box at high reasoning effort.

Known limits:

- A long *new* prompt runs every expert and pushes the cached ones out, so the next few replies drop to 10-19 tok/s
  until the cache settles. Follow-up turns reuse the cached prompt and are not affected.
- DSpark speculative decoding runs while one request is live and pauses while several are
  (`TF_DSV41_SLOT_DRAFT_MAX_LIVE=1`): drafting for several streams at once reads far more experts from disk on this
  memory budget and halves 4-stream throughput.
- Each token runs the fewest of its 6 routed experts that carry 85 % of the router weight
  (`TF_DSV41_EXPERT_TOPP=0.85`). Running all 6 is a third slower here.

## Running it

The images are prebuilt on GitHub, so there is nothing to clone or compile:

```bash
mkdir -p ~/.local/bin
docker run --rm --entrypoint cat ghcr.io/loke-60000/dsv41-1xgb10-toolbox:latest /recipe/bin/dsv41 > ~/.local/bin/dsv41
chmod +x ~/.local/bin/dsv41
dsv41 graft && dsv41 engram && dsv41 assets --profile serve
dsv41 serve start --profile serve
```

The API is then on `http://<host>:8888/v1`.

- [docs/running.md](docs/running.md): each step, settings and profiles, the derisked model id, benchmarks, and
  working from a clone.
- [AGENTS.md](AGENTS.md): read this first if you are an agent deploying the model.
- [patches/](patches/README.md): the engine changes.

## Credits

DeepSeek for the model. ashhart for [TensorFold](https://github.com/ashhart/TensorFold). jayleaton for the
[DeepSeek-V4.1 TensorFold kit](https://github.com/jayleaton/deepseek-v41-tensorfold-spark) and the flash17 pack.
vcruz305 for the SAGE and DERISKED packs, the cold-expert approach, and the starting expert list in
`config/cold/vllm-recipe-1380.json`, taken from vcruz305's
[Spark recipe](https://github.com/vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe) (MIT). turboderp for EXL3.
