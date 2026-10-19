# Cold expert sets

`TF_DSV41_COLD` names one of these files: `{"layers.<i>": [expert, ...]}`. Listed experts stay in the engine's
prepared files and the GPU reads them through a read-only mapping; every other expert is uploaded.

| File | Experts | Trellis bytes | Source |
|---|---:|---:|---|
| `vllm-recipe-1380.json` | 1,380 | 7.36 GiB | routing-frequency selection from [vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe](https://github.com/vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe) (MIT), `configs/cold-alias-selection.json` |
