# Cold expert sets

`TF_DSV41_COLD` names one of these files: `{"layers.<i>": [expert, ...]}`. Listed experts stay in the engine's
prepared files and the GPU reads them through a read-only mapping; every other expert is uploaded.

| File | Experts | Trellis bytes | Source |
|---|---:|---:|---|
| `vllm-recipe-1380.json` | 1,380 | 7.36 GiB | routing-frequency selection from [vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe](https://github.com/vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe) (MIT), `configs/cold-alias-selection.json` |
| `measured-{10,15,20,25,30,35}g.json` | 1,708 to 5,450 | 10 to 35 GiB | `dsv41 cold --hits hits-calibration.json`: 0.30 % to 4.88 % of the calibration run's picks |

`hits-calibration.json` is `TF_DSV41_HITS` output from `dsv41 calibrate --per-kind 12` (60 requests, ~93K tokens: code,
prose, math, tool calls, six languages): routing is skewed enough that the coldest 30 % of experts take 2.8 % of picks.
