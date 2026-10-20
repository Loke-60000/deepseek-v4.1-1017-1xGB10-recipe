# Cold expert sets

`TF_DSV41_COLD` names one of these files: `{"layers.<i>": [expert, ...]}`. Listed experts stay in the engine's
prepared files and the GPU reads them through a read-only mapping; every other expert is uploaded.

| File | Experts | Trellis bytes | Source |
|---|---:|---:|---|
| `vllm-recipe-1380.json` | 1,380 | 7.36 GiB | routing-frequency selection from [vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe](https://github.com/vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe) (MIT), `configs/cold-alias-selection.json` |
| `measured-{10..65}g.json` | 1,656 to 9,324 | 10 to 65 GiB | `dsv41 cold --hits hits-calibration.json`: 0.29 % to 18.5 % of the calibration run's picks |

`hits-calibration.json` is `TF_DSV41_HITS` output from `dsv41 calibrate --per-kind 12` (60 requests, ~93K tokens: code,
prose, math, tool calls, six languages): routing is skewed enough that the coldest 30 % of experts take 2.8 % of picks.

On one GB10 with drafts off, 45 GiB is the best split (`serve`): larger mapped sets leave the page cache, which follows
the current traffic, more experts to manage. Measured sweeps: 25 GiB 66-75 tok/s at 4 streams and 8.0 GB read from
NVMe, 35 GiB 79-93 and 3.1 GB, 45 GiB 92-97 and 0.5 GB, 55 GiB 77-95 and 4.5 GB, 65 GiB 89-97 and 1.9 GB.
