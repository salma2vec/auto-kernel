# Search-decision calibration: family-wise version passes locally

Measured on 2026-10-14 from frozen source revision `641295e` on Apple M1 CPU and
MPS, with MPS CPU fallback disabled. Both reports contain all 100 identical-code
null trials, 20 deliberately slowed positive controls, exact correctness records,
raw orders and timings. Neither device passed the prospective gate.

| Device | False promotions | Slowdown detections | Frozen gate |
| --- | ---: | ---: | --- |
| CPU | 0 / 100 | 8 / 20 | Failed sensitivity |
| MPS | 1 / 100 | 17 / 20 | Failed both requirements |

On CPU, the positive detections by shape were 0/5, 2/5, 5/5 and 1/5 in ascending
protocol shape order. Eleven of the twelve misses passed the 1.05 median effect
requirement but failed the global all-samples nonoverlap rule. On MPS, detections
were 3/5, 5/5, 4/5 and 5/5. The single false promotion was null seed 3046 at
shape `(4,256,512)`: identical code measured with a candidate/incumbent median
ratio of 0.597881, and all three old decision conditions passed. This is direct
evidence that separately timed blocks can drift enough to fool the rule on this
session. It does not estimate a universal false-promotion probability.

The negative result motivated the separately frozen [version 2 paired protocol](search-calibration-v2-protocol.md),
which interleaves candidate, incumbent and reference timings and replaces global
range separation with a practical paired effect plus exact sign gate.

Raw reports:

- [CPU v1](../experiments/results/search-calibration-cpu-v1.json.gz) — uncompressed
  SHA256 `bcb242ecbd403c47ce908c4c2f9765f886e7ba273c3b04a600f7a33b036b8aa3`.
- [MPS v1](../experiments/results/search-calibration-mps-v1.json.gz) — uncompressed
  SHA256 `e7febf9fc547d11380a35bf0278ef5fb4d2a482fd668dac3c61f2c26cc362085`.

These controls test selection reliability. They do not rescue the previously
failed added-search-value gate or establish a useful new kernel.

## Version 2: pairing improved sensitivity but did not pass reliability

Version 2 ran once per device from frozen revision `2c83e2e`. It interleaved
candidate, incumbent and reference inside each block, required median paired
speedup of at least 1.05, and required all five non-tied pairs to favor the
candidate (`p = 1/32` under the one-sided exact sign test). The fail-closed gate
reconstructed every timing summary, paired ratio, p-value and decision from raw
arrays before calculating the aggregates.

| Device | False promotions | Slowdown detections | Sensitivity gate | Frozen gate |
| --- | ---: | ---: | --- | --- |
| CPU | 1 / 100 | 19 / 20 | Passed | Failed reliability |
| MPS | 2 / 100 | 20 / 20 | Passed | Failed reliability |

Pairing materially improved detection of the artificial slowdown, from 8/20 to
19/20 on CPU and from 17/20 to 20/20 on MPS. It did not meet the predeclared zero
false-promotion requirement. The CPU false promotion was seed 3099 at
`(8,128,512)` with median paired speedup 1.080247. MPS false promotions were seed
3031 at `(8,128,512)` with 1.061495 and seed 3044 at `(1,32,512)` with 1.081122.
Every one had five favorable pairs and sign p-value 0.03125. The single missed CPU
positive control had a large median effect but one reversed pair, so its sign gate
correctly rejected it.

Raw reports:

- [CPU v2](../experiments/results/search-calibration-cpu-v2.json.gz) — uncompressed
  SHA256 `496dc29a223eb4d9dc2c589ced6b4c3117f5f7013241f56d66b624f7465215eb`.
- [MPS v2](../experiments/results/search-calibration-mps-v2.json.gz) — uncompressed
  SHA256 `0cfecb2b32fa36bb82dd35740386b3a0c27c84d75283102cb00dbb8651cf65a7`.

The paired implementation improved measurement design, but version 2 did not
validate automatic promotion. That failure motivated a newly frozen rule with
disjoint seeds; the observed v2 seeds were not reused for validation.

## Version 3: family-wise rule passed both frozen gates

Version 3 was frozen at `9a4e60c` before measurement and run once per device with
fresh disjoint seeds. It kept the 1.05 practical median effect, increased each
trial to 11 balanced paired blocks, and divided alpha 0.05 across the declared
family of 100 null decisions. The resulting per-decision sign threshold was
0.0005; only 11/11 favorable non-tied pairs (`p = 1/2048`) could pass.

| Device | False promotions | Slowdown detections | Per-shape detections | Frozen gate |
| --- | ---: | ---: | --- | --- |
| CPU | 0 / 100 | 19 / 20 | 4/5, 5/5, 5/5, 5/5 | Passed |
| MPS | 0 / 100 | 19 / 20 | 5/5, 4/5, 5/5, 5/5 | Passed |

Both missed positives had a large median effect but one reversed pair, producing
`p = 12/2048`; the family-wise sign gate rejected them as specified. All eight
control-correctness checks passed exactly without input mutation. The gate
independently reconstructed raw orders, summaries, ratios, p-values and decisions.

Raw reports:

- [CPU v3](../experiments/results/search-calibration-cpu-v3.json.gz) — uncompressed
  SHA256 `40ee49475c13d26092004acd16d3155899236a1b735e63f630404e2a229c6301`.
- [MPS v3](../experiments/results/search-calibration-mps-v3.json.gz) — uncompressed
  SHA256 `6b2f5aaac969a0157b0201e306976f9fcde047e89223275dcc9ed697d37f5da5`.

This establishes that the declared decision family met its prospective controls
on one Apple M1 CPU/MPS session. It is not a universal false-promotion rate, CUDA
validation, or evidence that the candidate family adds value over native operators.
The earlier workload added-search-value gate remains failed.
