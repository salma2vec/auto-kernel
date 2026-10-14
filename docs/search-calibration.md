# Search-decision calibration: both frozen versions failed

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

The paired implementation is retained as a falsifiable improvement to measurement
design, but its automatic promotion decision remains unvalidated. A subsequent
rule needs a newly frozen, disjoint calibration; these observed seeds cannot be
reused to tune and then validate it.
