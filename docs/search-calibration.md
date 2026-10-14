# Search-decision calibration: version 1 failed

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
