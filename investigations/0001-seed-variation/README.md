---
id: I0001
kind: question
status: closed
data: sweep-d12-d16-v1; segments d12-s7, d12-s8, d12-s9, d12-s10, d12-s11
selection: defined rows only; matched normalized_progress; both acceptance arms
  reported separately; curvature restricted to per-direction passing verdicts
universe: every metric family present in the continuous, periodic and sparse
  tiers of these five runs; all are reported, none are selected out
allowed inputs: ../../../telemetry-data/sweep/DATASET.md; the five d12
  segments; ../../loader/; profiles/ of these five runs (cite the commit)
---

## Question

How much does a d12 result vary between seeds, for each metric family?

The five runs differ only in seed. They share depth, width, recipe, schedule
and telemetry settings. Any difference between them is seed variation: a
different initialization, a different data order, a different frozen probe,
and the platform's own non-determinism.

## Test

For each metric family:

1. Align the five runs on `normalized_progress`, not on step.
2. At each aligned point, compute the spread across the five runs. Report both
   an absolute spread and a relative one (spread divided by the median value).
3. Summarize each family by its typical relative spread over training, and by
   its worst point.
4. Rank all families by relative spread.

Report every family. Do not drop families that look uninteresting. The ranking
is the product, so a partial ranking is a wrong answer.

Note the known asymmetries before you start:
- Probes are drawn per seed, so probe-based families include probe sampling
  variance as well as trajectory variance. Say which families this affects.
- The native bf16 acceptance arm is uncertified everywhere. Report its spread,
  but do not present its values as measurements.
- Warmup is a fixed number of steps, so the early fraction of training is not
  comparable to the late fraction. Report early and late separately.

## Output

A table of every metric family with its relative seed spread, ordered. A short
list of families whose spread is small enough to detect a change, and a short
list of families too noisy to be useful. This becomes the reference that every
later comparative claim cites.
