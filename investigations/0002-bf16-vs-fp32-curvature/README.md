---
id: I0002
kind: question
status: open
data: sweep; all seven schema-v3 segments (d12-s7..s11, d14-s7, d16-s7)
selection: deep checkpoints where BOTH arms have a defined value for the same
  metric at the same step in the same run; paired by (segment, step, metric)
universe: every curvature/* and update/* scalar family present in both arms
allowed inputs: DATASET.md; the seven v3 segments; ../../loader/;
  ../../profiles/; ../0001-seed-variation/conclusion.md
---

## Question

How much does bf16 arithmetic distort measured curvature, compared with the
IEEE fp32 shadow measured on the same model state?

## Why this is the strongest comparison available

Both arms measure the SAME parameters at the SAME checkpoint. The comparison
is paired, so seed variation cancels. It does not need the I0001 reference and
is not limited by it. Cite I0001 only if you make an unpaired claim.

## Test

1. For each paired (run, step, metric), compute the signed relative difference
   `(native - shadow) / |shadow|`, and also the absolute difference.
2. Summarize per metric: median, interquartile range, and the fraction of
   pairs where the two arms differ in SIGN.
3. Report whether the distortion changes over training (against
   `normalized_progress`) and whether it changes with depth.
4. Use the median as the typical value and state it. Report the distribution,
   not only a point estimate.

Do not condition on verdicts for this test: the native arm fails everywhere,
so conditioning would empty the comparison. State this in your result.

## Output

A table of every paired metric with its typical relative distortion, its
spread, and its sign-disagreement rate. A statement about whether distortion
grows over training and with depth.
