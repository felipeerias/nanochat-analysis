---
id: I0005
kind: question
status: open
data: sweep; the five d12 segments (d14/d16 may be described but not pooled)
selection: shadow_fp32 arm ONLY; gradient direction ONLY; only checkpoints
  where curvature/verdict_code_gradient == 0 (passed). I0001 established that
  the random and update directions never pass, so they are out of scope here
  and must be reported as unavailable rather than silently omitted.
universe: curvature/gHg, curvature/eta_star, curvature/dhd,
  curvature/vhv_gradient, curvature/e_curv_gradient, plus any other certified
  scalar available at those checkpoints
allowed inputs: DATASET.md; the five d12 segments; ../../loader/;
  ../../profiles/; ../0001-seed-variation/conclusion.md
---

## Question

What does certified curvature do over the course of training?

## Test

Describe the trajectory of each quantity against `normalized_progress`, across
the five seeds. Report the across-seed band at each checkpoint, not only the
median.

Distinguish two claims and label which you are making:
- the SHAPE of a trajectory within a run (seed noise does not limit this);
- a COMPARISON of values between runs or between depths (limited by I0001:
  gHg 29%, eta* 25%, dhd 13% at one standard deviation).

Note that eta* is a ratio whose denominator can approach zero. The instrument
applies a reliable-sign gate; report how many checkpoints are excluded by it
and do not interpolate across them.

## Output

The trajectories, with seed bands and the count of certified points. A plain
statement of what can and cannot be concluded about sharpening over training,
given that only 25 of 30 checkpoints per run are certified and only along the
gradient direction.
