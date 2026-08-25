---
id: I0004
kind: hypothesis
status: open
data: sweep; all seven schema-v3 segments
selection: shadow_fp32 arm; curvature/e_sym_* and curvature/e_lin_* at deep
  checkpoints; defined rows only; report each direction separately
universe: the e_sym and e_lin families for all three probe directions, both
  arms reported, shadow arm used for the decision
allowed inputs: DATASET.md; the seven v3 segments; ../../loader/;
  ../../profiles/; ../0001-seed-variation/conclusion.md
---

## Claim to test

The acceptance suite's self-consistency errors in the IEEE fp32 shadow arm
grow with model depth.

This matters for the campaign, not only for the science. The threshold is
1e-4. A short d16 shakedown showed `e_sym` values of 1.1-1.35e-4, marginally
over it, while d12 sits well below. If this is a depth trend, larger models
may not be measurable with the current thresholds.

## Decision rule, fixed before looking

- **Supported**: the d16 median exceeds the maximum across the five d12 seeds,
  and the ordering d12 < d14 < d16 holds for the median.
- **Refuted**: the d16 median lies inside the d12 five-seed range.
- **Inconclusive**: anything else, including non-monotone orderings.

## Also required

Estimate, with an explicit statement of how uncertain the extrapolation is, at
what depth the median `e_sym_gradient` would cross the 1e-4 threshold. Three
depths is a very short lever arm: say plainly if the extrapolation is not
supportable rather than fitting a line to three points and reporting it as
fact.

## Output

The decision, the per-depth values with the d12 seed band, and a
recommendation on whether d18/d20 runs would produce certifiable curvature.
