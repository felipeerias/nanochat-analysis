# X01 — is the warmdown curvature rise a power law?

Exploratory, single analysis, 2026-08-25. Not a conclusion.

## Why

Two candidate models for I0005's finding that curvature rises during the
learning-rate warmdown:

- **Power law** (Claude): smaller norm-controlled steps let the iterate settle
  into a sharper region, giving `lambda ~ lr^-alpha`. A single ratio (LR falls
  20x, normalized curvature rises 4.76x) implies alpha = 0.52 — but with one
  ratio, alpha is fitted, not tested.
- **Saturated schedule response** (Codex, T01): the apparent exponent is an
  artifact of early saturation.

These differ in the SHAPE of log(curvature) against log(LR) within a single
warmdown: a power law is a straight line, saturation bends.

The warmdown is a linear LR ramp, so the five existing d12 runs already
contain that sweep. No new data needed.

## Method

Certified shadow-arm gradient-direction `curvature/vhv_gradient` at deep
checkpoints with update index >= 882 (the warmdown onset), against the
`optim/lr` multiplier at the same update. 14 certified points per run, five
runs. Linear and quadratic fits in log-log, plus separate early-half and
late-half slopes.

## Result

| run | n | pooled alpha | R^2 (linear) | quadratic term | early alpha | late alpha |
|---|---|---|---|---|---|---|
| d12-s7 | 14 | 0.104 | 0.119 | -0.164 | 0.57 | -0.04 |
| d12-s8 | 14 | 0.370 | 0.447 | -0.354 | 2.26 | 0.03 |
| d12-s9 | 14 | 0.312 | 0.259 | -0.409 | 2.77 | -0.06 |
| d12-s10 | 14 | 0.304 | 0.411 | -0.264 | 1.64 | 0.04 |
| d12-s11 | 14 | 0.293 | 0.234 | -0.403 | 2.33 | -0.12 |

mean pooled alpha 0.277 (sd 0.101); mean quadratic term -0.319 (sd 0.104).

**The power law is refuted.** A straight line fits badly (R^2 0.12-0.45), the
quadratic term is negative in 5 of 5 runs, and the slope collapses from about
1.9 early in the warmdown to about 0 late. Curvature rises steeply as the
learning rate first decays, then stops responding.

The single-ratio alpha of 0.52 was an average over two different regimes and
should not be quoted.

## Independent cross-check

The T01 theory session fitted the same quantity from the same certified table,
with its own implementation and without seeing this analysis. It reports a
pooled straight-power estimate of **alpha = 0.277**, identical to the value
above to three digits, and a free two-slope fit with a late slope of 0.021,
matching the slope collapse reported here. It adds a hinge fit that this
analysis did not run: break at log r = 0.374, or a learning-rate multiplier of
about 0.688, with leave-one-seed-out error less than half the straight model's.
Four of five held-out runs favour the hinge; `d12-s7` is the exception.

Two independent implementations agreeing on alpha, and agreeing that the late
slope is indistinguishable from zero, is a genuine cross-check. It is still the
same five runs and the same probe, so it removes implementation error, not the
measurement's limits.

Per-run break estimates span log r of 0.238 to 0.576, so the location of the
bend is **not** well determined even though its existence is consistent across
runs.

## What this does not establish

The mechanism. Saturation is consistent with the data but this analysis does
not identify what saturates. Neither this analysis nor T01's derives the bend
location; `r_s` is fitted, not predicted.

Nor does it establish that the learning rate is the cause. The learning-rate
multiplier, normalized progress and the Muon momentum are collinear in every
run collected so far, because `get_muon_momentum` recomputes its window from
the same `--warmdown-ratio` that drives the learning-rate warmdown
(`scripts/base_train.py:378`). Saturation in the learning rate and saturation
in progress or momentum are the same curve in this dataset.

It also inherits every curvature caveat: one 256-token probe, gradient
direction only, and 14 points per run.

Code: `X01-warmdown-exponent.py`.
