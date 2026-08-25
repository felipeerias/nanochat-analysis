# I0004 — conclusion

Status: **closed**. Verdict: **refuted**. Evidence level: **reproduced**
(two blind analyses, agreeing to three significant figures).

Runs: [A0001](A0001/result.md) (Claude Code, Opus) and
[A0002](A0002/result.md) (Codex), both blind against protocol commit
`e76859c`.

## The claim does not hold

Shadow-arm `curvature/e_sym_gradient`, per-run median over deep checkpoints —
**both analyses report the same numbers**:

| depth | median | d12 five-seed range |
|---|---|---|
| d12 | 7.75e-7 | 3.68e-7 – 1.55e-6 |
| d14 | 8.81e-7 | |
| d16 | **7.04e-7** | inside the d12 range |

The ordering is non-monotone (d12 < d14 > d16) and d16 sits *inside* the d12
seed band, so both branches of the frozen refutation clause fire.

## The threshold is nowhere near being crossed

The 1e-4 threshold sits in a four-order-of-magnitude gap between the two arms,
and depth does not move either arm across it. All 42 shadow per-run medians
are **53-271x below** the threshold; all 42 native medians are **42-222x
above** it. The largest depth effect anywhere in the declared universe is
2.26x.

The extrapolation the protocol asked for is **not supportable**, and A0001
demonstrated why rather than asserting it: the three-point log-linear fit
slopes *downward* (slope −0.024 per unit depth, R² = 0.18), so it predicts no
crossing at any depth. Sweeping 45 plausible variants — d12 over its five
seeds, d14 and d16 perturbed by the d12 seed spread since both are n=1 — gives
**no crossing at all in 27 of 45**, and the 18 that do cross land anywhere
between depth 32 and 224. An answer set of "never, or somewhere between 32 and
224" is not an estimate, and reporting it as one would have been misleading.

## What the d16 shakedown actually saw

The marginal failures that motivated this investigation are real but local:
the only two `e_sym_gradient` exceedances at any depth are d16 updates 1 and 2
(1.11e-4 and 1.36e-4), at normalized progress 0.0004. Across all 1,290 shadow
values there are seven exceedances, and they do **not** increase with depth:
three at d12, two at d14, two at d16.

There is a genuine start-of-training transient — over the first four deep
checkpoints, d16 reaches 1.01e-4 against a d12 seed median of 1.19e-5, the
largest effect in the analysis at 8.4x. But d14 sits *below* d12, so the
ordering fails again, and these updates lie deep inside the absolute 40-step
warmup where depths are least comparable (`DATASET.md` caveat 3). This is
flagged as exploratory, not as a finding.

## Consequence for the campaign: d18 and d20 will certify

Gradient-direction pass rates **rise** with depth: 83-87% at d12 (25-26 of 30),
87.5% at d14 (28 of 32), 87.9% at d16 (29 of 33). The non-certifying
checkpoints are the same early ones (updates 0, 1, 2, 4) at every depth; at
d16 two of them flip from inconclusive to failed. That is the only
depth-related degradation visible and it costs two checkpoints out of 33.

**No threshold change is indicated.** Deeper runs are worth doing if
gradient-direction curvature is what you need.

I0001's finding extends unchanged to the new depths: random and update
directions pass at **0 of 215 checkpoints across every depth**. Certified
curvature exists along the gradient direction only, at all three scales.

## Where the analyses differed

Only in how they summarized, not in what they measured. A0002 reported the
investigation overall as inconclusive (three shadow channels supported the
trend, three refuted it). A0001 reached refuted by treating
`e_sym_gradient` as the decisive channel, on the grounds that it is the only
direction that ever certifies.

**This exposes a defect in my protocol**, the same class as I0001's: the
decision rule named six channels (`e_sym_*` and `e_lin_*` across three
directions) without saying how to combine them. Both readings were reasonable.

The coordinator's call is **refuted**, for the reason A0001 gives: the three
"supported" channels are `e_sym_random`, `e_lin_random` and `e_lin_update`,
which sit around 1e-6 and would need roughly 130x growth to matter — and the
random and update directions never certify at any depth regardless. A0001 also
notes the two `e_lin` ratios bracket sqrt(16/12), which looks like an fp32
arithmetic floor scaling with dimension rather than an observable.

## Recommendations taken from A0001

1. Report acceptance separately for the first five updates and for the rest.
   They are different regimes, and pooling them hides the only real transient.
2. Budget more than one seed per depth in future sweeps. With n=1 at d14 and
   d16, every cross-depth number here borrows its error bar from d12.
