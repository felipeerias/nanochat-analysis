# I0003 — conclusion

Status: **closed**. Verdict: **supported**, with one arm marginal.
Evidence level: **reproduced** (two blind analyses agree on direction,
magnitude and structure).

Runs: [A0001](A0001/result.md) (Claude Code, Opus) and
[A0002](A0002/result.md) (Codex), both blind against protocol commit
`e76859c`, both citing `I0001/conclusion.md@4ac11f3`.

## Muon decoherence decreases as the models get larger

| | A0001 | A0002 |
|---|---|---|
| d14 offset vs d12 | −6.61% | −6.44% |
| d16 offset vs d12 | −11.23% | −12.65% |
| outside the d12 seed range | 17/20 and 20/20 | 21/30 and 26/30 |

Two independently written pipelines, different checkpoint-matching choices
(20 versus 30 matched points), and the effect sizes agree to about one
percentage point. Absolute levels over the matched tail are 4.24% at d12,
3.89% at d14, 3.62% at d16.

Against the seed floor from I0001 (3.5% relative standard deviation for this
channel), **d16's −11% is 3.2 standard deviations and clears the 2-3x bar;
d14's −6.6% is 1.9x and is marginal**. The direction is consistent at both
depths.

A0001 disclosed an ambiguity in my phrase "in a consistent direction": counting
excursions that share a direction gives supported for both depths, while
requiring zero excursions the other way leaves d14 short by exactly one
checkpoint of twenty. The strict parse still returns supported for d14 under
mean and geometric-mean summaries and fails only under the median. Refuted is
nowhere close on either parse.

## The structure is about parameter role, not depth position

Both analyses found the same thing, by different routes. Role explains the
overwhelming majority of the variation between matrices — A0001 puts it at
80-83% of within-checkpoint log variance, A0002 at 87.8% — while relative
position in the network explains almost nothing (A0001: 5-7%; A0002: 0.4%).

The role ordering is essentially identical at all three depths (rank
correlation +0.96 to +1.00): `mlp_out` lowest, then `attn_k`, then `attn_q`
and `mlp_in`, then `attn_out`, `attn_v`, and `ve_gate` highest, spanning
roughly 0.84x to 2.08x each depth's own median.

A0001 also found and dismissed a trap: a clean odd/even sawtooth in the raw
per-layer profile is a `ve_gate` parity artifact and vanishes once role is
controlled for.

## The initialization zeros behave identically at every depth

Both analyses independently report that **69.2% of matrices** have exactly
zero decoherence at the first update — everything upstream of the zero-init
output projections — and that this fraction is the same at all three depths.

A0001 quantified what averaging them in would have cost: over active matrices
only, update-0 decoherence is about 0.123 at all three depths with no depth
effect at all, whereas blind averaging reports 0.0388 and a median of exactly
zero. That is a 3.25x understatement produced purely by a handling choice.

## The largest caveat: this may be width, not depth

A0001 makes the point sharply and it should govern how the result is quoted.
`DATASET.md` caveat 1 applies with full force: width co-varies with depth
(768, 896, 1024), and the within-depth correlation between matrix shape and
decoherence points at **width** as the plausible mechanism rather than depth.
Every shape class drops with depth (attention 0.877x, MLP 0.847x, `ve_gate`
0.916x at d16), so no single class carries the effect.

The honest statement is: **decoherence falls as the nanochat recipe scales
up**, not "depth causes decoherence to fall". Separating the two would need a
run that varies width at fixed depth, which this dataset does not contain.

Neither d14 nor d16 has an error bar of its own — both are n=1, borrowing
their uncertainty from d12's five seeds. That limitation is unavoidable here
and is the strongest argument for the multi-seed recommendation in I0004.

## One disagreement worth recording

A0002 reported that matrix shape was untestable because all 18,044 selected
rows have a null `shape` column. A0001 tested shape anyway and found a
correlation (larger `min(m, n)` gives lower decoherence, about −0.42),
evidently deriving the shape from parameter identity rather than the column.

Both are correct about what they did. The column is indeed unpopulated for
these records, which is worth knowing as an instrument fact, and shape can be
recovered from the parameter name. A0001's route gives the more informative
answer, and it is also the route that surfaced the width confound above.

## Follow-up this suggests

The effect is real, reproducible and modest, and it is confounded with width
by construction. If it matters, the experiment that resolves it is a
fixed-depth width sweep at d12 — which is cheap, and which the d12
configuration is already characterized for.

## Caveat added after I0006 (2026-08-25)

I0006 found that `muon/*` families are unsafe for depth claims: the choice
between absolute-step and normalized-progress alignment changes cross-depth
differences substantially, and `muon/replay_update_relerr` shows about 92%
alignment disagreement inside the warmup window against 13% after it.

This investigation's matched range began at progress 0.05, which lies inside
the warmup window at d12 (step 400 is progress 0.159 there). A0001 did check
that dropping the two earliest matched points preserves the verdict at a
similar effect size (d14 −5.0%, d16 −11.2%), so the finding is likely to
survive.

It should nevertheless now be quoted as the weaker claim: decoherence appears
to fall as the recipe scales up, measured on the progress axis, over a range
that partially overlaps the warmup window, with width and depth confounded.
Settling it properly needs the fixed-depth width sweep suggested above, run on
the alignment rule I0006 sets out.
