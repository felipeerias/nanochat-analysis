# I0002 — conclusion

Status: **closed**. Evidence level: **reproduced** (two blind analyses, agreeing
on the central result and on the pair count exactly).

Runs: [A0001](A0001/result.md) (Claude Code, Opus) and
[A0002](A0002/result.md) (Codex), both blind against protocol commit
`e76859c`. Both analysed 11,366 both-arms-defined pairs over 215 deep
checkpoints in seven runs.

## bf16 does not corrupt curvature values. It destroys the error bars.

This is the finding, and both analyses reached it independently.

**The curvature values themselves are barely distorted.** Across the
quadratic-form families, A0001 measured a median absolute relative difference
of **0.34%** (95% CI 0.29–0.38%) with a signed median of −0.004%, meaning the
bf16 measurement is essentially *unbiased*. A0002 measured the same population
at **0.21–0.94%**. Both report **zero sign disagreements** — A0001 across all
1,906 quadratic-form pairs.

**The apparatus that validates those values is annihilated.** The symmetry and
linearity residuals that the acceptance suite tests against are
**6,800–14,500x larger** in bf16, and probe signal-to-noise ratios collapse by
99.8%. Nine families (`e_curv_*`, `e_fd_*`, `fd_cos_*`, all directions) are
not merely distorted but **undefined in bf16 at all 215 checkpoints**, marked
`noise_floor_inconclusive`, while fp32 defines them.

A0002 quantified the same thing from the finite-difference side: FD curvature
distortion is 3.94% along the gradient direction but **688x along random
directions and 1,854x along update directions**, with sign agreement at those
directions no better than a coin flip (53.0% and 49.8% disagreement).

So the mechanism behind `DATASET.md` caveat 4 is now precise. The native arm
fails its self-consistency checks not because the Hessian-vector products are
wrong in bf16, but because the finite-difference cross-checks used to
*validate* them are unresolvable at bf16 precision — catastrophically so along
directions that have almost no curvature to measure.

Both analyses noted a clean confirmation that the pairing is real: `arith_eps`
differs between the arms by **exactly 65,536 = 2^16** at all 215 checkpoints
(the ratio of bf16 to fp32 machine epsilon), and `update/direction_norm` is
**bit-identical** across arms at all 215 pairs.

## What this means for using the data

Native-arm curvature *values* are considerably better than their uncertified
status suggests: unbiased, sub-1%, no sign flips. But "uncertified" still
means what it says — the checks that would catch a genuine operator error are
the ones that cannot run at bf16. The shadow arm remains the one to quote.
The reasonable use of the native arm is as a *description of what the
optimizer's own arithmetic sees*, which is a different and legitimate question
from what the loss surface actually looks like.

## Update effectiveness is the real casualty, and it degrades over training

A0001 found this by splitting the universe into classes, where A0002's pooled
median (0.009 correlation with progress; 25 families up, 21 down) hides it.
The class split is the more informative treatment.

The `update/*` families — the local quadratic model of how much an update
should have reduced the loss — **grow worse over training, unanimously**:
positive correlation with progress in 7 of 7 runs for all 8 non-degenerate
families, median late-versus-early ratio **5.1x**, and `residual_p2` at 34x.

The mechanism A0001 identified is catastrophic cancellation, and it is
convincing: `loss_before` and `loss_after` are each distorted by only about
0.025%, but their *difference* shrinks roughly 9x over training while the
absolute error in it stays flat. The consequence is concrete: at **7 of 215
checkpoints (3.3%), the two arms disagree about the sign of the achieved loss
decrease**, and these are concentrated late in training.

Update-effectiveness records from the native arm should therefore be treated
as unreliable late in a run, independent of the curvature question.

## Depth: no effect claimed

Both analyses declined, which is the right call. A0001 measured quadratic-form
distortion rising 0.309% → 0.366% → 0.414% from d12 to d16 (a 1.34x ratio),
then checked it against the five d12 seeds, which alone span 0.274–0.402%.
That puts d16 at **1.84 standard deviations**, below the 2–3x bar from I0001,
so no claim. A0002 reached the same place from a different direction: a median
d16/d12 ratio of exactly 1.000, with 26 families higher and 18 lower.

## Where the analyses differ

Only in universe construction and framing, not in results.

A0001 counted 62 scalar families of 100 total (36 vector families excluded per
the protocol's "scalar" wording, which it flagged as a genuine coverage gap);
A0002 counted 53. Both reported their entire declared universe including
families that yielded no usable pairs.

On training-time growth they appear to disagree — A0002 says no uniform
growth, A0001 says strong unanimous growth — but they are describing different
populations. A0002's statement is about the pooled median across all families;
A0001's is about the `update/*` class specifically, which A0002's pooling
averages away against the curvature families that do *not* grow. Both
underlying facts are correct and the class split is the useful one.

## Follow-up this suggests

The 36 vector families (the epsilon sweeps) were excluded by both analyses on
the protocol's wording. They are exactly where the finite-difference collapse
would be visible in detail, so a targeted look at them would sharpen the
mechanism — though it would not change the conclusion.
