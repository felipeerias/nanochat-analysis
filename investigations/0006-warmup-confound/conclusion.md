# I0006 — conclusion

Status: **closed**. Evidence level: **reproduced** (two blind analyses agree on
the direction and on the structural limits; they differ on counts because they
chose different thresholds).

Runs: [A0001](A0001/result.md) (Claude Code, Opus) and
[A0002](A0002/result.md) (Codex), both blind against protocol commit
`e76859c`.

**This is the most restrictive result in the project so far. Read it before
making any cross-depth claim.**

## The recipe is a hybrid schedule, so no single alignment is correct

A0001 established this by direct measurement, and it reframes the question I
asked. The three schedule components align on three different axes:

- **Warmups are absolute.** `optim/lr` and `optim/momentum` are *bit-identical*
  between d12 and d16 at every step from 0 to 882.
- **The warmdown onset is proportional.** It begins at step 882/1316/1882,
  which is progress 0.3500/0.3501/0.3501 — the same fraction at every depth.
- **Weight decay follows neither**, beginning at progress 0.72/0.66/0.58.

So my premise — that the warmup window is the confound — was too narrow. The
dominant distortion is the choice of x-axis itself, and neither axis is right
everywhere.

There is a second asymmetry A0001 found that I had not considered:
**absolute-step alignment also holds the data fixed.** All seven runs share a
total batch size of 524,288 and identical `batch/*` values at every one of the
2,520 common steps, so the token stream is the same run to run. At equal
*progress*, d16 has therefore seen **2.13x the tokens**. Equal progress means
equal fraction of the recipe, explicitly not equal data.

## How much the axis matters

For `loss/train_mean`, I0001's best detector at 0.06% seed spread, the d12-to-d16
difference late in training is **+0.05% on the step axis and −8.25% on the
progress axis**. The **sign** of the difference flips between alignments at
32.6% of shared points.

A0002 measured the same effect as a median alignment disagreement of 21.05%
inside the window versus 5.75% after — 12.4 seed standard deviations.

Both analyses find far more warmup-dominated families under progress alignment
than under step alignment (A0001: 20 versus 2; A0002: 36 versus 10), and A0002
notes the two sets barely overlap (4 families in common). A0001 quantified the
instability directly: **51 of 157 testable families change verdict with the
axis**, and almost always in one direction.

Notably, on the schedule-correct axis only **two** families are genuinely
warmup-dominated, and both are native-arm acceptance internals rather than
observables — which is chance-order for 248 families at a 3-sigma threshold.
The window itself is not the problem. The axis is.

## Hard structural limits on cross-depth comparison

Both analyses found these independently, and they are not fixable by analysis:

- The deep-checkpoint geometric prefix is defined in **absolute steps and is
  identical across depths**, so on the progress axis **no deep checkpoint below
  progress 0.05 has a cross-depth counterpart**. A0002 adds that only 22 of 30
  checkpoints match on progress at all, and that the step-40 and step-400
  landmarks are among those excluded.
- Symmetrically, the step axis has no d12 reference beyond step 2,520, so
  **53.1% of d16's training cannot be compared on it**.
- The periodic tier places only **two d16 samples inside step 400**, which
  makes 91 families untestable in the warmup window regardless of axis.

## The deliverable: what is unsafe for depth claims

A0001's list is the usable artifact: **160 of 248 families (135 of 192
dynamics families) are unsafe**, by reason — 91 untestable in the window, 85
alignment-unstable after it, 54 unstable within it, 51 flipping verdict with
the axis, 2 genuinely warmup-dominated.

By prefix: `probe/` 38 of 38, `muon/` 15 of 15, `noise/` 5 of 5, `attn/` 5 of
5, `param/` 2 of 2, `update/` 14 of 18, `curvature/` 42 of 95,
`loss/train_mean` 1 of 1.

**Every observable in I0001's usable-spread table is flagged except
`curvature/eta_star | shadow_fp32`.** That is the sobering headline: this
dataset supports cross-depth claims far more weakly than the seed reference
alone suggested, because the seed floor answers "is the difference bigger than
noise" while this answers "is the difference even well defined".

## The practical rule, from A0001

- Align on **absolute step** for step <= 882. Both the schedule and the token
  stream are matched there.
- Align on **normalized progress** for progress >= 0.159. Equal fraction of the
  recipe, explicitly not equal data.
- **Never** quote a warmup-window cross-depth number for a periodic-tier
  family — there are only two d16 samples in that window.

## Consequences for earlier conclusions

**I0003 (decoherence falls with scale) needs a caveat**, and I have added one
there. `muon/*` families are flagged unsafe here — A0002 measures
`muon/replay_update_relerr` at 92% alignment disagreement inside the window
versus 12.8% after. I0003's matched range began at progress 0.05, which lies
inside the warmup window at d12. Its A0001 did verify that dropping the two
earliest matched points preserves the verdict at a similar effect size, so the
finding is likely to survive, but it is now a weaker claim than it read as.

**I0005 (sharpening locked to warmdown) is strengthened, not weakened.** The
warmdown onset is proportional across depths, so the phenomenon sits on the
axis where depths are genuinely comparable, and entirely outside the warmup
window.

**I0004 (acceptance does not degrade with depth) is unaffected** in its
verdict, and its refusal to treat the early-window transient as a finding now
looks well judged: those checkpoints are exactly where cross-depth comparison
is least defined.

## Where the analyses differ

Only in counts, driven by threshold choices the protocol did not specify —
A0001 chose a 3-sigma z-test and reported 248 families; A0002 used its own
criteria over 263. Both disclosed the choice. The ordering (progress alignment
produces far more warmup-dominated families than step alignment) and the
structural limits are identical.

A0001 also disclosed that its extra unsafety criteria were introduced after
observing that only two families are warmup-dominated in the narrow sense, and
labelled them exploratory. That is the right handling, and it is why the
headline number of 160 should be read as a conservative screen rather than a
significance test.
