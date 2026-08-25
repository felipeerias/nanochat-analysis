# I0005 — conclusion

Status: **closed**. Evidence level: **reproduced** (two blind analyses agree on
the selection exactly and on the trajectory shape).

Runs: [A0001](A0001/result.md) (Claude Code, Opus) and
[A0002](A0002/result.md) (Codex), both blind against protocol commit
`e76859c`.

## Sharpening is real, certified — and locked to the learning-rate warmdown

This is the headline, and it changes the story rather than confirming it.

A0001 found that `gHg` is **flat while the learning rate is flat**, then rises
about **15x** during the warmdown phase, then plateaus. The warmdown begins at
normalized progress 0.350 at all three depths. Before it, the five seeds show
no agreed direction (Spearman −0.26 to +0.28, only 3 of 5 positive). During
it, all five agree (+0.62 to +0.95).

A0002 measured the same phenomenon over a window straddling that boundary
(progress 0.30 to 0.75) and reports `vhv_gradient` rising 2.98–6.61x and
`gHg` ending 9.31–13.87x higher. The two are consistent; A0001's phase split
is what reveals *when* the rise happens.

So the honest description is **not** "curvature increases progressively
throughout training". It is: curvature is flat under a constant learning rate,
and sharpens sharply once the learning rate begins to decay. That is a
different claim, with a different mechanism, and it is the kind of statement
the earlier uncertified reading could not have supported.

Both analyses agree the rise greatly exceeds the seed floor, so this is a
legitimate comparison claim and not only a shape claim.

## The selection agrees exactly

| | A0001 | A0002 |
|---|---|---|
| gradient direction certified | 129/150 | 129/150 |
| common to all five seeds | 25 | 25 |
| random direction | 0/150 | 0/150 |
| update direction | 0/150 | 0/150 |
| eta* sign gate exclusions | 12/150, none certified | 12/150, none certified |

The 21 uncertified checkpoints are all at the head of training (update indices
0, 1, 2, 4), which is consistent with what I0004 found about early-window
behaviour at every depth.

## The precise diagnosis of the earlier error

A0001 reproduced the mistake that started this line of work and identified
exactly which safeguard fails to catch it.

At 9 of the 21 uncertified head checkpoints, `gHg` is small but **positive**
(9.6e-4 to 5.9e-3). The reliable-sign gate that was added to `eta_star` for
exactly this purpose therefore **passes**, and eta* is reported at values
between **754 and 4,657** — four decades above the entire certified range of
0.061 to 1.44.

The safeguard that removes them is not the sign gate but the **per-direction
acceptance verdict**. This matters as an instrument lesson: the sign gate
prevents a zero-denominator artifact, but it cannot tell a genuinely tiny
curvature from a meaningless one. Any analysis of eta* must filter on the
direction verdict, not merely on definedness.

The original claim of eta* falling from 1079 was precisely one of these
points. The certified fall is real but far smaller: eta* ends at 0.21–0.27x
its starting value, roughly a factor of four, not a factor of thousands.

## Two instrument observations worth keeping

**eta\* and `vhv_gradient` are the same channel.** A0001 verified that eta* is
the exact reciprocal of `vhv_gradient` to 1.2e-15. This follows algebraically
— eta* = gg/gHg and vhv along the normalized gradient is gHg/gg — but it means
the two are reported as if they were independent observables and should not be
treated as corroborating each other.

**`dhd`'s dramatic fall is mostly the shrinking update, not the landscape.**
Both analyses see `dhd` fall by hundreds of times (A0002: 385–606x; A0001:
1600–4500x). A0001 decomposed it: normalized by update size,
`dhd/||delta||^2 = vhv_update` falls only about 2x across warmdown. The rest
is simply that updates get smaller as the learning rate decays. Quoting the
raw fall as a curvature result would be misleading.

A0001 also flags a genuine defect in my protocol: `dhd` is selected by the
**gradient** verdict but measures curvature along the **update** direction,
whose verdict never passes. The protocol's clause required this pairing.
Flagged rather than silently dropped, correctly.

## An unplanned validation of I0001

A0001 computed its own across-seed spread on the certified set and got
**27.6% for gHg, 23.9% for eta*, 13.2% for dhd** — against I0001's independent
figures of 29%, 25% and 13%, derived by a different method on a different
selection. That is a clean consistency check on the seed reference itself.

Applying it: plateau levels differ between seeds by only 15–19%, which is
*inside* the reference band. **No run can be called sharper than another.**
The trajectory is a within-run fact; the level is not distinguishable across
seeds.

## Depth, described but not claimed

Both analyses report the same shape at d14 and d16 with larger ratios (A0001:
49.6x and 34.3x for gHg; warmdown Spearman +0.72 and +0.74). Both correctly
refuse to make it a finding: n=1 per depth, no seed reference above d12, and
caveats 1 to 3 all apply. It is a hypothesis for a future sweep.

## Follow-up

The warmdown lock is the interesting thread. It suggests the sharpening is a
response to the learning-rate schedule rather than an intrinsic property of
training progress — which is testable directly, with a d12 run using a
different warmdown ratio, at a cost of about one hour and a few dollars.
