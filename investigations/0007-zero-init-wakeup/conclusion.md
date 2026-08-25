# I0007 — conclusion

Status: **closed**. Evidence level: **reproduced** — the closest agreement of
any investigation so far. Both analyses independently recovered the same
partition, the same fractions, and the same perfect seed stability.

Runs: [A0001](A0001/result.md) (Claude Code, Opus) and
[A0002](A0002/result.md) (Codex), both blind against protocol commit
`e76859c`.

## There is no interesting order. There are two tiers, and everything is awake by the second update.

| tier | roles | wake-up update |
|---|---|---|
| A | `attn_out`, `mlp_out` — exactly the zero-initialized projections | 0 |
| B | `attn_q`, `attn_k`, `attn_v`, `mlp_in`, `ve_gate` | 1 |

**No matrix anywhere has a wake-up index of 2 or later.** Since 0 and 1 are
consecutive updates, this is not an artifact of the checkpoint grid — there is
no finer grid to look at.

Tier B is **69.23%** of matrices at every depth (54 of 78, 63 of 91, 72 of
104), reported to that precision by both analyses independently. As A0001
notes, that is exactly 9/13 for any even layer count, not a coincidence.

## Exact zero is categorical here, not a threshold choice

A0001 tested on the IEEE-754 bit pattern; A0002 used literal equality. Both
land in the same place, and A0001 showed why the choice cannot matter: the
distribution at update 0 is bimodal with an empty interval. The smallest
nonzero decoherence at update 0 is 0.108–0.115, while the smallest nonzero
value anywhere in the dataset is 3.2e-3. No tolerance could move a cell
between states.

Corroboration is unusually thorough. A0001 confirmed the tier-B set against
five independent channels (`grad/norm`, `grad/rms`, `muon/data_norm`,
`muon/u_final_norm_observed` all exactly zero; `muon/cos_raw_final` honestly
undefined with reason `degenerate_or_zero`), and noted the complementary fact
that `muon/decay_norm` is exactly zero on precisely **tier A** — those
parameters *are* zero, so weight decay has nothing to act on. A0002 confirmed
that decoherence and probe-gradient wake times agree for **585 of 585**
run-matrix instances.

## The ordering is perfectly stable across seeds

All 78 d12 matrices have the same wake-up index in all five seeds: **100%
agreement**, every pair, Kendall tau-b of exactly 1.000. The spread is exactly
zero.

That places wake-up ordering alongside the configuration channels in I0001,
far outside the 3.5% seed floor that reference gives decoherence *magnitude*.
The ordering is structural; only the magnitude is stochastic.

## Depth governs nothing, and the mechanism explains why

Wake-up sits at the same **absolute** update index at every depth, and is
constant within each role, so relative position in the network explains
nothing (A0002 verified the rule holds for 273 of 273 matrices across all
three depths).

A0001 gives the mechanism, and it is the most satisfying part of this result:
wake-up is governed by **graph distance to the nearest zero-initialized
projection, which is 1 for every affected matrix**, because each block's
`c_proj` gates only its own block. Deepening the network adds gates **in
parallel, not in series**. That is why depth cannot produce a propagation
wave here — there is nothing for a wave to travel through.

In normalized progress the wake-up point does differ by 2.1x across depths
(3.97e-4 at d12 versus 1.86e-4 at d16), purely because the same absolute step
is a different fraction of a longer run. This is the same absolute-versus-
relative tension that I0006 is examining.

## Exploratory: the graded structure is in the magnitude, not the timing

A0001 flagged this as exploratory because the pre-registered test returned a
degenerate two-valued ordering. At update 1 the *magnitudes* reproduce
strongly across seeds (Spearman 0.971–0.985), and after removing role medians
the residual per-matrix ordering still reproduces at about 0.70 and correlates
with relative depth at −0.41 to −0.54. d14 and d16 both fall inside the d12
five-seed range, so no depth effect is distinguishable at n=5.

## An instrument gap both analyses found

The periodic-tier Muon stage families are not emitted at the early deep
checkpoints — only at step 0 and then at the periodic cadence (step 101 at
d12). So the early window that the Pythia geometric prefix was designed to
resolve is covered by the sparse tier but not by the periodic one.

Both analysts documented this rather than interpolating across it. A0001
closed the resulting gap by inference from the Muon replay rather than by
assumption, and flagged that inference as the single non-measured step in its
result. This is worth recording as a limitation of the instrument: if early
Muon stage geometry matters in a future run, the periodic tier would need to
emit on the deep schedule too.
