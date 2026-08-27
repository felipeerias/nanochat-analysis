---
investigation: I0004
analyst: A0001
design: confirmatory         # frozen protocol; one exploratory addendum, labelled
outcome: refuted             # headline channel; 3 of 6 shadow channels came out supported — see Result
saw: |
  investigations/0004-acceptance-vs-depth/README.md@e76859c (protocol);
  analysis/README.md@e76859c (procedure);
  investigations/0001-seed-variation/conclusion.md@4ac11f3 (seed reference);
  loader/telemetry_load.py@c0419ef;
  telemetry-data/sweep/DATASET.md (not under git; local copy read 2026-08-25);
  nanochat/nanochat/telemetry.py@db916ef, lines 1395-1484 only, to record how
  e_sym/e_lin are defined and how they enter the verdict — the instrument at
  that commit is the one that produced the data (provenance git_sha 5c2fb16
  post-dates it with telemetry.py untouched);
  profiles/ — directory names listed only, no profile.md opened;
  NOT read: any A0002/ material, 0004 conclusion.md, other investigations.
data: sweep; the seven schema-v3 segments (d12-s7/s8/s9/s10/s11, d14-s7,
  d16-s7). Legacy v1 segment d12-iter excluded as the protocol requires.
selection: |
  sparse tier; metric in curvature/{e_sym,e_lin}_{random,gradient,update};
  phase == post_update (deep checkpoints); is_defined == True applied
  explicitly (3870 rows selected — 2580 error + 1290 verdict — and 0 were
  undefined, so the filter removed nothing); acceptance_arm split into
  shadow_fp32 (decisive) and native (reported). Verdict conditioning was NOT
  applied — see Limitations, it would be circular here.
universe: 12 channels tested (2 families x 3 directions x 2 arms), 12 reported.
  Plus curvature/verdict_code_{random,gradient,update} in both arms read as
  context only, not tested.
code: uncommitted at submission (coordinator commits):
  investigations/0004-acceptance-vs-depth/A0001/{extract,analyze,tails,early,
  seedmatched,extrapolate,figures}.py
seed_reference: investigations/0001-seed-variation/conclusion.md@4ac11f3;
  dispersion statistic used = standard deviation relative to the median
  (I0001's canonical figure), recomputed on this investigation's own
  per-run-median statistic rather than taken from the table, because I0001
  does not tabulate e_sym/e_lin.
supersedes: none
---

## Result

**Decision on the headline channel: REFUTED.** For shadow-fp32
`curvature/e_sym_gradient`, the d16 per-run median is **7.04e-7**, which lies
inside the d12 five-seed range **[3.68e-7, 1.55e-6]**. The ordering is also
non-monotone (d12 median 7.75e-7 < d14 8.81e-7 > d16 7.04e-7). Both branches
of the protocol's "refuted" clause are satisfied.

The one-sentence version: **the 1e-4 threshold sits in a four-order-of-magnitude
gap between the two acceptance arms, and depth does not move either arm across
it.** Every one of the 42 shadow per-run medians is **53-271x below** 1e-4;
every one of the 42 native per-run medians is **42-222x above** it; the largest
depth effect anywhere in the declared universe is **2.26x**.

### The per-depth values, with the d12 seed band

Per-run median over that run's deep checkpoints (30 at d12, 32 at d14, 33 at
d16). d12 is summarised as median [min, max] over its five seeds; d14 and d16
have one seed each (seed 7).

Shadow-fp32 arm — decisive:

| channel | d12 median [5-seed range] | d14 | d16 | d16/d12 | decision |
|---|---|---|---|---:|---|
| `e_sym_random`   | 8.27e-7 [3.81e-7, 1.14e-6] | 1.27e-6 | 1.87e-6 | 2.26x | **supported** |
| `e_sym_gradient` | 7.75e-7 [3.68e-7, 1.55e-6] | 8.81e-7 | 7.04e-7 | 0.91x | **refuted** |
| `e_sym_update`   | 8.32e-7 [6.13e-7, 1.07e-6] | 1.09e-6 | 9.45e-7 | 1.14x | **refuted** |
| `e_lin_random`   | 7.67e-7 [7.49e-7, 7.74e-7] | 7.75e-7 | 8.72e-7 | 1.14x | **supported** |
| `e_lin_gradient` | 4.96e-7 [4.95e-7, 5.32e-7] | 4.65e-7 | 5.12e-7 | 1.03x | **refuted** |
| `e_lin_update`   | 7.25e-7 [7.15e-7, 7.52e-7] | 8.08e-7 | 8.63e-7 | 1.19x | **supported** |

Native arm — reported, not decisive, and uncertified everywhere (caveat 6):

| channel | d12 median | d14 | d16 | decision |
|---|---|---|---|---|
| `e_sym_random`   | 1.16e-2 | 9.34e-3 | 2.22e-2 | inconclusive |
| `e_sym_gradient` | 5.98e-3 | 4.29e-3 | 1.22e-2 | inconclusive |
| `e_sym_update`   | 1.43e-2 | 1.72e-2 | 9.45e-3 | refuted |
| `e_lin_random`   | 1.13e-2 | 1.04e-2 | 9.96e-3 | inconclusive |
| `e_lin_gradient` | 5.65e-3 | 4.37e-3 | 4.90e-3 | inconclusive |
| `e_lin_update`   | 1.08e-2 | 9.24e-3 | 1.04e-2 | refuted |

Tally over the declared universe: shadow 3 supported / 3 refuted / 0
inconclusive; native 0 supported / 2 refuted / 4 inconclusive.

### Is any of this bigger than seed noise?

Comparing the d16-vs-d12-median effect to the d12 five-seed sd-relative spread
of the same statistic. I0001's rule is that an effect must clear roughly 2-3x
the sd-relative spread.

| shadow channel | d12 sd-relative | d16 effect | effect / sd | clears 2-3x bar? |
|---|---:|---:|---:|---|
| `e_sym_random`   | 41.5% | +126.0% | 3.0 | yes, marginally |
| `e_sym_gradient` | 56.7% |   -9.2% | -0.2 | no |
| `e_sym_update`   | 19.8% |  +13.6% | 0.7 | no |
| `e_lin_random`   |  1.4% |  +13.7% | 9.9 | yes |
| `e_lin_gradient` |  3.2% |   +3.1% | 1.0 | no |
| `e_lin_update`   |  2.2% |  +19.0% | 8.6 | yes |

The three "supported" channels are real: they clear the seed-noise floor by
3-10 sd. They are also tiny (13-126% on a quantity two orders of magnitude
below the threshold) and, for the two `e_lin` ones, look like an arithmetic
floor rather than an observable — the d12 seed spread is 1-3%, i.e. these
channels barely respond to the seed at all, and their d16/d12 ratios
(1.14x, 1.19x) bracket sqrt(16/12) = 1.155. A plausible reading is that fp32
rounding accumulates roughly as the square root of the layer count. That is an
interpretation, not a measurement; nothing here tests it.

The channel the campaign cares about — `e_sym_gradient`, the only direction
that ever certifies — moved **-9.2%** against a **57%** seed floor. There is no
depth effect there to detect.

### Extrapolation to the 1e-4 crossing: not supportable

The protocol asks at what depth the median `e_sym_gradient` crosses 1e-4 and
asks for this to be answered plainly. **It cannot be answered from this data.**

The measured medians are 7.75e-7, 8.81e-7, 7.04e-7 — a sequence with no trend.
A three-point log-linear fit gives slope **-0.024 per unit depth, R2 = 0.18**:
the fitted trend is *downward*, so there is **no crossing at any depth**.
Reaching 1e-4 from 7.75e-7 requires a **129x** increase; the whole observed
d12→d16 change is **-9%**.

The instability is total. Refitting under changes that are all equally
defensible:

| variant | slope /depth | crossing depth |
|---|---:|---|
| d12 = five-seed median (primary) | -0.024 | none |
| d12 = five-seed min | +0.162 | 46 |
| d12 = five-seed max (= seed-7, seed-matched) | -0.197 | none |
| drop d12 (fit d14, d16) | -0.113 | none |
| drop d14 (fit d12, d16) | -0.024 | none |
| drop d16 (fit d12, d14) | +0.065 | 87 |

Choosing a different one of the five d12 seeds flips the sign of the slope.
Because d14 and d16 have one seed each, their points carry an unmeasured seed
error; taking the d12 sd-relative (57%) as a proxy and sweeping all 45
combinations, **27 of 45 variants have no crossing at all**, and the 18 that do
put it between depth **32 and 224**. An estimate whose answer set is
{never, 32, ..., 224} is not an estimate.

For completeness, the same envelope on the three channels that did move
monotonically — none comes anywhere near d18/d20:

| shadow channel | variants that cross | crossing depth (min / median / max) |
|---|---:|---|
| `e_sym_random` | 42/45 | 23 / 33 / 508 |
| `e_lin_random` | 45/45 | 131 / 160 / 199 |
| `e_lin_update` | 45/45 | 106 / 131 / 182 |

The nearest lower bound anywhere in the universe is **depth ~23**, on
`e_sym_random` — a direction that passes at **0 of 215** checkpoints in this
dataset for reasons unrelated to `e_sym`. Nothing projects a median crossing at
d18 or d20.

### Where the shakedown's 1.1-1.35e-4 actually came from — exploratory

This slice was chosen *after* seeing the data and is exploratory, but it
resolves the observation that motivated the investigation.

Across all 1290 shadow checkpoint-channel values, **seven** exceed 1e-4:

| run | depth | channel | update | progress | value |
|---|---|---|---:|---:|---:|
| d12-s10 | 12 | `e_sym_update`   |  630 | 0.250 | 1.37e-4 |
| d12-s11 | 12 | `e_sym_random`   | 1008 | 0.400 | 1.08e-3 |
| d12-s11 | 12 | `e_sym_random`   | 1889 | 0.750 | 5.35e-4 |
| d14-s7  | 14 | `e_sym_random`   | 1879 | 0.500 | 1.20e-4 |
| d14-s7  | 14 | `e_sym_random`   | 3006 | 0.800 | 4.69e-4 |
| d16-s7  | 16 | `e_sym_gradient` |    1 | 3.7e-4 | **1.11e-4** |
| d16-s7  | 16 | `e_sym_gradient` |    2 | 5.6e-4 | **1.36e-4** |

The last two reproduce the shakedown's "1.1-1.35e-4" exactly. They are the
**second and third update of the run** — normalized progress 0.0004. They are
the only two exceedances in all 198 d16 shadow values, and the only
`e_sym_gradient` exceedances at any depth. Exceedances themselves do not
increase with depth: 3 at d12 (over 900 values), 2 at d14 (192), 2 at d16 (198).

There *is* a real signal in the first four deep checkpoints (updates 0, 1, 2, 4),
where `e_sym_gradient` is elevated at every depth and decays away by update 8:

| | d12 (5 seeds) | d14 | d16 |
|---|---|---|---|
| median over first 4 ckpts | 1.19e-5 [4.17e-6, 2.50e-5] | 8.28e-6 | **1.01e-4** |
| max over first 4 ckpts | 2.22e-5 [4.68e-6, 2.88e-5] | 1.26e-5 | **1.36e-4** |

d16 is 8.4x the d12 seed median there — 10 sd on the d12 spread of the same
statistic, and by far the largest effect in this analysis. But **d14 sits below
d12**, so the frozen rule returns *inconclusive* on this slice too, and the two
adjacent-pair slopes it would average have opposite signs (-0.18 for d12→d14,
+1.25 for d14→d16) and differ 6.8x in magnitude. With one seed at d14 and one at d16, "a depth effect that switches on
above d14" and "a d16-run idiosyncrasy" are not separable. The seed-matched
ladder (seed 7 at all three depths) is non-monotone as well: 2.23e-5 → 8.28e-6
→ 1.01e-4.

Excluding those first four checkpoints entirely leaves the run-level picture
unchanged (`e_sym_gradient` still refuted: d12 6.27e-7 [3.03e-7, 1.22e-6],
d14 8.08e-7, d16 5.86e-7).

### Certification availability — the campaign question

Per-direction shadow verdicts, counted over each run's deep checkpoints:

| depth | run(s) | gradient passed | random | update |
|---|---|---|---|---|
| 12 | s7, s8, s10, s11 | 26/30 (86.7%) | 0/30 | 0/30 |
| 12 | s9 | 25/30 (83.3%) | 0/30 | 0/30 |
| 14 | s7 | 28/32 (87.5%) | 0/32 | 0/32 |
| 16 | s7 | 29/33 (87.9%) | 0/33 | 0/33 |

**The gradient-direction pass rate does not fall with depth — it rises slightly,
86.7% → 87.5% → 87.9%.** This extends I0001's finding (certified curvature
exists only along the gradient direction) from d12 to d14 and d16 unchanged:
random and update pass at 0 of 215 checkpoints across all seven runs.

At every depth the non-passing gradient checkpoints are the same four — updates
0, 1, 2, 4. At d12 and d14 they are *inconclusive* (curvature/finite-difference
SNR, not `e_sym`). At d16 two of them are *failed*, because `e_sym` crossed
1e-4. That is the only depth-related degradation visible anywhere, and it costs
two checkpoints out of 33.

**Recommendation: d18 and d20 will produce certifiable curvature.** On this
evidence they should certify along the gradient direction at roughly 85-90% of
deep checkpoints, as d12/d14/d16 all do, with medians ~100x under the
threshold. The threshold is not at risk for the body of a run. The residual risk
is confined to the first ~5 updates, where the d16 elevation is real but
non-monotone in depth; the way to settle that is one d18 run, not an
extrapolation from three points. No threshold change is indicated. Two
tightenings would make the campaign more robust at negligible cost: (a) report
acceptance separately for the first-5-update window and the rest, since they are
different regimes; (b) budget more than one seed at each new depth — with n=1 at
d14 and d16, every cross-depth number in this report has an error bar borrowed
from d12.

### Figures

- `figures/fig1_esym_gradient_trajectories.png` — every shadow `e_sym_gradient`
  value against normalized progress, all seven runs, with the 1e-4 line.
- `figures/fig2_medians_vs_depth.png` — the six shadow channels' per-run medians
  vs depth on a shared axis with the threshold, showing the ~100x gap.
- `figures/fig3_early_transient.png` — the first eight deep checkpoints.
- `figures/fig4_medians_indexed.png` — the same medians indexed to the d12
  five-seed median, so the 1.03x-2.26x effects are visible against the seed band.

Raw outputs are in `analyze.out`, `tails.out`, `early.out`, `seedmatched.out`,
`extrapolate.out`; extracted rows in `rows.csv`, per-channel decisions in
`decisions.csv`.

## Limitations

**Deviations and judgement calls, stated explicitly.**

1. *No verdict conditioning.* The protocol's selection line did not ask for it,
   and applying it here would be circular: `e_sym > 1e-4` is one of the
   conditions that *makes* a direction's verdict `failed`
   (`nanochat/telemetry.py@db916ef:1466-1469`). Filtering to passing verdicts
   would delete precisely the values the hypothesis is about. This resolves, for
   this investigation, the ambiguity I0001 flagged in its closing note.
2. *"Median" required an operational choice.* The protocol says "the d16 median"
   and "the maximum across the five d12 seeds" without saying median of what. I
   took the per-run median over that run's deep checkpoints, then summarised d12
   by the median and the min/max of the five per-run medians. Pooling all 150
   d12 checkpoint values instead would change the d12 centre but not the
   decision, since the d16 median sits mid-band.
3. *Sensitivity checks all agree with the headline.* Restricting to
   normalized_progress >= 0.2, to the last ten checkpoints, or to steps > 5
   leaves `e_sym_gradient` refuted or inconclusive in every case, and never
   supported. `e_lin_update` is supported under all three. Full output in
   `analyze.out` sections 4-5 and `early.out` section D.
4. *The exploratory slice is labelled as such.* The first-four-checkpoint
   analysis was chosen after seeing where the exceedances were. It is
   hypothesis-generating only and would need a fresh run to confirm.

**DATASET.md caveats that bear on this result.**

- *Caveat 1 (size ray, not depth sweep).* Depth co-varies with width, head
  count, batch size, LR, weight decay and horizon. Nothing here says "depth
  causes X"; every statement is about the nanochat recipe at three scales.
- *Caveat 2 (n=3 depths).* This is the central limitation and the reason the
  extrapolation fails. Worse than n=3: d14 and d16 are **n=1 seed each**, so two
  of the three points have no measured error bar at all. The seed-uncertainty
  envelope above imports d12's spread as a proxy, which I0001 explicitly warns
  may not transfer to d14 or d16.
- *Caveat 3 (absolute warmups).* The 40-step LR warmup and 400-step Muon ramp are
  ~16% of d12 but ~7% of d16, and the deep schedule's geometric prefix therefore
  covers different progress fractions at each depth. This lands directly on the
  exploratory finding: the d16 exceedances are at updates 1 and 2, deep inside
  the warmup, where the depths are least comparable. It is the reason I do not
  read the early transient as a depth effect. I0006 is the investigation for
  this.
- *Caveat 4 (single-sequence curvature).* Every acceptance result is local to
  the one 256-token sequence used by the HVP path, not the training objective.
- *Caveat 6 (native bf16 uncertified).* The native-arm table is reported because
  the protocol's universe requires it; those numbers are uncertified and are not
  used for any conclusion.
- *Caveat 10 (multiple comparisons).* Twelve channels were declared and twelve
  tested; the three "supported" shadow results are individually above the seed
  floor, but with twelve channels and a rule that has no explicit false-positive
  control, the two `e_lin` ones should be read as what they are — a 14-19% shift
  on a quantity that behaves like an arithmetic floor.

**What could make this wrong.**

- If the d16 seed-7 run is atypical, the refutation of `e_sym_gradient` could be
  a seed accident. With n=1 at d16 this cannot be excluded; the d12 spread on
  this channel is 57% sd-relative, and the effect being tested is 9%.
- The decision rule keys on the median, which is insensitive to the tail. The
  tail is where the campaign's actual risk lives, and the tail does show a d16
  effect in the first five updates. A protocol keyed on "max over the run" or
  "fraction of checkpoints over threshold" would have returned a different
  answer for a defensible reason. I applied the rule as frozen and report the
  tail separately rather than substituting a statistic after the fact.
- Local copies lack `checkpoints/*.pt`, so the lineage-hash leg of the verifier
  cannot be re-run here (expected per DATASET.md, not corruption). I relied on
  the recorded on-pod verification. All 1290 selected rows were `is_defined`.
