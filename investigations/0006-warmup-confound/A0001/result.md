---
investigation: I0006
analyst: A0001
design: confirmatory
outcome: supported
saw: read with the analysis repo at HEAD 75801be. Protocol
  `investigations/0006-warmup-confound/README.md`@e76859c;
  `analysis/README.md`@cc5ecea; `investigations/TEMPLATE-result.md`@2a460b5;
  `investigations/0001-seed-variation/conclusion.md`@4ac11f3;
  `analysis/loader/telemetry_load.py`@c0419ef;
  `telemetry-data/sweep/DATASET.md` (working tree); the seven schema-v3 segments
  (provenance.json + continuous/periodic/sparse/offline parquet).
  NOT read: the sibling `A0002/` folder, any `conclusion.md` for I0006, any
  other investigation's results, `analysis/profiles/`.
data: sweep collection, seven schema-v3 segments (`d12-s7..s11`, `d14-s7`,
  `d16-s7`). The legacy v1 segment `d12-iter` is excluded per protocol.
selection: `is_defined == True` (8,742 of 2,405,132 rows dropped explicitly);
  `value_scalar` not null (141,209 vector-payload rows excluded, 69 vector
  families); tiers continuous+periodic+sparse (offline excluded - 15 metrics,
  one row per run, no step axis); rows sharing a (run, metric,
  acceptance_arm, phase, step) key collapsed with the MEDIAN. Comparison points
  are d16's own measurements; only the d12 reference is interpolated, and never
  extrapolated.
universe: 250 scalar families keyed (metric, acceptance_arm) exist in the v3
  data; 248 are present at all three depths in all seven runs and all 248 were
  tested and are reported (`families.csv`). Two were dropped for having defined
  rows at d12 only: `curvature/e_fd_update|shadow_fp32`,
  `curvature/fd_cos_update|shadow_fp32`. Each family was tested in two regions
  x two alignments: up to 992 family-region-alignment tests (3 families have
  only one populated region). No channel was selected after looking.
code: `investigations/0006-warmup-confound/A0001/` -
  `run.sh` drives `{extract,structure,analyze,classify,headline,robustness,plots}.py`
  end to end. Commit pending: the coordinator commits.
seed_reference: investigations/0001-seed-variation/conclusion.md@4ac11f3
  (standard-deviation column; effects are quoted in units of the local
  five-seed sd, and the 3-sigma decision threshold is I0001's "two to three
  times the sd-relative spread" rule)
supersedes: none
---

## Result

**The warmup windows do distort cross-depth comparison, but the dominant
distortion is not the one the protocol anticipated.** Almost no family has a
genuine depth difference that lives inside step <= 400. What the data shows
instead is that *the choice of x-axis* changes the cross-depth answer for a
third of the testable universe, and that `normalized_progress` -- the axis
`DATASET.md` names as "the cross-depth x-axis" -- is the wrong axis inside the
window and manufactures apparent warmup effects there.

### 1. The recipe is a hybrid schedule, so no single axis is correct everywhere

Measured from `optim/lr` and `optim/momentum` (`headline.txt` section 1;
`figures/fig1-schedule.png`). Values are the maximum |d16 - d12| in each phase:

| phase | on the ABSOLUTE-STEP axis | on the NORMALIZED-PROGRESS axis |
|---|---:|---:|
| LR warmup (step 0-40) | **0** (bit-identical) | 0.0104 LR / 0.0135 mom |
| Muon ramp (step 41-400) | **0** (bit-identical) | 0 LR / **0.0638 mom** |
| plateau (step 401-882) | **0** (bit-identical) | 0 / 0 |
| d12 warmdown (step 883+) | 0.0155 LR / 0.0572 mom | ~2e-6 (discretisation) |

The progress-axis column is evaluated over the progress interval that phase
occupies at d12 (p < 0.0159, [0.0159, 0.1587), [0.1587, 0.35), p >= 0.35).

Peak LR is 0.02 and peak momentum 0.97 at every depth, so those step-axis
disagreements are 78% of peak LR and a momentum gap of 0.907 vs 0.970.

The warmups are absolute; the **warmdown onset is proportional** -- landmark 3
is step 882/1316/1882 at d12/d14/d16, i.e. progress 0.3500/0.3501/0.3501.
Therefore:

- **absolute step is the schedule-correct axis on steps 0-882** (35% of d12,
  16% of d16);
- **normalized progress is the schedule-correct axis for p >= 0.159**;
- they overlap only in p in [0.159, 0.35], where both are in the flat plateau.

`optim/weight_decay` is a third case, aligned on neither axis: its decay begins
at progress 0.723 (d12), 0.655 (d14), 0.577 (d16).

### 2. Absolute-step alignment also holds the data fixed; progress alignment does not

All seven runs use `total_batch_size = 524288`, and `batch/bos_count`,
`batch/valid_targets`, `batch/mean_segment_length`, `batch/segments_per_row_mean`
and `batch/segments_per_row_max` are **identical at every one of the 2,520
common steps across all seven runs**. `tokens_seen == step * 524288` everywhere.
So the two alignments differ in two ways at once: schedule phase, and token
budget -- at equal progress d16 has seen **2.133x** the tokens d12 has.

### 3. What the alignment choice costs, per family

Because both alignments compare the *same* d16 sample, the disagreement between
them is exactly `d12(step s) - d12(progress s/5376)`: a property of the d12
reference alone. In units of the pooled five-seed sd:

| group | n | inside window: median / p90 | after window: median / p90 |
|---|---:|---:|---:|
| dynamics | 192 | 1.39 / 6.07 sigma | 1.58 / 11.10 sigma |
| resource | 31 | 0.02 / 0.47 | 0.05 / 0.54 |
| config | 25 | 0.00 / 4.26 | 0.00 / 4.44 |

54 families have an in-window alignment gap >= 3 sigma. The largest:
`update/loss_before` (36.0 sigma, 21.8% of level), `curvature/curv_floor_gradient|native`
(33.2), `probe/loss` (19.5, 24.1%), `update/loss_after` (18.8-19.2, 25.6%),
`update/direction_norm` (17.3, 41.7%), `loss/train_mean` (16.9, 19.2%).
Full column in `families.csv` (`w_align_dz`, `w_align_drel`).

**`loss/train_mean`, I0001's best detector (0.06% sd-relative whole-run).**
Same d16 run, two references (`figures/fig2-loss.png`):

| region | aligned on step | aligned on progress |
|---|---:|---:|
| LR warmup, step 0-40 | -0.67% (2.6 sigma) | -14.67% (25.4 sigma) |
| Muon ramp, step 41-400 | -2.82% (4.1 sigma) | -18.57% (16.1 sigma) |
| step 401-882 | -3.34% (28.2 sigma) | -10.91% (30.4 sigma) |
| step 883-2520 | +0.05% (28.7 sigma) | -8.25% (87.3 sigma) |
| step 2521-5375 | *no d12 exists* | -7.61% (170.8 sigma) |

The **sign** of the d12->d16 loss difference disagrees between the two
alignments at 32.6% of the 2,520 shared points -- every one of them at step >
1667, where d12 is deep in warmdown and d16 is not. The local d12 seed sd is
0.506% inside the window and 0.056% after it: the window is ~9x noisier per
seed than the rest of training, which further weakens in-window detection.

### 4. Classification (the protocol's test)

Verdicts use the median |z| over the region's comparison points, z = (d16 -
median of 5 d12 seeds) / (sd of the 5 d12 seeds) at the same x; threshold 3
sigma per I0001's rule; "dominated" means >= 3x larger in one region than the
other. Deterministic channels (zero seed spread) are judged by exact equality.

| verdict | ABSOLUTE-STEP alignment | NORMALIZED-PROGRESS alignment |
|---|---:|---:|
| warmup-dominated | **2** | **20** |
| post-dominated | 10 (+4 det.) | 3 (+1 det.) |
| uniformly different | 29 (+8 det.) | 44 (+15 det.) |
| not different | 75 | 49 |
| identical (deterministic) | 29 | 25 |
| underpowered | 91 | 91 |

*(dynamics + config + resource pooled; per-group breakdown in `classify.txt`
and `families.csv`.)*

**51 of the 157 families testable under both alignments (32%) change verdict.**
The flips are one-directional: 14 families that are *not different* on the step
axis become *warmup-dominated* on the progress axis, and 13 more become
*uniformly different*; the reverse flip (warmup-dominated on step -> not
different on progress) happens twice. `figures/fig3-alignment.png`: inside the
window, **70 of 94** dynamics families sit above the diagonal, i.e. progress
alignment reports a larger effect. After the window it is 47 of 97 -- no bias.

The **only two families that are genuinely warmup-dominated when the schedule
and the data are held fixed** are `curvature/e_sym_gradient|native` (|z| 3.49
in-window vs 0.74 after) and `curvature/fd_floor_random|native` (4.34 vs 0.35).
Both are native-arm *acceptance-test internals* -- an HVP symmetry error and a
finite-difference noise floor -- not observables; `DATASET.md` caveat 6 says the
native arm is uncertified everywhere, and I0001 puts the neighbouring
`curvature/c_fd_*` / `curv_floor_*` channels in the "instrument noise
diagnostics, not observables" row. **No observable is warmup-dominated in the
protocol's sense.** Across 248 families, two crossings of a 3-sigma line is the
same order as chance alone would produce (caveat 10), so this pair should not be
read as a finding.

### 5. The measurement cadence blocks the test for a third of the universe

- **continuous** (every step): 401 d16 samples inside the window. Full power.
- **sparse** (deep checkpoints): 23 d16 sample rows inside the window, of which
  17 step labels are shared with d12 -- the geometric prefix {0,1,2,4,8,16,32,
  40,64} plus the step-400 landmark. Adequate.
- **periodic** (`ceil(N/25)`, progress-aligned by construction): **2** d16
  samples inside the window (steps 0 and 216, or 1 and 217 for post-update
  families), against 4 at d12. This is below
  any usable threshold, and it makes the entire periodic tier untestable in the
  window: **91 families**, including all 38 `probe/*`, both `param/*`, all 5
  `noise/*`, all 5 `attn/*`, all 5 periodic `grad/*`, 14 of 15 `muon/*` and 4
  `optim/adamw_*`.

The protocol's hint is confirmed and is stronger than stated
(`figures/fig4-grid.png`): the deep-checkpoint geometric prefix is defined in
**absolute steps** and is bit-identical across depths, so on the progress axis
**no deep checkpoint below p = 0.05 has a cross-depth counterpart** (d12's
p=0.05 is step 126, d16's is step 269 -- these do match; everything earlier does
not). Any progress-aligned sparse-tier analysis of the first 5% of training is
interpolating across the geometric gaps, not comparing measurements.

Symmetrically, the absolute-step axis has no d12 reference beyond step 2520:
**53.1% of d16's training cannot be compared on that axis at all.**

### 6. Deliverable: families unsafe for depth claims

**160 of 248 families (135 of 192 dynamics families) are flagged.** Full list
with per-family numbers in `families.csv` (`unsafe`, `why_unsafe` columns).
Reasons, in descending count:

| reason | n | meaning |
|---|---:|---|
| window untestable (< 5 comparison points) | 91 | cadence cannot rule a warmup artifact in or out |
| alignment-unstable after the window (>= 3 sigma) | 85 | the post-window answer depends on the axis |
| alignment-unstable inside the window (>= 3 sigma) | 54 | the in-window answer depends on the axis |
| verdict flips with alignment | 51 | the three-way label itself depends on the axis |
| warmup-dominated on the schedule-correct axis | 2 | a real in-window-only difference |

By metric prefix (unsafe / total): `probe/` 38/38, `attn/` 5/5, `noise/` 5/5,
`grad/`(periodic) 5/5, `muon/` 15/15, `param/` 2/2, `calib/` 2/2, `sketch/` 4/4,
`scalars/` 2/2, `update/` 14/18, `curvature/` 42/95, `optim/` 12/17,
`overhead/` 9/24, `batch/` 4/6, `loss/train_mean` 1/1.

**Every observable named in I0001's usable-spread table is flagged except one.**
`curvature/eta_star|shadow_fp32` is the sole survivor, and only because it is
below the seed band under both alignments. (I0001's last row, `curvature/c_fd_*`
and `curv_floor_*`, is labelled there as instrument-noise diagnostics rather
than observables; several of those are indeed safe, which is not useful.)

The 57 dynamics families that are **safe** are almost entirely sparse-tier
curvature/update diagnostics that show no cross-depth difference on either axis
(e.g. `curvature/e_sym_*`, `curvature/curv_eps_*`, `curvature/c_fd_random|*`,
`update/normalized_residual|*`, `update/residual_p2|*`) plus 15 deterministic
constants. Listed in full at the end of `classify.txt`.

### 7. Practical rule this supports

For a cross-depth claim in this dataset:

1. inside step <= 882, align on **absolute step** -- schedule and token stream
   are then both exactly matched, and the claim is "at equal data and equal
   optimizer state";
2. for progress >= 0.159, align on **normalized progress** -- the claim is then
   "at equal fraction of the recipe", explicitly *not* at equal data;
3. never quote a warmup-window cross-depth number for a `periodic`-tier family;
4. state which axis was used, because for a third of families it changes the
   answer, and for `loss/train_mean` after step 1667 it changes the sign.

## Limitations

**Deviations from the protocol.**

- The protocol says "compute the d12-to-d16 difference twice" without naming a
  statistic. I chose z = (d16 - median of the 5 d12 seeds) / (sd of the 5 d12
  seeds), evaluated pointwise and summarised with the median of |z| over the
  region, plus the relative difference. This is a disclosed choice.
- The protocol's three labels do not cover two cases the data forces:
  *post-dominated* (difference only outside the window) and *underpowered*. I
  added both rather than forcing families into an ill-fitting label.
- I added unsafety criteria beyond "warmup-dominated" -- alignment instability,
  verdict flipping, and window-untestability. These were introduced **after**
  seeing that only two families are warmup-dominated on the schedule-correct
  axis, so by `analysis/README.md`'s definition they are **exploratory**, and the
  three counts they contribute to the deliverable (54 / 51 / 91) carry that
  weaker evidence level. Without them the deliverable would be a list of two
  native-arm acceptance-test internals, which would badly understate the problem
  the protocol was written to find. The declared test itself -- the three-way
  classification computed under both alignments over the declared universe -- is
  confirmatory and was run exactly as written.
- I did not restrict curvature to passing per-direction verdicts. Per I0001's
  finding, only the shadow *gradient* direction ever passes; restricting would
  delete most curvature channels from the universe, and the protocol asked for
  "every scalar family present in all three depths". Curvature numbers here are
  therefore **uncertified** (caveat 6) and describe channel behaviour, not
  certified curvature. This mirrors the A0001/A0002 split in I0001.
- `ANSWER_SHEET.md`, the self-test contract `DATASET.md` points at, is not
  present in this analysis repo, so the pipeline could not be validated against
  it. Instead `robustness.py` recomputes the `loss/train_mean` headline directly
  from the parquet, bypassing every helper in this folder, and reproduces
  -2.772% / -18.371% (window) and -1.077% (post, step-aligned) exactly.

**Choices that could make this wrong.**

- **Aggregation.** Rows sharing a step were collapsed with the median. 55 of the
  77 sparse families carry more than one row per step (up to 113); for almost
  all of them median and mean agree to better than 1e-4 relative (p90 = 0.0000),
  but `sketch/probe_grad_sq_norm` (median |mean-med|/|med| = 6.3) and
  `muon/replay_update_relerr` (0.17) are sensitive. More seriously, for
  per-parameter families the aggregated *populations differ across depths* --
  d16 has 16 blocks, d12 has 12 -- so those families compare unlike sets of
  tensors regardless of alignment. That is an additional confound this
  investigation did not control.
- **Interpolation.** The d12 reference is linearly interpolated onto the
  requested x, never extrapolated. Median bracket widths in the window are 8
  steps (step axis) and 0.0048 progress = 12 d12-steps (progress axis) for the
  sparse tier, so neither axis is advantaged. For the continuous tier the step
  axis needs no interpolation at all.
- **Threshold.** At z >= 2 / 3 / 4, the count of dynamics families clearing the
  band inside the window is 35 / 18 / 11 on the step axis and 66 / 49 / 42 on
  the progress axis. The direction of the finding is threshold-independent; the
  counts are not.
- **Pointwise sd from five seeds** is itself noisy; the median over many
  comparison points is the mitigation, but a family with few points (the sparse
  tier's 12-13 in the window) has a correspondingly noisy band.
- **The seed band is d12-only.** I0001 says so explicitly, and this
  investigation is one of the reasons it may not transfer: the d12 in-window
  seed sd on training loss is 9x the post-window value, and there is no
  equivalent measurement at d16 (n=1 seed).

**`DATASET.md` caveats that apply.** 1 (size ray, not a depth sweep -- every
"depth" statement here is about the nanochat recipe at scale); 2 (n=3 depths,
n=1 seed at d14/d16, so d16-vs-d12 rests on a single d16 trajectory);
3 (the confound under study); 4 (all curvature is local to one short-probe
sequence); 5 (Muon replay error is a reference-frame quantity); 6 (native
curvature uncertified -- both warmup-dominated families are native-arm); 7
(all runs use the same probes, so `probe/*` is sample-local but its seed band
has no probe-selection variance -- and `probe/*` is the largest untestable block); 9
(`noise/*` descriptive only); 10 (multiple comparisons: 992 family-region-
alignment tests were run, so the two 3-sigma warmup-dominated crossings should
be read as consistent with chance).

**What would settle it.** The alignment finding is structural and would hold on
any rerun of this recipe. The "no observable is warmup-dominated" null is weak:
it is limited by the periodic tier's 2-sample window and by n=1 at d16. A run
with a proportional warmup (40 * N/2520 steps) at each depth, or simply a denser
periodic cadence inside the first 400 steps, would convert most of the 91
untestable families into testable ones.
