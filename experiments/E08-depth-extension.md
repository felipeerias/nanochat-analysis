# E08 — extending the depth sweep to d18 and d20

Status: **draft, not frozen**. Depends on telemetry v4 for nothing, and on one
operations change (manifest fields) for its recommended arm.

## Question

Is it worth extending the d12–d16 size ray to d18 and d20, given that I0006
showed cross-depth comparison on this dataset is mostly ill-defined — and if
so, what specifically do the third and fourth points buy that d12–d16 does not?

**What answers it either way.** "Yes" requires all three: (a) the extension
carries a claim that d12–d16 cannot carry, on a family that survives I0006's
unsafe screen; (b) the extension is powered against the effect sizes that
actually exist in this system, which I0004 measured at **at most 2.26x**; (c)
the runs are physically and operationally possible at the recipe's own settings.
"No" requires only that one of those fails. Most of the evidence needed is
already in hand and is arithmetic, not experimental — §§2–5 below are computed
from the existing provenance, and they are this design's first deliverable.

## 1. The short answer

**Not as a continuation of the size ray, and not yet.** Three findings, all
derivable today:

1. **I0004's "d18 and d20 will certify" is a statement about a denominator, not
   about depth.** Pass rate = (n_ckpt − 4)/n_ckpt reproduces every published
   figure exactly. Running the depths to confirm the trend confirms arithmetic.
2. **The recipe is discontinuous between d16 and d17.** nanochat's own batch-size
   rule doubles the total batch from 2^19 to 2^20 at d17, which scales every
   learning rate by √2 and rescales weight decay. d18 and d20 are therefore
   **not further points on the d12–d16 ray**; they are the first two points of a
   second ray.
3. **Cost scales with the square of parameter count**, because the horizon is
   12:1 tokens:params and FLOPs/token is ~6·params. A d20 run is **15x a d12
   run** in compute.

There are two things worth buying, and neither is a full-length size-ray
extension. They are in §6: a ~$12 prefix probe that answers the certification
question properly and measures the discontinuity, and — if the probe passes — a
~$43 **iso-token, iso-schedule** depth contrast that dissolves I0006's structural
limits entirely for the pairs it covers, at the price of changing the estimand.

## 2. The certification result is already known, and it is arithmetic

I0004 reports gradient-direction shadow pass rates of 25–26/30 at d12, 28/32 at
d14, 29/33 at d16, and states that the non-certifying checkpoints are "the same
early ones (updates 0, 1, 2, 4) at every depth". Those two statements are the
same statement:

| depth | deep checkpoints | failing set | (n − 4)/n | I0004 reported |
|---|---:|---|---:|---:|
| d12 | 30 | {0, 1, 2, 4} | 86.7% | 83–87% (25–26/30) |
| d14 | 32 | {0, 1, 2, 4} | 87.5% | 87.5% (28/32) |
| d16 | 33 | {0, 1, 2, 4} | 87.9% | 87.9% (29/33) |
| d18 | ~34 | *predicted* {0,1,2,4} | 88.2% | — |
| d20 | ~34 | *predicted* {0,1,2,4} | 88.2% | — |

The pass rate rises with depth because the deep-checkpoint schedule adds tail
points as the run lengthens while the failure count stays fixed at four. The
rate is a denominator effect. Predicting 88% at d18 requires no GPU.

**The empirical content is elsewhere, and it is narrow**: does the failing set
stay exactly {0, 1, 2, 4}, or does a *late* checkpoint fail for the first time?
That is a real question — I0002 established that update-effectiveness records
degrade late through catastrophic cancellation, and the checkpoint-level shadow
verdicts already show two `failed` checkpoints at d14 and d16 against zero at
d12. It is also a question about **the first 64 updates**, because every
observed failure lives there. A full-length run to answer it is 3,600 wasted
steps at d18 and 4,900 at d20.

## 3. The recipe is discontinuous between d16 and d17

`scripts/base_train.py` derives the batch size from the compute-optimal horizon
following Power Lines (B ∝ D^0.383), then clamps to the nearest power of two.
Recomputing that rule at each depth, from the model's own scaling-parameter
count:

| depth | width | matmul params | horizon D | B_opt (pre-clamp) | **total batch** | LR scale | wd scaled | steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| d12 | 768 | 110.1 M | 1.321 G | 524,288 | **524,288** | 1.000 | 0.2800 | 2520 |
| d14 | 896 | 164.2 M | 1.971 G | 611,057 | **524,288** | 1.000 | 0.1877 | 3759 |
| d16 | 1024 | 234.9 M | 2.819 G | 700,750 | **524,288** | 1.000 | 0.1313 | 5376 |
| d17 | 1088 | 277.1 M | 3.326 G | 746,700 | **1,048,576** | 1.414 | 0.1573 | 3171 |
| d18 | 1152 | 324.4 M | 3.893 G | 793,097 | **1,048,576** | 1.414 | 0.1344 | 3712 |
| d20 | 1280 | 435.2 M | 5.222 G | 887,486 | **1,048,576** | 1.414 | 0.1002 | 4980 |

The d12/d14/d16 batch sizes and weight decays here reproduce the recorded
provenance exactly, so the rule is being applied correctly. The clamp boundary
sits at B_opt = 2^19.5 = 741,455, which the recipe crosses at **d17**.

Three consequences, each of which attacks something I0006 relies on:

- **Absolute-step alignment loses its data guarantee.** I0006's foundation is
  that "all seven runs share a total batch size of 524,288 and identical
  `batch/*` values at every one of the 2,520 common steps, so the token stream
  is the same run to run". At d18 a step consumes twice the tokens. d18 and d20
  would be the first runs in the dataset for which that sentence is false.
- **Absolute-step alignment loses its schedule guarantee too.** I0006 measured
  `optim/lr` and `optim/momentum` as *bit-identical* between d12 and d16 from
  step 0 to 882. At d18 every learning rate is √2 higher. The bit-identity that
  made step alignment safe below 882 simply does not exist.
- **Progress alignment weakens.** I0006's progress rule licenses "equal fraction
  of *the recipe*, explicitly not equal data". At d18 it is equal fraction of a
  *different* recipe — different batch, different LR amplitude, different weight
  decay constant. The rule was derived for a pair that shared all three.

Step count is not even monotone in depth: d16 runs 5,376 steps and d18 runs
3,712. Any figure with depth on one axis and step on the other becomes
unreadable across the boundary.

The alternative is to pin the batch size and learning rates to d12's. That is
possible (`--total-batch-size 524288`, and a `--weight-decay` argument to
restore the scaled value) but it is **not the recipe**, and DATASET.md is
explicit that an official run is exactly its resolved manifest row. The
current runner can carry these existing trainer flags, so the pinned scenario
needs a new immutable manifest and a declared estimand, not a runner-schema
change.

Both scenarios are named below and must be declared before any run:

- **Scenario A** — the recipe as it derives itself. B = 2^20, LR ×√2. Measures
  "the nanochat recipe at scale" (DATASET caveat 1), and crosses a discontinuity.
- **Scenario B** — B and LR pinned to d12's. Measures depth at fixed batch and
  fixed schedule. Comparable to the existing runs, but no longer the recipe.

## 4. What the alignment rule permits at d18 and d20

I0006's practical rule, with d12 as the shallow reference: align on **absolute
step for step ≤ 882** (d12's warmdown onset), align on **normalized progress for
progress ≥ 0.159** (= 400/2520, the Muon momentum ramp as a fraction of d12).
Both thresholds are set by the *shallowest* member of the pair, so adding depths
does not move them.

| run | steps | step 882 → progress | progress 0.159 → step | uncovered band | % of run with no d12 step counterpart |
|---|---:|---:|---:|---:|---:|
| d16 | 5376 | 0.1641 | 855 | none (overlap 855–882) | 53.1% |
| d18, Scenario A | 3712 | 0.2376 | 590 | none | 32.1% |
| d20, Scenario A | 4980 | 0.1771 | 792 | none | 49.4% |
| d18, Scenario B | 7424 | 0.1188 | 1180 | steps 883–1179 (**4.0%**) | 66.1% |
| d20, Scenario B | 9961 | 0.0885 | 1584 | steps 883–1583 (**7.0%**) | 74.7% |

Read this carefully, because the naive reading is backwards. Scenario A shows
*no* uncovered band — but only because the doubled batch shortens the run, and
the step-axis half of the coverage is invalid there for the reasons in §3.
Scenario B keeps the step axis honest and opens a genuine 4–7% band where
neither rule applies.

Two of I0006's three structural limits get monotonically worse with depth, and
neither is fixable by analysis:

- The step axis has no d12 reference beyond step 2,520. That excluded 53.1% of
  d16; it excludes **66.1% of d18 and 74.7% of d20** in Scenario B.
- The deep-checkpoint geometric prefix `{0,1,2,4,8,…}` is defined in absolute
  steps and identical at every depth, so on the progress axis no deep checkpoint
  below progress 0.05 has a cross-depth counterpart. At d18 the prefix runs to
  step 256 against a 0.05 cutoff at step 372, so the entire prefix except step 0
  is again unmatched. Recomputing `deep_step_schedule` at d18 gives ~34
  checkpoints of which ~22 have a d12 progress counterpart — the same 22-of-30
  ratio A0002 measured for d16, unimproved.

The third limit (only two d16 samples inside step 400, making 91 families
untestable in the warmup window) is unchanged in kind and worse in degree: the
periodic tier is 25 points per run in normalized progress, so at d18 in Scenario
A there are two periodic samples below step 400 and at d20 there are two.

## 5. Can five depth points carry a trend?

Only for one family, and only with seeds.

I0006's deliverable is that 160 of 248 families are unsafe for depth claims, and
that **every observable in I0001's usable-spread table is flagged except
`curvature/eta_star | shadow_fp32`**. Adding depth points does not unflag
anything: the flags are about whether a comparison is *defined*, not about
whether it is *significant*. So the trend-fitting question reduces to a single
family, evaluated at a progress anchor above 0.159.

Power for a log-linear depth slope, residual sd taken as the d12 five-seed
initialization spread for eta* (25% sd-relative, I0001; 23.9% on the certified
set, I0005), so σ_log ≈ 0.246. Detectable total change across the depth span at
α = 0.05 two-sided, 80% power:

| design | depths | seeds/depth | Σ(d−d̄)² | df | detectable change over the span |
|---|---|---:|---:|---:|---:|
| today | 12,14,16 | 1 | 8 | 1 | **133x** |
| naive extension | 12,14,16,18,20 | 1 | 40 | 3 | **3.7x** |
| powered extension | 12,14,16,18,20 | 3 | 120 | 13 | **1.72x** |
| d12 seeds only | 12 | 5 | — | — | (reference: 25% floor) |

The 133x figure is why nobody should quote the current three-point trend, and it
vindicates I0004's refusal to extrapolate its own three-point fit. The naive
extension improves that to 3.7x — but **I0004 measured the largest depth effect
anywhere in the declared universe at 2.26x**, so a one-seed five-point design is
still, by construction, unable to see the effects this system actually contains.
Three seeds per depth clears it. Anything less is a design that cannot fail
informatively.

This also settles I0004's recommendation 2 ("budget more than one seed per depth
in future sweeps"): it is not a nicety, it is the difference between a design
that can and cannot detect the known effect scale.

## 6. Design

Factors, levels, and what is held fixed.

- **Factor 1 — depth** ∈ {18, 20}, with width, head count, horizon, weight decay
  and (in Scenario A) batch size and learning rate co-varying by nanochat's own
  rules. DATASET caveat 1 applies in full: this is a size ray, never "depth
  causes X".
- **Factor 2 — scenario** ∈ {A, B} as defined in §3. Present only in Stage 0.
- **Blocking — initialization seed**, 3 levels, crossed with every depth. Per
  I0008 the seed changes initialization only; there is exactly one data ordering
  in this dataset, so under Scenario B all runs at all depths see a bitwise
  identical token stream at every shared step, and seed is a clean block.

Held fixed across everything: `head_dim = 128`, `aspect_ratio = 64`,
`window_pattern = SSSL`, `max_seq_len = 2048`, the dataset snapshot
(`dataset_files_hash 39a108f6…`), the tokenizer (`387cfc08…`), the three frozen
probes (all three probe ids are **byte-identical across all seven existing runs
and all three depths** — verified), `telemetry_periodic_points = 25`,
`deep_schedule = pythia`, `shadow = fp32`, `tolerance_version = 1`, and
`telemetry_config_hash`.

### Stage 0 — the prefix probe (the only unconditional spend)

**12 runs of 64 updates each**: 2 depths × 2 scenarios × 3 seeds. Deep
checkpoints pinned to the full-run geometric prefix `{0, 1, 2, 4, 8, 16, 32, 40,
64}` — 9 per run, 108 total.

This is not an official sweep run and must not be recorded as one. It is a probe
run, labelled as such in its own manifest (`probe-d18-d20-prefix-v1`), and its
records must not be pooled with sweep segments.

It delivers four things a full run cannot deliver more cheaply:

1. **The certification question, properly answered**, with a seed error bar at
   d18 and d20 for the first time — 3 seeds × 2 depths × 9 checkpoints.
2. **The size of the recipe discontinuity**, measured directly as the Scenario
   A-versus-B difference at matched update indices, on `loss/train_mean` (0.06%
   seed floor, the best detector in the dataset) and on
   `curvature/e_sym_gradient` (the channel I0004 declared decisive).
3. **Peak memory and step time at both depths**, converting §7's projections
   into measurements before anything expensive is committed.
4. **A calibration of the early transient** that I0004 flagged as exploratory —
   d16 reaching 1.01e-4 on `e_sym_gradient` over the first four checkpoints
   against a d12 seed median of 1.19e-5 — at two more depths, with seeds.

### Precommitted gate

Extend to full runs **only if all three hold**:

- **G1.** The non-certifying set at d18 and d20 is exactly {0, 1, 2, 4} in all
  three seeds at both depths, with no checkpoint at update index ≥ 8 failing its
  gradient-direction shadow verdict. *If G1 fails, the campaign has learned the
  thing it wanted for about $12 and stops here* — a new failure mode at depth is
  a bigger result than a trend, and it changes what the instrument must record
  before anything longer runs.
- **G2.** The Scenario A-versus-B difference on `loss/train_mean` at update 64
  is characterized and reported, whichever way it falls: above 3× the 0.06%
  initialization floor means the discontinuity is real and every cross-boundary
  figure must carry it; below means Scenario B is a defensible pin and the
  extension proceeds in Scenario B.
- **G3.** Measured peak memory at d20 leaves ≥ 8 GB of headroom at
  `device_batch_size = 32`. If it does not, `device_batch_size` must drop, and
  that is not free: the gradient-noise-scale estimator is computed on a device
  batch (DATASET caveat 9), so all `noise/*` families would stop being
  comparable to the existing seven runs and must be declared incomparable rather
  than quietly pooled.

### Stage 1-A — iso-token, iso-schedule depth contrast (recommended)

If the gate passes, **this is the arm worth buying, and it is not a size-ray
extension.** Run d18 and d20 for exactly d12's token budget on exactly d12's
schedule: `--total-batch-size 524288 --num-iterations 2520`, plus
`--weight-decay 0.8250` (d18) / `1.1067` (d20) so that `weight_decay_scaled`
lands on d12's 0.28.

Why it is worth buying: the *entire* recipe becomes bit-identical to d12's. The
40-step LR warmup, the 400-step Muon momentum ramp, the warmdown onset at step
882 (progress 0.350), the final LR fraction, the weight-decay amplitude and
cosine, and — because the loader has no RNG — the token stream itself. Every one
of d12's 30 deep checkpoints has a counterpart at the same step *and* the same
progress *and* the same tokens seen. **I0006's three structural limits do not
apply to this pair at all.** The 160-family unsafe list was derived for runs
whose schedules and token budgets differ; for an iso-schedule, iso-token pair
the alignment question does not arise, and the families become testable against
the I0001 seed floor directly.

What it costs in meaning: the models are trained at a 4.07:1 tokens:params ratio
instead of 12:1, so they are deliberately undertrained and this is **not** a
statement about the nanochat recipe at scale. It is a statement about depth at
fixed data and fixed schedule. Those are different questions, and this one is
the one the existing dataset cannot answer at any price.

**10 new runs.** The core arm is 2 depths × 3 seeds — 6 runs, ~13.2 GPU-h,
**about $43**. The optional back-fill is 2 iso-token seeds each at d14 and d16
(4 runs, ~4.9 GPU-h, ~$16), which extends the fully-aligned pair set to four
depths. The d12 reference needs no new runs at all: the five existing d12 runs
*are* the iso-token point. Total with back-fill: **~18 GPU-hours, ~$59.**

### Stage 1-B — powered size-ray extension (the expensive alternative)

If the campaign specifically wants "the recipe at scale" rather than depth at
fixed data: 3 seeds each at d18 and d20 in Scenario A (42.4 GPU-h, ~$140), plus
2 back-fill seeds each at d14 and d16 (8.5 GPU-h, ~$28). **10 new runs, about
51 GPU-hours and $168.** Its trend is powered to 1.72x per §5, but it spans the
d17 discontinuity and its every figure must say so.

For scale: $168 is three times E01's entire 24-run factorial, and buys a powered
trend on exactly one metric family. Stage 1-A costs a third of that and buys a
comparison with no alignment caveat at all — on a different estimand.

## 7. Costs, from measurement not assumption

Wall-clock is projected from the three measured runs at 4.9e14 achieved FLOP/s
(measured: 498, 473, 507 TFLOP/s at d12/d14/d16, from logged training time minus
logged telemetry overhead). Telemetry overhead is extrapolated from the recorded
per-tier totals (575.3 / 786.3 / 1039.6 s).

| run | steps | tokens | peak memory | compute | telemetry | wall | $ @3.29/h | × a d12 run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| d12 | 2520 | 1.321 G | 39.5 GB (measured) | 0.56 h | 0.16 h | **0.72 h** | $2.37 | 1.0 |
| d14 | 3759 | 1.971 G | 42.4 GB (measured) | 1.31 h | 0.22 h | **1.52 h** | $5.02 | 2.1 |
| d16 | 5376 | 2.819 G | 55.9 GB (measured) | 2.45 h | 0.29 h | **2.74 h** | $9.01 | 3.8 |
| d18 (A) | 3712 | 3.892 G | ~72 GB (projected) | 4.81 h | ~0.35 h | **~5.2 h** | ~$17.0 | 7.2 |
| d20 (A) | 4980 | 5.222 G | **~91 GB (projected)** | 8.54 h | ~0.43 h | **~9.0 h** | ~$29.5 | 12.5 |
| d18 iso-token | 2520 | 1.321 G | ~72 GB | 1.63 h | ~0.25 h | **~1.9 h** | ~$6.2 | 2.6 |
| d20 iso-token | 2520 | 1.321 G | ~91 GB | 2.16 h | ~0.30 h | **~2.5 h** | ~$8.1 | 3.4 |
| Stage 0 probe | 64 | 0.067 G | — | ~0.1 h | ~0.09 h | **~0.25 h** | ~$0.80 | 0.35 |

**Cost scales with the square of parameter count.** The horizon is 12·P tokens
and FLOPs/token is ≈ 6·P, so total FLOPs ≈ 72·P². The measured d12→d16 compute
ratio is 4.45 against a P² prediction of 4.55; the d20 projection is 15.0x a d12
run against a P² prediction of 15.6x.

**The d20 memory projection is the largest single feasibility risk.** Peak memory
across the three measured runs is 39.5, 42.4, 55.9 GB, and the d14→d16 jump
(+13.6 GB) is far larger than d12→d14 (+2.9 GB), so a linear-in-parameters
extrapolation (~98 bytes/param on the d14–d16 slope) is not trustworthy — it
gives ~72 GB at d18 and **~91 GB at d20, which does not fit an 80 GB H100**.
G3 exists because this number must be measured, not modelled.

## 8. Primary outcome, precommitted

**Stage 0**: the *identity* of the non-certifying checkpoint set, per run —
categorical, not a rate. Measured by `curvature/verdict_code_gradient` on
`acceptance_arm = "shadow_fp32"`, sparse tier, on the frozen `short` probe
(probe id `2d9baec3…`, byte-identical across every run in the dataset).
Reported separately for updates 0–4 and for the rest, per I0004 recommendation 1.
Secondary: `loss/train_mean` at update 64 for the Scenario A-versus-B contrast.

**Stage 1** (either arm): the depth slope of `log(curvature/eta_star)`,
`acceptance_arm = "shadow_fp32"`, gradient-direction-certified, at the progress
≈ 0.75 deep checkpoint — which exists at every depth by construction, since the
schedule's uniform tail is uniform in progress (d12 step 1889, d16 step 4031,
d18 step 5567 in Scenario B). eta* is chosen because it is the **only** entry in
I0001's usable-spread table that I0006 did not flag as unsafe for depth claims,
and progress 0.75 because it is above the 0.159 threshold and inside the
post-warmdown plateau I0005 established.

Note for the record that eta* and `vhv_gradient` are the same channel to 1.2e-15
(I0005), so they must not be reported as corroborating each other.

Probe partition: none of the v4 partitions exist yet, and the primary outcome
does not need them — it is measured on the frozen `short` probe, which no
training process consumes. If a validation-loss outcome is added later it must
use the **sealed** partition (v4 item 4) and not the existing val probe.

## 9. Power, honestly

- **Certification (Stage 0, categorical).** 3 seeds × 2 depths × 5 late
  checkpoints in the 64-step window, plus the 29-ish late checkpoints per run if
  Stage 1 proceeds. On Stage 1-A alone (6 runs × ~26 late checkpoints = 156),
  the rule of three gives: observing zero late failures bounds the late-failure
  probability below **1.9%** at 95% confidence. That is the honest form of the
  claim "certification extends to d18 and d20".
- **eta\* depth slope.** Per §5: 1.72x detectable over the depth span with 3
  seeds per depth against I0001's 25% initialization floor for eta*; 3.7x with
  one seed; 133x with the current three depths. Against I0004's measured maximum
  depth effect of 2.26x, only the three-seed design is adequate.
- **Loss.** `loss/train_mean` has a 0.06% initialization floor and would detect
  essentially any depth difference — but I0006 flags it 1 of 1 unsafe for depth
  claims, so it is usable here **only** in the iso-schedule iso-token arm, where
  the alignment question does not arise. That is a concrete example of what
  Stage 1-A buys.
- **Curvature generally.** I0001's practical rule stands: curvature effects need
  roughly 50–75% to clear five-run seed noise, and detecting 20% needs about 29
  seeds per arm. No affordable depth design changes that.
- **Not covered by any floor here.** Both scenarios hold data order fixed, so
  I0001's initialization floor is the right floor. Any future design that varies
  order or batching must establish its own, at roughly 10x (I0008).

## 10. What this design does not answer

- **Whether depth causes anything.** Depth co-varies with width, heads, batch,
  LR, weight decay and horizon by construction (DATASET caveat 1). E03 and E04
  are where fixed-depth geometry questions belong.
- **Anything about the 160 flagged families.** More depth points and more seeds
  do not make an ill-defined comparison well-defined. Only the iso-schedule
  iso-token arm changes that, and only for the pairs it covers.
- **Where the d17 discontinuity's effects come from.** Batch size, LR scale and
  weight decay all move together at the clamp boundary; Stage 0 measures the
  bundle, not its parts. Decomposing it is a fixed-depth batch-size experiment
  that does not exist yet.
- **Whether the compute-optimal horizon is right.** The 12:1 ratio is taken as
  given; nothing here tests it.
- **Anything about λ_max or spectra.** E06's memory contract already fails at
  d14 and d16 (80 GB and 107 GB of Lanczos basis); at d18 and d20 it is 140 GB
  and 179 GB. Spectral quantities are not available at these depths on this
  hardware, and no design here should promise them.
- **Whether curvature at these depths means what it means at d12.** All of it is
  measured on one 256-token sequence (DATASET caveat 4). E09 is the design that
  asks whether that matters, and it should land before Stage 1 either way.
- **The early transient.** Stage 0 characterizes it with seeds but stays inside
  the absolute warmup window, where I0006 says cross-depth comparison is least
  defined. It is a within-depth description, not a cross-depth finding.

## 11. Instrument dependencies

**Telemetry v4: none required.** The instrument as frozen at schema v3 records
everything Stage 0 and Stage 1 need.

Two **operations** dependencies, both outside the instrument:

1. **A deep-step pin.** `add_telemetry_cli_args` offers `--telemetry-deep-every`
   and `--telemetry-deep-schedule {pythia, every}`, and `deep_step_schedule`
   computes its geometric prefix from `ceil(0.05·N)` — so a 64-step run gets a
   64-step-shaped schedule, not the full run's prefix, and misses updates 1 and
   2, which are two of the four that matter. Stage 0 needs either an explicit
   `--telemetry-deep-steps` list (a one-line CLI affordance, and the honest ask)
   or `--telemetry-deep-every 1`, which costs 65 deep checkpoints per run and
   raises Stage 0 from ~$12 to ~$30. Prefer the flag.
2. **New manifests for Scenario B and for the iso-token arm.** The current
   runner already accepts parser-declared `total_batch_size`, `num_iterations`,
   and `weight_decay` fields. Put them in new manifests
   (`probe-d18-d20-prefix-v1`, then `sweep-d18-d20-v1` or
   `isotoken-d18-d20-v1`), since manifests are immutable once used.

Two v4 items would materially improve any Stage 1: item 1 (separated
`init_seed`/`data_seed`, so the "differs only in X" claim is checkable rather
than assumed) and item 5 (the loader's direction-certification fix, completed
in analysis commit `8950b04`, since every outcome here is
gradient-direction-certified).

## Open questions before freezing

1. **Is Scenario A even the right object?** If the answer to "what does the
   recipe do at scale" is "it discretely changes recipe at d17", then the
   interesting experiment might be the clamp boundary itself — d16 versus d17 at
   fixed depth-adjacent settings — rather than d18 and d20. That is a cheaper
   and better-posed question, and this design does not currently contain it.
2. **Does the iso-token arm's undertraining invalidate its curvature outcomes?**
   I0005 established that sharpening is locked to the warmdown, and the
   iso-token arm reproduces d12's warmdown exactly — but at 4.07:1 the model is
   in a different part of the loss landscape at every progress point. Whether
   "same schedule, less data" preserves the phenomenon is itself untested, and
   if it does not, Stage 1-A's primary outcome may be measuring a different
   regime rather than a deeper model.
3. **Should Stage 0 run at d17 instead of d18?** d17 is the first depth past the
   clamp and is 15% cheaper than d18. If the goal is to characterize the
   discontinuity, d17 is the better probe; if the goal is a trend, d18 and d20
   are the better spacing. The two goals point at different runs.
4. **What is the actual d20 memory ceiling?** The projection (~91 GB) sits above
   an H100's 80 GB but below an H200's 141 GB. If d20 needs different hardware,
   every timing and cost figure in §7 changes, and `noise/*` comparability
   changes with `device_batch_size`. Should the design simply exclude d20 and
   propose d17/d18 instead?
5. **Three seeds or five?** §5 sizes the trend at three. I0001's reference uses
   five, and matching it would make the new depths' floors directly comparable
   to d12's rather than interpolated. Five seeds at d18 and d20 in Scenario A is
   ~$234; in the iso-token arm it is ~$72. The iso-token arm can afford five and
   the size-ray arm cannot, which is another argument for the former.
6. **Does the deep-checkpoint schedule need re-cutting at these depths?** The
   geometric prefix is absolute and the tail is proportional, which is exactly
   the hybrid that I0006 identified as the root of the alignment problem. A
   schedule with a *proportional* prefix would give cross-depth counterparts
   below progress 0.05 for the first time — but it would break comparability
   with all seven existing runs. This is an instrument decision that should be
   settled before, not during, an extension.
7. **What happens to `parameter_schema_hash`?** Sketches are comparable only
   within a matching schema hash, and every new depth has a new one. No sketch
   family can participate in a five-depth trend. Is that already implied by
   I0006's `probe/` 38-of-38 flag, or is it an additional exclusion that the
   family accounting has not recorded?
