# E02 — warmdown onset versus certified sharpening

Status: **draft, not frozen**. Runs on schema v3 as it stands, plus a manifest
and runner extension for two schedule fields (see instrument dependencies).

## Question

Does certified curvature sharpen **because the learning rate starts to decay**,
or because training has reached a certain point in its progress?

[I0005](../investigations/0005-certified-curvature-trajectory/conclusion.md)
found, in two independent blind analyses of the same five d12 runs, that
curvature along the gradient direction is flat while the learning rate is flat,
rises steeply once warmdown begins, then plateaus. The onset is proportional:
normalized progress **0.350** at d12, d14 and d16 alike, which is exactly where
nanochat's default `--warmdown-ratio 0.65` puts it. Before the onset the five
seeds show no agreed direction (Spearman −0.26 to +0.28, three of five
positive); after it, all five rise (+0.62 to +0.95).

That co-timing is currently a **coincidence of schedule**. Nothing in the
dataset varies the schedule, so I0005 explicitly refuses to call it a cause,
and the finding is a same-data reproduction rather than a confirmation.

This design moves the onset and asks whether the sharpening moves with it. If
it does, sharpening is a response to the schedule. If it stays at 0.350
regardless, sharpening is a property of training progress and the co-timing was
an accident of the default recipe.

## Why this is cheap, and why the outcome is a time and not a level

Curvature is the weakest usable channel in the dataset:
[I0001](../investigations/0001-seed-variation/conclusion.md) puts its
initialization noise floor at **25% sd-relative for `eta_star`/`vhv_gradient`
and 29% for `gHg`**, and the README's standing constraint follows from that —
detecting a 20% effect on a curvature *level* needs about 29 seeds per arm.
A design that compared curvature levels between arms would be unaffordable.

The design therefore does not compare levels. It compares **the shape of the
trajectory inside each run**. A per-run curvature offset is a constant in
log space, so it is absorbed exactly by the intercept of a within-run fit and
cannot move a within-run change point or a within-run ratio. I0005's own
numbers demonstrate this: the plateau *levels* differ across seeds by 15–19%,
which the 25–29% reference cannot resolve, while the within-run *ratio* across
warmdown is 15.6x for `gHg` with only a **23% across-seed spread on the ratio**
and the same sign in five of five runs.

So the primary outcome is a **within-run change point** — a location in
normalized progress — and the whole design fits in fifteen 40-minute runs.

## Design

Five arms at d12, one factor: the learning-rate schedule.

| arm | `--warmdown-ratio` | `--final-lr-frac` | LR decay onset | LR at end | Muon momentum decay onset |
|---|---:|---:|---:|---:|---:|
| A reference | 0.65 (default) | 0.05 (default) | 0.350 (step 882) | 0.05 | 0.350 |
| B early | 0.75 | 0.05 | 0.250 (step 630) | 0.05 | 0.250 |
| C late | 0.35 | 0.05 | 0.650 (step 1638) | 0.05 | 0.650 |
| D constant LR | 0.65 | 1.00 | none | 1.00 | 0.350 |
| E shallow decay | 0.65 | 0.30 | 0.350 (step 882) | 0.30 | 0.350 |

A, B and C move the onset and are the primary contrast; their onsets span 0.400
of normalized progress, eight times the deep-checkpoint spacing. D and E answer
the two follow-on questions the finding raises.

**D is the constant-learning-rate arm and it is informative.** With
`--final-lr-frac 1.0` the multiplier in `get_lr_multiplier` reduces to exactly
1.0 for every step after warmup, so the LR is flat for the whole run while
*everything else, including the Muon momentum warmdown at 0.350, is unchanged*.
Since `--warmdown-ratio` moves the LR decay and the momentum decay together in
A, B and C, D is the only available way to separate them: it holds the LR flat
and lets the momentum schedule run. Its predictions are sharp and opposite —
no change point means LR decay is necessary; a change point at 0.350 means the
momentum schedule is doing the work and B and C were tracking momentum, not LR.

**E asks whether the plateau belongs to the schedule or to the model.** A, B
and C all end at the same final LR, so if curvature tracks the LR level they
are all forced to the same plateau and cannot discriminate. E holds the onset
fixed and makes the decay shallower: it ends at 0.30 of peak LR instead of
0.05. If the plateau is where the LR stops falling, E must plateau lower; if it
is a ceiling the model reaches, E must plateau at A's level.

Blocking: three previously unused initialization seeds, provisionally 201–203,
crossed with all five arms. Run order randomized within seed blocks. **15 runs,
about $35 and 10 H100-hours; budget $45**, since official runs do not resume
and a failure restarts from zero at $2.30.

Held fixed: depth 12, width 768, head dimension 128, sequence length 2048,
window pattern, vocabulary, 2,520 updates, 524,288-token logical batch, device
batch 32, matrix/embedding/unembedding/scalar LRs, 40-step LR warmup, the
hard-coded 400-step Muon momentum ramp and its 0.85/0.97/0.90 endpoints, Muon
weight decay 0.28 and its cosine schedule, dataset snapshot, tokenizer, data
order, frozen probes, telemetry cadence (25 periodic points, `pythia` deep
schedule, `fp32` shadow, 3 lineage checkpoints), hardware class and software
revision.

The data control here is unusually strong. `nanochat/dataloader.py` contains no
RNG and the seed changes initialization only (I0008, and DATASET.md), so **every
arm sees the same batch at every one of the 2,520 updates**. An arm contrast at
a given step is a contrast at matched data, not merely at matched step count.

The deep-checkpoint grid is also nearly common. `telemetry.py` derives the
landmark `warmdown_start` from `--warmdown-ratio`, so each arm's own onset is
always observed; and because 630 and 882 are already uniform-tail points, arms
A, B, D and E share **exactly the same 30 deep steps**, while C has 31 (its
landmark 1638 lands one update after the tail point 1637 — an accidental but
welcome one-update repeatability check on the curvature channel). Certified
points per run in the existing d12 data are 25–26 of 30; with the arms' onsets
that leaves roughly 10/16 (B), 12/14 (A, D, E) and 19/8 (C) certified points
before and after the onset.

## Primary outcome

Precommitted before the first run: the **change-point location τ̂**, in
normalized progress, of `curvature/vhv_gradient` on the `shadow_fp32` arm,
restricted to rows with `is_defined == True` and
`curvature/verdict_code_gradient == 0` (passed).

Channel choice, stated because it differs from I0005's headline. `gHg` rose
15.6x but `gHg = vhv_gradient * gg`, and `gg` — the squared probe-gradient norm
— itself rose 2.73x; A0001 attributed 61% of the log rise to curvature per unit
direction and 37% to gradient growth. `vhv_gradient` is the Rayleigh quotient
along the *unit* gradient and is the landscape quantity; it rose 4.76x.
`eta_star` is its exact reciprocal (verified to 1.2e-15) and is **not** an
independent observable, so it is not reported as a separate outcome. `gHg` and
`gg` are declared secondaries with the same log-share decomposition.

Direction: gradient only. Certified curvature exists along the gradient
direction alone at every depth measured (I0001, I0004, telemetry-v4 §3), the
random and update directions certify at zero checkpoints, and the native bf16
arm certifies nowhere (DATASET caveat 6). Every claim here is about the fp32
shadow surface at θ_s, along the gradient.

Estimator, precommitted. Per run, with y_k = ln `vhv_gradient` at the certified
deep steps and p_k = (update index)/2520 — note that sparse post-update rows
carry step s+1, so the update index is `step − 1`:

- M0: y = a. M1: y = a + b·p. M2: y = a + b·max(0, p − τ).
- τ profiled over the run's own deep grid restricted to 0.05 ≤ p ≤ 0.95.
- τ̂ = the profile minimizer; its uncertainty is the profile-likelihood
  interval at the 95% level.
- A run's change point counts as **identified** if M2 beats M1 by the
  precommitted F criterion and the profile interval is at most 0.15 wide.

Probe partition: none is required. Curvature is measured at sparse deep
checkpoints on the frozen probes already in the run, not on a held-out
partition. Validation bpb is recorded per arm and reported descriptively.

## Decision rule, before the data

The primary statistic is the OLS slope of τ̂ on the arm's LR-decay onset across
arms A, B and C — nine runs at three onsets (0.250, 0.350, 0.650). The lock
hypothesis predicts slope 1 and intercept 0; the progress hypothesis predicts
slope 0. The intercept is the systematic lag between the schedule kink and the
curvature response, and is reported whatever the slope is.

- **Confirms the lock** if the 95% CI for the slope includes 1 and excludes
  0.5, **and** the per-arm mean |τ̄ − onset| is at most 0.05 (one grid spacing)
  in all three arms, **and** the change point is identified in at least 7 of
  the 9 runs.
- **Refutes the lock** if the slope CI includes 0 and excludes 0.5, **and** τ̄
  in all three arms lies within 0.05 of 0.350. Sharpening is then a property of
  progress, and the default recipe's onset coincidence was an accident.
- **Partial lock** if the slope CI excludes both 0 and 1, or if τ̂ tracks the
  onset in one direction only (moves late in C but not early in B, or the
  reverse). Reported as partial, with the direction named; not presented as
  either of the clean outcomes.
- **Inconclusive** if the slope CI includes both 0 and 1; or fewer than 7 of 9
  runs identify a change point; or any run has fewer than 6 certified points on
  either side of its onset, in which case that run's τ̂ is dropped and its arm
  is reported as underdetermined.

Secondary rules, also precommitted:

- **Arm D.** No identified change point and no consistent rise (Spearman of y
  on p not positive in all three seeds) ⇒ **LR decay is necessary** for the
  sharpening. An identified change point at 0.350 ± 0.05 ⇒ the **Muon momentum
  warmdown** is sufficient, and A/B/C's tracking must be re-read as momentum
  tracking. A rise with no identified change point supports neither and is
  reported as exploratory.
- **Arm E, the plateau.** Per run, the fold f = median of the last 8 certified
  points ÷ median of the certified pre-onset points (update index > 4) — the
  same phase split I0005 used, so f is directly comparable to its published
  4.76x. A 95% CI for ln f_A − ln f_E that excludes 0 and includes 0.93
  supports the LR-level reading; one that includes 0 and excludes 0.5 supports
  the model-ceiling reading; anything else is inconclusive. The 0.93 comes from
  extrapolating I0005's own numbers log-linearly (α = ln 4.76 / ln 20 = 0.52 ⇒
  f_E ≈ 1.9), and is a prediction from an assumed form, not a measurement.
- **LR collapse.** Centre each run's y on its own pre-onset mean and plot
  against ln of the LR multiplier. If curvature responds to the LR level, arms
  A, B, C and E collapse onto one curve there despite decaying at −1.08 to
  −2.71 per unit progress. Declared confirmatory-secondary; it is the
  mechanistic version of the primary test.

Everything else — per-layer structure, `dhd`, `update/*`, decoherence, loss —
is exploratory and will be labelled as such (DATASET caveat 10).

## Power, honestly

**The level comparison is hopeless and is not attempted.** At 3 seeds per arm
against I0001's 25% floor, a two-sample CI on a curvature level has a
half-width of roughly a factor of 1.8. No arm can be called sharper than
another. This is the same wall I0005 hit when it found plateau levels differing
by 15–19% inside a 29% reference band.

**The timing comparison is easy.** The predicted separation between arms B and
C is 0.400 in normalized progress, eight deep-checkpoint spacings. With the
three onsets crossed with three seeds, se(slope) = sd(τ̂)/0.51. If sd(τ̂) is one
full grid spacing (0.05) the slope has se 0.098, so slope 1 and slope 0 are
about ten standard errors apart and the confirm/refute decision is
overdetermined; the 95% CI is then about ±0.23, which separates 1 from 0.5 but
only marginally from 0.75. If sd(τ̂) is 0.03 the CI tightens to about ±0.14.
**The design decides sharply whether the change point tracks the onset; it
decides only weakly whether it tracks it fully or partially.**

That calculation depends on sd(τ̂), which no investigation has measured. It
costs nothing to measure, and doing so is a **prerequisite to freezing**: run
the estimator above on the five existing d12 runs, where I0005 says the break
is at 0.350, and record (i) whether it recovers 0.350 in all five, (ii) the
across-seed sd of τ̂, and (iii) the identification rate. If it does not recover
the known answer, the estimator is wrong and the design does not proceed. If
sd(τ̂) exceeds 0.075, add a fourth seed to A, B and C (+3 runs, +$7) before
freezing — decided from existing data, before any E02 run, so it is not a
post-hoc addition.

**The fold contrast needs its three seeds.** For the A-versus-E comparison the
relevant floor is the 23% across-seed spread of the within-run ratio measured
by I0005/A0001 — not the 25–29% level floor, because a per-run offset cancels
in a ratio. Against a predicted ln difference of 0.93, an exact two-sample
two-sided t at α = 0.05 gives about 45% power at 2 seeds per arm and about 93%
at 3. Three seeds is the minimum for this outcome, and is why the whole design
uses three rather than two.

**Loss is recorded, not tested.** The 0.06% loss floor makes final validation
bpb differences between arms trivially visible, but the arms are deliberately
different recipes — D and E never anneal and will end worse — so those
differences are descriptive. This design ranks no schedule.

**Threat to validity: differential certification.** Certification depends on
each run's own gradient magnitudes, so the arms could certify at different
rates and the comparison would then be over differently selected checkpoints.
Certified counts per arm are the first diagnostic reported, before any outcome;
a systematic between-arm difference in certification rate is itself a result
and must be stated alongside the primary.

## What this does not answer

- **Whether the LR or the Muon momentum is the driver in B and C.**
  `--warmdown-ratio` moves both. D breaks the tie in one direction only (LR
  flat, momentum decaying); the converse arm — momentum flat, LR decaying —
  does not exist, because the 0.97/0.90 endpoints and the 400-step ramp are
  hard-coded in `scripts/base_train.py`. Fully dissecting the two needs a
  recipe change this design does not make.
- **Anything about the loss landscape off the gradient direction.** No λ_max,
  no spectrum, no subspace. "The landscape sharpens" is not what is measured;
  "curvature along the gradient sharpens" is. And ηλ_max is not a valid
  stability statistic for Muon (telemetry-v4 §3), so no stability reading
  follows even if λ_max were available.
- **Anything about the bf16 surface the optimizer actually runs on.** Native
  curvature certifies nowhere; the claim is about the fp32 shadow surface.
- **Anything cross-depth.** d12 only. The larger d14/d16 ratios I0005 saw
  remain a hypothesis; DATASET caveats 1–3 apply and there is no seed reference
  above d12.
- **Whether one run is sharper than another.** Levels stay unresolvable.
- **Whether sharpening matters.** No link to loss, generalization, capability
  or optimizer stability is tested, and no schedule is recommended.
- **Whether the response is a kink or a smooth lagged response.** The estimator
  locates a kink; a constant lag would show up as slope 1 with a positive
  intercept, but a slowly-responding smooth mechanism is not excluded.
- **Anything about data order.** One ordering exists in the whole project, all
  arms share it, and no data-order floor is established here.

## Instrument dependencies

Schema v3 as it stands provides everything measured: the `shadow_fp32`
acceptance arm, per-direction verdicts, `curvature/{vhv_gradient, gHg, gg,
eta_star, verdict_code_gradient}` at sparse deep checkpoints, and a deep
schedule that already unions in `warmdown_start` as a landmark. No new metric
is required.

What is required is an **operations extension**, and without it these are not
official runs. `runs/telemetry_run.sh` refuses extra `base_train` arguments by
design, and the manifest row allowlist is `{depth, seed, shadow,
periodic_points, checkpoints, deep_schedule, head_dim}`. Freezing needs:

1. A new immutable manifest — `runs/manifests/sweep-d12-warmdown-v1.json` —
   with `warmdown_ratio` and `final_lr_frac` as validated float row keys in
   range, and the runner passing them through. Existing manifests are untouched.
2. `runs/verify_telemetry_run.py` extended to `--expect` the realized
   `derived.lr_schedule.warmdown_ratio` and `.final_lr_frac` and the resolved
   deep-step landmark, so the **treatment is verified in the artifact rather
   than assumed** — the failure mode that produced the wrong statement in the
   first dataset card.

From telemetry v4: item 5 (per-direction certification in the loader) is wanted,
since the entire analysis conditions on the gradient verdict; without it the
filter is applied by hand, as I0005 did. Item 1 (separated init and data seeds)
is wanted so the design matrix is machine-checkable, and if v4 has landed the
`data_seed` reproducing the legacy ordering must be fixed across all arms. Items
2, 3, 4, 6, 7 and 8 are **not** needed: no data-side treatment, no controller,
no sketch comparison, and no quoted noise scale, so v4's gate on `b_noise` is
not triggered.

## Open questions before freezing

1. What does the change-point estimator return on the five existing d12 runs?
   This is the design's power calculation and its estimator validation, it is
   free, and nothing should be run before it is answered.
2. Which instrument version do these runs use? If E02 runs post-v4 while the
   existing five runs are v3, arm A is a confirmation of I0005 on new runs but
   not a paired replication. Should one arm-A run reuse initialization seed 7,
   as an explicit software-drift and platform-nondeterminism control, at the
   cost of breaking the clean seed-block structure?
3. Is arm B's onset far enough from the other recipe transitions? At 0.250 it
   sits about two grid spacings after the 400-step Muon momentum ramp end
   (0.159). An earlier onset would lengthen the lever arm but risks the
   estimator confusing the two transitions; 0.250 was chosen over 0.150 for
   exactly that reason, and the choice should be checked against the answer to
   question 1.
4. Does arm D train stably for 2,520 steps at full LR? nanochat holds this LR
   for the first 35% of a default run, so it is plausible, but a divergence
   would make D's late series meaningless. A short non-scientific shakedown
   should confirm stability, and confirm that every arm produces the deep-step
   grid computed above.
5. Is the F criterion for M2-over-M1 the right identification test at ~25
   points, or should it be a residual bootstrap? The threshold must be a number
   in the frozen document, not a judgement made after seeing the fits.
6. Should `gHg` or `vhv_gradient` be primary? This draft chooses
   `vhv_gradient` because it is curvature per unit direction, at the cost of
   not being the channel I0005 headlined. If the two disagree on τ̂ that is a
   finding in itself, but the primary must be named before the data.
7. Is E's `--final-lr-frac 0.30` a large enough dose to separate the two
   plateau explanations while remaining a plausible recipe, and is the
   log-linear extrapolation behind the 0.93 prediction defensible enough to put
   in a decision rule?
