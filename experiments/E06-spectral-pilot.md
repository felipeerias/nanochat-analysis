# E06 — spectral probe-sizing and Lanczos pilot

Status: **draft, not frozen**. Runs offline on saved checkpoints. Needs no new
training and no instrument change.

## Question

At d12, what probe size and Lanczos configuration give a **stable** estimate of
λ_max and of the top-k invariant subspace, and what does that cost?

An answer either way is useful. If 16 rows at T=256 already meet the
precommitted stability criterion, spectral quantities become cheap enough to
put on the deep-checkpoint schedule. If 64 rows do not meet it, spectral
quantities are offline-only for the foreseeable future and any design that
wanted them must say so before it is written.

## Why this design is unusual, and why it comes now

Every other design in this folder generates training data. This one measures
the **instrument**. It consumes GPU time but no *training* time: it reads the
model+optimizer lineage checkpoints already sitting on the network volume and
computes on them.

It exists because the project is about to start quoting spectral quantities it
has never sized. Today the entire certified curvature record — the sharpening
trajectory of I0005, the seed reference of I0001 — rests on a **one-row** probe
at T=256 (`telemetry.py:2484`, `self._hvp_x, self._hvp_y = sx[:1], sy[:1]`).
That was a defensible choice for a scalar directional quantity measured inside
a training step. It is not a defensible basis for λ_max, and it is certainly
not one for a top-k subspace, where the number of rows and the dimension of the
subspace are the same order.

Sizing this before an architecture study is the cheap ordering. Sizing it
afterwards means discovering that an expensive factorial measured a property of
four particular rows.

## Standing constraints this design is written under

These come from `../telemetry-v4-plan.md` §3 and are not negotiable here.

1. **Averaging λ_max across small probes is wrong.** λ_max is a supremum of
   linear functionals of H, hence convex in H, so by Jensen
   `mean_j λ_max(H_j) ≥ λ_max(mean_j H_j)` — the naive average is not noisy,
   it is **biased upward**, and the bias grows as the chunks shrink. The only
   legal averaging is inside the operator: `Hv := Σ_j w_j (H_j v)`, assembled
   before each Lanczos iteration consumes it. §"The operator" below states the
   contract and the exactness check that guards it.
2. **Quoted spectra come from the shadow fp32 arm.** A Krylov sequence
   compounds small operator errors: they perturb Ritz ordering and accelerate
   loss of orthogonality, so the single-HVP 0.34% bf16 accuracy of I0002 does
   not transfer to a 50-iteration recurrence. bf16 spectra are a **precision
   study at two anchors**, never the headline.
3. **A gradient-direction verdict does not certify the top eigendirection.**
   I0001 established that among shadow-arm checkpoints only the *gradient*
   direction ever passes. The Ritz direction is a new direction and must earn
   its own per-direction verdict. §"Certifying the Ritz direction" says how.
4. **ηλ_max is not a valid stability statistic for Muon.** This is not an
   edge-of-stability study, and nothing here may be read as one. The Hessian
   spectrum is a landscape descriptor.

## Checkpoints and anchors

Full model+optimizer lineage checkpoints exist for all seven collected runs:
θ_0 plus three interiors plus the final triplet, hash-inventoried, on the
Runpod volume `nanochat_experiment` (AP-JP-1, ~70 GB). **They are not in the
local copy** — this design runs on a pod with the volume mounted.

d12 labels are `[0, 631, 1261, 1890]` plus the recipe's final save at 2520.
Warmdown begins at step 882 (progress 0.350), so:

| anchor | label | progress | regime | role |
|---|---:|---:|---|---|
| A0 | 0 | 0.000 | initialization | stress case only (54 structurally zero-gradient matrices, I0007) |
| A1 | 631 | 0.250 | constant LR, gHg flat | secondary + audit |
| A2 | 1261 | 0.500 | mid-warmdown, gHg rising | secondary |
| A3 | 2520 | 1.000 | plateau | **primary** + audit |

A3 is primary because a future architecture study will most likely compare
final models. A1 and A2 exist to check that the chosen size holds across the
regimes I0005 showed are genuinely different. Label 1890 is available and
unused, to hold cost down. The final checkpoint is stored in the recipe's
`model_/optim_/meta_` triplet format, not the `ckpt_NNNNNN.pt` lineage format;
the harness must read both.

## Factors, levels, held fixed

**Factor 1 — probe size m ∈ {4, 16, 64} rows at T=256, nested.** Eight
mutually disjoint banks per size, with nesting *within* replicate:
bank r's 4 rows ⊂ bank r's 16 rows ⊂ bank r's 64 rows. Nesting removes the
confound between the size effect and the bank draw; disjointness across r is
what makes the between-bank spread an honest estimate.

**Factor 2 — Lanczos iterations k**, adaptively extended (see below), which
makes k an *outcome* as much as a factor.

Banks are a deterministic, salted, index-based partition of the val stream
(the loader contains no RNG — I0008), materialized with `save_probe()` so each
carries a content-derived `probe_id`, and frozen before the first evaluation.
They are drawn from the **monitoring** partition; the sealed partition is never
touched by an instrument-calibration measurement.

Two extra banks, neither part of the sizing decision:

- **Bank L (legacy)** — the four rows of the existing frozen `probe_short`,
  plus its four singleton sub-banks. This ties the pilot to the published
  record: it says how much of the I0005 trajectory is a property of one row.
- **Bank C (confirmation)** — a further **disjoint** bank at the chosen size
  m*, drawn but never evaluated until the sizing rule has already fired.

Held fixed: depth 12, T=256, the fp32 shadow arithmetic, the chunk size
(c = 4 rows at every bank size, so only the *number* of chunks varies and the
arithmetic path is identical across sizes), the reduction order, the start-vector
generator, the reorthogonalization scheme, and the checkpoint set.

## The operator

For a bank B of m rows, `L_B(θ)` is the model's own batch loss over the whole
bank, and `H_B = ∇²L_B(θ)`. HVPs use the existing double-backward
(`telemetry.py:1102`), which forces math-SDPA — flash attention has no
backward-of-backward — and contracts g·v in fp64 inside the graph.

Chunking contract: with chunks of c = 4 rows,

```
Hv := Σ_j w_j (H_j v),   w_j = n_valid_targets(chunk j) / n_valid_targets(B)
```

so that `Σ_j w_j L_j = L_B` exactly. **Verification, precommitted:** at each
anchor, compute `L_B` unchunked at m=4 and assert
`|Σ_j w_j L_j − L_B| ≤ 8·eps_fp32·|L_B|`. This is the guard that the illegal
averaging never silently reappears.

**Precommitted illustration of constraint 1.** At A3, on one m=4 bank, report
`mean_i λ_max(H_{row i})` beside `λ_max(H_bank)`. The Jensen gap is a
one-number demonstration on real data of why per-probe λ_max must not be
averaged. It is an illustration, not an outcome.

## Lanczos configuration

- **Start vector**: normalized Gaussian via the existing
  `_normalized_direction_like` with a recorded seed. Note the geometry: with
  N ≈ 286.3M parameters a random start has ~6e-5 overlap with the top
  eigenvector, so the start vector is not a formality. At the two **audit
  anchors** (A1, A3) every configuration is run twice with independent start
  seeds; the difference between those two runs is a pure numerical-reproducibility
  term with no probe-sampling content.
- **Reorthogonalization is mandatory and full** (DGKS, one repeat), inner
  products accumulated in fp64 per the codebase's dot contract. Without it,
  fp32 loss of orthogonality manufactures ghost Ritz values and the top-k
  subspace becomes meaningless.
- **Iteration schedule**: start at k=20; extend in blocks of 5 to at most 50.
  Stop when the explicit relative residual on the top Ritz pair is ≤ 1e-2
  **and** θ_1 moved < 0.5% over the last block. At k=50 stop regardless and
  record `converged=false`. The rule is precommitted so k is measured, not
  tuned.
- **Memory contract**: the basis is 50 × 4N bytes = **57.3 GB** in fp32, which
  fits in an 80 GB H100 beside the fp32 shadow model (1.15 GB) and c=4
  activations. Host-pinned streaming is the fallback and costs roughly two
  orders of magnitude more per iteration; which path ran is recorded. This
  contract is the reason the pilot is d12-only: d14 needs 80 GB of basis and
  d16 needs 107 GB.

Diagnostics recorded for every Lanczos run:

| diagnostic | why it is kept |
|---|---|
| Ritz values per iteration (full trajectory) | convergence rate; ordering changes |
| recurrence residual `β_k·|s_{k,i}|`, i=1..10 | cheap per-pair convergence bound |
| **one explicit final residual** `‖Hv_1 − θ_1 v_1‖/|θ_1|` | the recurrence bound is not a measurement |
| basis orthogonality error `max_{i<j}|q_i·q_j|`, `‖QᵀQ−I‖_F` | detects reorthogonalization failure |
| estimated eigengap `θ_1 − θ_2`, and `‖r‖/gap` | the eigenvector angle bound; flags clusters |
| β_j trajectory and breakdown status | lucky vs numerical breakdown; recorded, never restarted |
| start seed | reproducibility of the audit pair |
| ghost count (duplicated Ritz values within tol) | the ghost canary itself |
| λ_min | H is not PSD; extremal-negative convergence comes free and is worth having |
| seconds per HVP, seconds per reorthogonalization pass, peak bytes | the cost model v5's online Lanczos needs |

At the audit anchors, explicit residuals are computed for Ritz pairs 2..5 as
well (4 extra HVPs), because an unconverged 5th pair makes a top-5 subspace
comparison meaningless. The subspace criterion is applied only to the largest
j ≤ 5 for which pairs 1..j are converged, and j is reported.

**Fault injection, per v4 §4.** One Lanczos run at A3 with reorthogonalization
**disabled**, at m=4 and at m=64. If the ghost count and the orthogonality
error do not fire there, those two canaries are not kept.

## Primary outcome, precommitted

**Between-bank coefficient of variation of λ_max**, shadow fp32, at A3, at each
size: `CV_bank(m) = sd_r λ_max(H_{B_r}) / mean_r λ_max(H_{B_r})` over the 8
disjoint banks, reported with its **one-sided upper 90% bound**
(`×√(7/χ²_{0.10,7}) = ×1.57` — with 8 banks the CV is itself imprecise and the
point estimate must not be the decision variable).

Secondary outcomes, all precommitted: between-bank top-j subspace overlap
`(1/j)‖VᵀW‖²_F`; between-bank CV of `‖∇L_B‖` and pairwise gradient cosine;
between-bank CV of `gHg` (which connects directly to the existing certified
channel); iterations-to-convergence; seconds per HVP; peak memory.

**Decision rule.** Choose the smallest m ∈ {4, 16, 64} such that all three hold:

1. upper 90% bound on `CV_bank(λ_max)` ≤ **10%**;
2. **no detectable size trend**: `|mean λ_max(m) − mean λ_max(m/4)|` ≤ the
   between-bank sd at m — a CV-only rule would happily select a size that is
   stably wrong, since finite-m Hessians have an upward-biased top eigenvalue;
3. lower bound on mean top-j overlap ≥ **0.8** at j ≥ 3.

If no size qualifies, that is the result. Report the projection: for
exchangeable rows `CV ∼ m^{-1/2}`, so the measured CV at 64 gives the required
m directly. The nested design also **tests** that scaling — CV should halve
from 4→16 and again 16→64. If it does not, per-row curvature is heavy-tailed
and the rows are not exchangeable, which is a finding in its own right.

Bank C is evaluated only after the rule has fired, as an out-of-sample check
that the criterion holds on a bank that had no part in choosing m*.

## Why 10%, tied to a future effect size

The criterion has to be tied to something. It is tied to what an architecture
study at fixed depth would need.

Take that study as d12, two arms, 6 paired initialization-seed blocks, outcome
log λ_max at a fixed progress anchor. At α=0.05 two-sided and 80% power a
paired design with n=6 detects about `(t_{.975,5}+t_{.80,5})/√6 ≈ 1.43`
within-block standard deviations. The within-block sd is
`σ_within ≈ √(σ_seed² + σ_bank² + σ_platform²)`.

σ_seed for λ_max is unknown. The nearest established channel is `gHg` at
**29% sd-relative across five d12 seeds** (I0001) — so, provisionally, 6 blocks
detect roughly a 40% change in λ_max. Requiring `σ_bank ≤ σ_seed/3` inflates
σ_within by `√(1+1/9) = 1.054`: the measurement never costs more than ~5% of
detectable effect size. 29%/3 ≈ 10%, which is the number above.

Two honesty notes on this.

- σ_seed for λ_max is **measured by this pilot**, not assumed (see below), so
  the criterion is re-derivable once the pilot lands. If λ_max turns out much
  more stable across seeds than gHg, 10% is too loose and the rule should be
  re-cut before any architecture study freezes.
- If every arm shares the same bank — which they would, since probes are
  already shared across seeds (DATASET caveat 6) — then bank sampling error is
  **common-mode and cancels** in the paired contrast. So why size at all? Because
  cancellation holds only for a *fixed* operator. The treatment moves θ, which
  moves H, and the finite-m bias is therefore treatment-dependent: it does not
  cancel, and its residual is unbounded a priori. And the top-k subspace is not
  a scalar; a bank too small to resolve the operator makes a cross-arm subspace
  comparison a comparison of bank idiosyncrasies. The criterion is a
  **resolution** criterion. That is exactly why it is set at a third of the seed
  floor and not at some far stricter value.

## Four variance components, all measured

This is what makes the pilot worth its cost beyond the sizing decision.

| component | how it is isolated | cost |
|---|---|---|
| σ_platform | same bank, same start seed, rerun | 4 reruns |
| σ_start | same bank, two independent start seeds, at A1 and A3 | 48 runs |
| σ_bank | 8 disjoint banks at each size (the criterion) | the sizing grid |
| σ_seed | **five d12 seeds**, one fixed bank at m*, three anchors | 12 runs |

σ_platform is not hypothetical: the compiled embedding backward contains an
atomic-accumulation race (DATASET caveat 7), so identical inputs need not
reproduce bitwise, and an HVP goes through that backward twice. Without the
rerun control, platform jitter would be silently attributed to probe sampling.

σ_seed is the **first measurement of the initialization noise floor for
λ_max**, and it is nearly free: the probes are shared across the five d12 seeds
and the checkpoints already exist. It is the number a future architecture
study's power calculation actually needs.

## Certifying the Ritz direction

The verdict machinery is per-direction and `hvp_acceptance` already accepts
`extra_directions` (`telemetry.py:1307`). So:

- After Lanczos converges, run the existing acceptance suite on the shadow arm,
  at the same θ and the same bank, with `extra_directions={"ritz_top": v_1}`.
  That produces `curvature/verdict_code_ritz_top` from the same four checks —
  FD-of-gradient, symmetry, linearity, and the scalar second difference —
  under the same tolerance version as every other direction in the dataset.
- Independently, the Lanczos-native check: the **explicit** residual
  `‖Hv_1 − θ_1 v_1‖/|θ_1|`, giving `|λ_1 − θ_1| ≤ ‖r‖` and the eigenvector
  angle bound `sin∠(v_1,u_1) ≤ ‖r‖/gap`; plus a Rayleigh-quotient consistency
  check `θ_1` against an explicitly recomputed `v_1ᵀHv_1`.

A precommitted prediction, which makes this falsifiable: the random and update
directions fail their verdicts for lack of curvature signal (I0001), so the
Ritz direction — the direction of *maximal* curvature — should have the best
SNR in the whole suite and should pass wherever the gradient direction passes.
If it does not, something is wrong with either the Lanczos output or our
understanding of why the suite fails elsewhere, and that is worth knowing
before any spectral quantity is quoted.

## Precision: which arithmetic produces the number

**Quoted spectra are shadow fp32 only.** The harness reproduces the existing
shadow construction offline: load the checkpoint's model state, build the fp32
copy with `build_shadow_model` under `shadow_precision(torch.float32)` (TF32
off, matmul precision "highest", rotary rebuilt in shadow arithmetic),
`estimator_id = "hvp-shadow-fp32-ieee-v1"`.

**bf16 is a precision study at exactly two anchors** (A1, A3), one bank, at m*,
one start seed. Reported: relative difference in λ_max, top-j subspace overlap
against the fp32 result, orthogonality error, ghost count, and iterations to
convergence. The hypothesis being tested is compounding: I0002 measured 0.34%
median distortion on a *single* HVP, and a 50-step Krylov recurrence should be
worse. How much worse is the deliverable.

There is no fp64 reference — `_SHADOW_DTYPES` supports fp32 only, because the
forward pins rotary and logits to fp32. So fp32 correctness is *assumed*,
backed by the acceptance suite rather than by a higher-precision computation.
Say so wherever these numbers are quoted.

## Number of evaluations and cost

No training runs. The unit is a Lanczos evaluation.

| block | runs |
|---|---:|
| sizing grid (3 sizes × 8 banks × 3 anchors, start seed A) | 72 |
| second start seed at audit anchors A1, A3 | 48 |
| θ_0 stress (3 sizes × 2 banks) | 6 |
| legacy bank L + its four singletons, 3 anchors | 15 |
| platform repeatability reruns | 4 |
| no-reorthogonalization fault injection | 2 |
| **Stage 1** | **147** |
| confirmation bank C at m*, 3 anchors, 2 starts | 6 |
| seed floor: 4 further d12 seeds × 3 anchors at m* | 12 |
| bf16 precision study | 2 |
| **Stage 2** | **20** + 6 acceptance suites |

Cost estimate, to be replaced by measurement — cost per HVP is an outcome, not
an input. At d12, N ≈ 286.3M parameters and ~110.1M of them in matmuls, so
fwd+bwd ≈ 6.7e8 FLOP/token at T=256 and an HVP is ~2.5× that. At m=64 that is
~27.5 TFLOP per HVP; on an H100 with TF32 disabled (67 TFLOP/s peak fp32, call
it 35% achieved) ≈ **1.2 s per HVP**, so ~0.3 s at m=16 and ~0.07 s at m=4.
Subspace overlaps are computed as one 40×40 Gram matrix per (anchor, size)
cell, streaming the eight banks' top-5 bases in a single pass, ~90 s per cell,
after which the bases are deleted; only the overlap matrix and a hash survive.

**About 2.5 GPU-hours of compute, ~3.5 h of pod wall-clock, ~$12 at $3.29/h,
and zero training GPU time.**

## What this does not answer

- **Whether λ_max or the top-k subspace is a useful outcome.** This is
  calibration. It tells you what the number costs and how stable it is, not
  whether it responds to anything.
- **Anything at d14 or d16.** The 57.3 GB basis contract fails at both (80 GB
  and 107 GB), and the m-dependence of the estimand is architecture-specific
  anyway. Do not transfer m*.
- **Sequence length.** T is fixed at 256 throughout, which means `H_probe` is
  the Hessian of a *different function* from the training loss, which is
  computed at `max_seq_len = 2048`. The pilot says nothing about that gap. This
  may be the most important thing it does not answer.
- **Clustered top eigenvalues.** Single-vector Lanczos converges one vector per
  cluster. The recorded eigengap is what flags the problem; block Lanczos is
  the fallback, and it is not in this design.
- **Anything about stability, step size, or edge of stability.** Per v4 §3,
  ηλ_max is not a valid Muon statistic, and no quantity here should be composed
  with a learning rate.
- **Bias with respect to the population Hessian.** The design measures the
  size trend across 4→16→64 and can extrapolate it; it has no ground truth.

## Instrument dependencies

**None new. This design deliberately consumes only frozen-instrument output.**
It uses, at their current versions: `hvp` (`telemetry.py:1102`),
`hvp_acceptance` with `extra_directions` (:1307), `build_shadow_model` (:1988),
`shadow_precision` (:1963), `save_probe`/`load_probe` (:1841/:1849),
`canonical_named_parameters`, the `Record` schema (v3) and `TelemetryWriter`.

The Lanczos driver, the chunked-HVP assembly, and the bank construction are new
**offline** code that never touches the training path, so the instrument stays
frozen and the non-perturbation contract is not in play.

Records are written at tier `offline` under a new `spectrum/*` namespace, which
adds no required columns to the schema; `probe_id`, `checkpoint_id`,
`acceptance_arm`, `estimator_id` and `dtype` are populated voluntarily. The
Ritz verdict is written as `curvature/verdict_code_ritz_top`, which does obey
the curvature-namespace validation and gets `acceptance_arm`,
`acceptance_status`, `tolerance_version` and `backend` from the suite itself.

E06 **produces an input to** telemetry v4 item 7 (it decides what "at least 64
rows at T=256" should actually be) and to the v5 deferred item "online Lanczos
and spectral density" (it produces the cost model that decides whether online
is affordable at all).

## Open questions before freezing

1. **T=256 versus the 2048-token training loss.** The probe measures the
   Hessian of a function the model is not trained on. Is a short-context
   spectrum the quantity we want, or should the sizing axis be *tokens*
   (m·T) rather than rows? Answering this may require a second factor and would
   change the cost table substantially.
2. **Is 8 banks enough?** The CV of a CV at 8 samples carries ~27% relative
   uncertainty, which is why the rule uses the upper 90% bound. Twelve banks
   would tighten it to ~21% at roughly 50% more Stage-1 cost. Worth it?
3. **Which anchor should be primary?** A3 was chosen because architecture
   studies compare final models. But I0005 showed the interesting dynamics live
   in the warmdown, and if the answer at A2 differs from A3 the design has a
   size that depends on regime and no rule for that case.
4. **What if σ_seed(λ_max) comes back much smaller than gHg's 29%?** Then the
   10% criterion is too loose and Stage 1's decision was made against the wrong
   yardstick. Should Stage 2's σ_seed measurement be moved *before* the sizing
   decision, at the cost of an extra checkpoint-loading pass?
5. **k=50 as a hard cap.** It is set by the 57.3 GB memory contract, not by the
   spectrum. If convergence at m=64 routinely wants more than 50 iterations,
   the design has no answer except a different algorithm — thick-restart or
   block Lanczos — which changes the estimand and would need its own id.
6. **Does the Ritz direction's acceptance verdict mean the same thing as the
   gradient direction's?** Both use the same thresholds and the same tolerance
   version, but the FD epsilon sweep was tuned on directions with far less
   curvature. The sweep range may need re-cutting for a direction that is, by
   construction, the steepest one available.
7. **Bank L's singletons** will show how sensitive the published I0005
   trajectory is to the single row it used. If that sensitivity is large, does
   the existing certified curvature record need a caveat added to `DATASET.md`
   before this pilot even finishes?
