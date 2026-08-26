# E05 — decoherence intervention

Status: **draft, not frozen**.

## Question

Does implementation-level Muon decoherence change a training outcome at d12?
Operationally: if the compiled bf16 optimizer is replaced by a bf16 execution
of the same nominal Muon decomposition that agrees closely with its recorded
reference, does final loss on a sealed probe partition move?

The distinction between cause and check is the point of the design. The
**primary outcome is sealed-probe loss**. `muon/replay_update_relerr` is not an
outcome and a reduction in it is not success; it is the treatment check that
shows the intervention actually changed decoherence.

"Decoherence" here is necessarily reference-relative. The causal estimand is
the consequence of choosing the current compiled bf16 implementation rather
than a reference-locked bf16 implementation of the same algebra. It is not a
claim that distance from an abstract, unique exact update has causal agency.

## Why this intervention

The existing data establish the phenomenon, not its consequence. Compiled
bf16 Muon updates differ from a recomputed eager decomposition by roughly
3–10% per matrix, commonly about 3–5% on real mid-training gradients. I0003
also shows that most cross-matrix variation is organized by parameter role,
not layer position. Neither result says whether the difference changes
training.

The source narrows the useful controls:

- `optim.py` names the group field `ns_steps`, but the current operation is a
  five-stage Polar Express iteration. `gpt.py` hard-codes `ns_steps=5`; there
  is no `--muon-ns-steps` training flag today.
- Values below five are implementable by slicing the five stored coefficient
  triples. Values above five currently add no iteration at all. More
  coefficients would define a new algorithm.
- Changing the iteration count changes the intended orthogonalization as well
  as its rounding amplification. It is therefore not a clean manipulation of
  decoherence and is held at five in this design.
- In the production kernel, `X` is cast to bf16 before MuonEq and Polar
  Express when `COMPUTE_DTYPE` is bf16, then cast back before Muon+
  renormalization and factored scaling. An fp32-`X` arm is feasible, but it
  changes both reference agreement and the numerical update itself.

The main contrast therefore stays in bf16 and changes execution/fusion and
rounding placement only. A separate fp32 arm helps interpret that contrast,
but cannot by itself identify an effect of decoherence.

## Design

One factor at fixed d12, with three levels:

| arm | Muon execution | `X` dtype through MuonEq and Polar Express | stages | purpose |
|---|---|---:|---:|---|
| A | current compiled fused path | bf16 | 5 | high-decoherence production control |
| B | reference-locked operation sequence | bf16 | 5 | low-decoherence primary intervention |
| C | compiled fused path | IEEE fp32 | 5 | low-decoherence precision triangulation |

Arm B applies the versioned eager bf16 reference operation sequence as the
optimizer update, including the same momentum, MuonEq, coefficients, Muon+
renormalization, factored second moment, cautious decay, shape-adjusted
learning rate, and state transitions as A. Its production implementation must
not call or share the final-update implementation with the telemetry replay;
otherwise a near-zero treatment metric would be true by construction rather
than a check. Equivalence to the independently implemented eager reference
must be demonstrated before freezing. The only intended difference from A is
execution/fusion and its rounding points.

In C, only the working tensor `X` from its initial cast through the last Polar
Express iteration is fp32. Model forward/backward, gradients, parameters,
momentum, post-polar Muon stages, and AdamW remain on the ordinary recipe.
"IEEE fp32" means TF32 is disabled for those optimizer matmuls; if that cannot
be scoped without changing model computation, C must be relabeled as fp32/TF32
or removed before the design freezes.

Blocking: **8 new initialization seeds**, with all three arms run from each
seed. The initial parameter tensors must be bit-identical within a block and
their hashes recorded. Arm order is randomized and balanced over wall-clock
order. This is **24 runs**.

Held fixed: depth 12 and width 768, model architecture, initialization within
block, bf16 model arithmetic, optimizer hyperparameters other than the stated
execution mode, five Polar Express stages, data order, logical batches, token
budget, LR/momentum/weight-decay schedules, tokenizer and dataset snapshot,
sealed-probe artifact, evaluation code and batch order, telemetry cadence,
PyTorch/CUDA versions, H100 SXM hardware, and compile settings outside the
intervention. Each arm gets a new immutable manifest; no used sweep manifest
is edited.

The precommitted primary contrast is B−A, paired by initialization seed. C−A
and C−B are mechanistic secondary contrasts and do not determine the main
verdict.

At the ordinary estimate of **$2.30 and 40 minutes per d12 run**, 24 unchanged
runs would cost about **$55 and 16 GPU-hours**. Only the Muon `X` subpath, not
the model, becomes fp32 in C. Until a non-outcome timing shakedown replaces
the estimate, budget C at **15% extra: about 6 minutes and $0.35 per run**, or
about **$2.80 extra across its eight runs**. Budget the reference-locked B arm
at the same ceiling for lost fusion and extra launches. The campaign budget is
therefore about **$61 and 17.6 GPU-hours**. An overrun above 15% is a reason to
revise and refreeze the design, not to silently change the arm.

## Primary outcome

Precommitted before the first run: final mean cross-entropy loss on the
**sealed evaluation probe partition**, reported also as the paired relative
change `100 * (L_arm / L_A - 1)`. The estimator is the mean of the eight
within-seed contrasts. There is one primary checkpoint: the final checkpoint
after the fixed token budget.

The sealed partition is a fixed, hashed artifact shared by every run. Its
index is unavailable to training and monitoring code and is opened only by
the offline evaluator after all 24 runs finish. Training loss, the monitoring
probe, best-checkpoint selection, and trajectory summaries are not substitutes
for this endpoint and cannot change the verdict.

## Treatment check, not an outcome

For each run, define the decoherence summary before opening sealed outcomes:

1. At every preregistered deep checkpoint with normalized progress in
   `[0.25, 0.75]`, take the median `muon/replay_update_relerr` over matrices
   with a defined, nonzero actual update.
2. Take the median of those checkpoint medians.
3. Compare B/A and C/A within initialization blocks.

The B manipulation passes if its median paired ratio to A is at most 0.5 and
B is below A in at least 7 of 8 blocks. The same rule checks C separately.
Role-stratified values must be reported because I0003 found strong role
structure, but they are descriptive and do not replace the all-matrix check.
Initialization and exactly-zero first updates are excluded by the progress
window.

The treatment check is evaluated before sealed loss is read. If B does not
pass it, the causal contrast failed regardless of the loss result.

## Decision rule

Compute a two-sided paired t confidence interval over the eight B−A relative
sealed-loss contrasts. Also report the paired median and an exact sign-flip
sensitivity analysis, but do not choose among them after seeing the result.

- **Decoherence matters at d12:** the B treatment check passes and the 95%
  interval excludes zero. The sign says whether the compiled path helped or
  hurt; the design does not presume that lower decoherence is better.
- **No material effect at this design's resolution:** the B treatment check
  passes, the 95% interval includes zero, and the 90% interval lies wholly
  inside the precommitted equivalence band of ±0.10% relative loss. This means
  no effect of 0.10% or larger was resolved, not proof of exact equality.
- **Inconclusive:** the treatment check fails, or the loss interval overlaps
  both zero and either equivalence boundary. A loss change when the treatment
  check fails is also inconclusive, not evidence against decoherence.

C is interpretation, not a second chance at the primary test. If A differs
from B while B and C are equivalent, the pattern supports an
execution/rounding explanation. If A and B agree but C differs, the result is
evidence for fp32 arithmetic, not for decoherence. If all three differ, or B
and C do not both reduce the treatment metric, the two mechanisms remain
entangled.

## Power, honestly

I0001 (`analysis/investigations/0001-seed-variation/conclusion.md`) establishes
a **0.06% sd-relative initialization floor for `loss/train_mean`** at d12 and
a **3.5% floor for `muon/replay_update_relerr`**. With eight paired blocks, a
0.06% within-pair loss standard deviation implies roughly 80% power for a
0.07% relative shift; conservatively treating the two arms as independent
raises that to about 0.10%. This is the basis for the ±0.10% equivalence band.
Pairing should help, but its correlation is not known in advance.

There is an important mismatch: I0001 measured **0.16%**, not 0.06%, for the
existing `probe/loss` channel, and the new sealed v4 partition has no empirical
floor yet. If its paired standard deviation is 0.16%, the detectable shift is
roughly 0.18–0.26%, and this design cannot support the planned 0.10%
equivalence claim. The eight A runs will report that floor, but the run count
and decision thresholds cannot be adapted after sealed outcomes are opened.
Before freezing, either justify 0.06% for the larger sealed evaluator with a
non-outcome calibration or increase the seed count. Quoting the train-loss
floor as though it had already been established for a sealed probe would be
false precision.

The decoherence intervention should be easy to verify: a 50% treatment shift
is about fourteen times the 3.5% channel floor. That statistical power does
not turn the treatment check into a training outcome.

## What this does not answer

- Whether an abstract or uniquely correct Muon reference exists. The result
  is relative to one versioned eager decomposition.
- Whether fp32 is better because it is more accurate or because it is more
  reference-coherent. C changes both. B supplies a cleaner bf16 contrast and
  the three-arm pattern can triangulate, but no comparison makes C's two
  changes separately identifiable.
- What happens with four stages, more than five stages, different Polar
  Express coefficients, or a genuinely converged Newton–Schulz iteration.
- Which parameter roles cause any outcome change. Role-resolved decoherence is
  observed, not independently manipulated.
- Anything cross-depth, cross-width, on another accelerator/backend, or for a
  longer token budget.
- Whether curvature or update-effectiveness changes. I0002 shows why bf16
  validation and late update-effectiveness are fragile; neither is an outcome
  here.
- Whether any loss effect improves downstream task quality or just the sealed
  pretraining distribution.

## Instrument dependencies

- **Telemetry schema v3, retained:** actual-update capture and per-matrix
  `muon/replay_update_relerr` at deep cadence, defined-row handling, parameter
  identity and role, optimizer hyperparameters, immutable manifests, and the
  run verifier.
- **Telemetry v4 item 4:** immutable monitoring and sealed evaluation probe
  partitions with hashes and purpose restrictions. No other v4 data-side
  capability is required because data order is held fixed.
- **Telemetry v5 (required before freezing E05):** record the Muon execution
  mode, working dtype, matmul precision/TF32 policy, and stage count in
  provenance and optimizer channels; compute the replay reference with the
  arm's declared working dtype; and give each reference policy a distinct
  versioned `estimator_id`.

The last item is not optional. Today `_MUON_REFERENCE_ID` and `muon_stages`
select bf16 versus fp32 from the module-global `COMPUTE_DTYPE`. That is wrong
for C, where model compute remains bf16 while only Muon's `X` is fp32. Running
this design on the current reference code would make the treatment check
mislabel its own arithmetic. The new paths also need CPU equivalence tests,
H100 compiled/eager calibration, and the existing non-perturbation and tamper
gates before any outcome run.

## Open questions before freezing

1. Can an independently implemented B match the reference closely while
   preserving production state mutation, without making the treatment check
   tautological or perturbative?
2. Can IEEE fp32 be scoped to C's Muon graph with TF32 disabled while leaving
   model matmuls untouched? What are the measured step-time and memory costs,
   and do they fit the 15% budget ceiling?
3. Can the sealed evaluator demonstrate a 0.06% paired initialization floor?
   If it remains near the existing probe's 0.16%, how many more seeds are
   affordable, and should the equivalence band be widened before freezing?
4. Is a 50% reduction in the preregistered decoherence summary achievable in
   both low-decoherence arms without changing any post-polar Muon stage?
5. Should a four-stage arm be a separate follow-up design? It may provide a
   decoherence dose, but it also changes the intended update and would weaken
   the causal interpretation here.
6. Is `telemetry v5` the correct version boundary for the per-arm reference
   policy, or will that capability be assigned another frozen schema version?
