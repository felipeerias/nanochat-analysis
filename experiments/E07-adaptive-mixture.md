# E07 — adaptive data mixing under a value-based controller

Status: **PROPOSAL — pending empirical grounding from E01.** Not a draft
design, not frozen, and not runnable. It depends on telemetry v4 plus two
items v4 explicitly defers to v5.

This document exists for one reason: writing it now says what
[E01](E01-packing-composition.md) must deliver and what the instrument must
record, before either is built. It is filed under `experiments/` because it
generates data, but it must not be treated as a design until the freeze
conditions at the end are met.

## Question

At a fixed token budget, does changing the training mixture over the course of
a run — under a policy driven by a measured per-group quantity — lower final
held-out loss relative to the **best static mixture**, and is any gain
attributable to the *timing* of the change rather than to the run-level average
mixture it happens to produce?

The comparator is the best static mixture, not the nominal one. Against
nominal, an adaptive controller that drifts to a better average composition and
stays there would "win" while demonstrating nothing about adaptivity. E01
produces that comparator; without it this question cannot even be posed.

## The theory this comes from, and how far it survives contact

The proposal in `sol-batch-construction-question.txt` models batch
construction as optimal control. For data group *k* with gradient
`g_k(t) = ∇_θ L_k(θ(t))`, the future value of training on *k* is
`s_k(t) = λ(t)ᵀ g_k(t)`, where λ is the costate/adjoint of the final objective
with respect to θ(t); the ideal sampler is `q_k(t) ∝ p_k exp(β s_k(t))`.

[I0008](../investigations/0008-adaptive-batching-feasibility/conclusion.md)
tested what of this is measurable and returned three findings that constrain
any experiment built on it:

- **λ is not estimable here.** The nearest surrogate is the myopic costate
  `λ̂ = −∇L_probe(θ_s)`, which A0001 rates *poor* by construction: it is the
  direction the optimizer already follows and carries no information about the
  remaining trajectory. Recovering the true adjoint needs a backward pass over
  the whole run, which the checkpoint budget does not support.
- **A first-order value model is insufficient.** Tested against realized loss
  change, `update/p1 = gᵀΔ` gives R² = −0.57 (97% median magnitude error);
  adding the curvature term gives R² = 0.87 (2.5%).
- **Noncommutativity is unavailable.** With one loss the bracket is identically
  zero, and `curvature/e_sym_*` is an arithmetic diagnostic, **not** a
  commutator. Group-restricted HVPs are deferred to telemetry v5.

So E07 does not test the optimal-control theory. It tests a **myopic,
first-order instantiation** of it, which the evidence we have predicts will
fail. That prediction is the reason the design below is built around positive
controls rather than around the controller.

## Why a null from this design would ordinarily mean nothing

E01 already states the general form of this problem: an adaptive-mixing null
risks meaning "this particular policy did not help" rather than anything about
adaptive mixing. E07 inherits **two** independent sources of that ambiguity,
not one:

1. **The policy.** `q_k ∝ p_k exp(β ŝ_k)` with a myopic λ̂ is one point in a
   large space, and I0008 gives us reason to think it is a bad one.
2. **The taxonomy.** Group value is defined relative to a grouping. A null
   under a taxonomy that does not separate materially different data is a fact
   about the taxonomy, not about mixing. E07 does not get to choose here: it
   must use the versioned taxonomy E01 freezes, or the two campaigns are not
   comparable.

A design that cannot distinguish "the channel is closed" from "this policy is
bad" from "this taxonomy is uninformative" is not worth running. Everything
structural below follows from closing that gap.

## Making a null mean something: a ladder of policies

Three policy classes are run, in increasing order of information available to
them. Each has a stated rationale and a stated role in interpretation.

### A1 — the controller under test (measured quantity, stated rationale)

`q_k ∝ p_k exp(β ŝ_k)` with `ŝ_k = λ̂ᵀ ĝ_k`, all quantities measured:

- **ĝ_k** comes free from group-pure accumulation microbatches (I0008 Tier 1),
  EMA-accumulated across steps. The half-life is **not** a free parameter: it is
  set by I0008's sizing rule, which requires 15 rows per group early rising to
  265 late for a self-cosine of at least 0.5, and warns those are themselves
  ≥8x underestimates because the draws are clustered. Target row counts of
  120 early and ~2,120 late; at 256 rows per logical batch and 6 groups that is
  roughly 3 steps of accumulation early and 50 late.
- **λ̂ = −∇L_ctrl(θ_s)**, the gradient on the **controller** probe partition —
  never the monitoring partition, never the sealed one.
- Inner products via the existing CountSketch. The sketch is unbiased (median
  error +0.2% against an exact identity) but has a **measured cosine floor of
  0.0086–0.016**, established by I0008 on disjoint-support role blocks where
  the true inner product is exactly zero. Below that floor a difference between
  two groups is not resolvable, so the controller carries an explicit
  **deadband**: if `max_k ŝ_k − min_k ŝ_k` falls inside the floor, `q` reverts
  to `p` for that decision, and the reversion is recorded. Acting inside the
  floor would be acting on sketch noise.
- **ŝ standardized** by a running scale so β is dimensionless, then clipped:
  `q_k ∈ [p_k/2, 2p_k]`, renormalized. The clip bounds the treatment, keeps it
  a plausible recipe, and — critically — preserves **positivity**: every group
  keeps support at every decision, without which no off-policy quantity is
  defined for the groups a policy would otherwise zero out.
- β precommitted from the shakedown against a stated target on the realized
  deviation, not tuned against any outcome. See the separation section.

Rationale for the null being informative *given* the controls below: A1 is the
faithful cheap instantiation of the proposal. If the channel is demonstrably
live and A1 does not move the outcome, that is a result about the myopic
first-order value model — which is exactly what I0008 predicts and what a
future quadratic variant (A2, below) would have to beat.

### P1± — the saturation arm (channel test)

The arm that makes a null interpretable. P1 is not a controller and reads
nothing: it is a precommitted open-loop schedule that produces the **largest
time-variation in composition the recipe tolerates at a matched marginal.**

The taxonomy is split into two halves. P1+ trains the first half of the run at
the deviation ceiling on one side and the second half at the ceiling on the
other; P1− is the exact reversal. Both are constructed so the **run-level token
share of every group equals `p_k` exactly**. The deviation ceiling is wider
than A1's — provisionally `q_k ∈ [p_k/4, 4p_k]` — and is fixed in the shakedown
as the largest deviation that keeps every group above the positivity floor,
leaves total trained tokens unchanged, and does not destabilize a 200-step run.

The precommitted contrast is **P1+ − P1−**: two arms with identical marginals,
identical taxonomy and identical token budget, differing only in the *time
order* of composition, at maximum dose. Its magnitude is the direct measurement
of how much the schedule of composition can carry. If it sits at the noise
floor, the channel is closed at this depth, horizon and taxonomy, and **no
result from A1 is interpretable** — which the decision rule enforces.

Note the criterion is two-sided on magnitude. An extreme schedule that *hurts*
is a live channel and a valid positive control. Requiring it to help would
confuse "composition timing matters" with "our guess about the direction was
right".

### P2 — the hindsight oracle (deliberately strong positive control)

Greedy branch-and-select on the **monitoring** partition. At each of four
preregistered boundaries the run forks into G continuations, one per group
emphasized at the A1 clip ceiling, each run L = 100 steps; the branch with the
lowest monitoring-partition loss is selected and the run continues from it.

P2 uses realized future loss where A1 uses a first-order surrogate of it, at
the same cadence granularity. It is therefore strictly stronger than any online
member of A1's family, and it is the arm that should move the outcome if
anything in this program can.

Two honest limits. First, greedy-with-hindsight over a 100-step horizon is
**not** the optimal-control solution and can be worse than static — greedy
myopia is a real failure mode, so P2 is a positive control, not a ceiling.
Second, P2 selects on the monitoring partition, so its monitoring-partition
loss is its own objective and reporting it as evidence would be circular; only
its sealed-partition outcome counts.

Extra cost: `(G−1) × L` steps per boundary. At G = 6, L = 100, four boundaries
that is 2,000 extra steps against a 2,520-step run — about **1.79x** a normal
run.

### A2 — the quadratic variant, named and scoped out

I0008's R² result says the correct next controller adds the curvature term.
That requires group-restricted HVPs (I0008 Tier 2, +3% wall clock), which
telemetry v4 defers to v5. A2 is named here so that E07's result is read as
being about the first-order model specifically, and so the instrument roadmap
knows what the follow-on needs. It is not part of this design.

## Design

Two stages. Stage 2's treatments are *constructed from stage 1's output*, so
stage 2 cannot be fully frozen in advance — only the rule that builds its
treatments can be.

### Stage 1

| arm | policy | role |
|---|---|---|
| S0 | static nominal `p` | bridge to E01; replication check on the new loader mode |
| S* | static best mixture from E01 | **the comparator** |
| A1 | `q_k ∝ p_k exp(β ŝ_k)`, myopic λ̂, deadbanded, clipped | the hypothesis under test |
| P1+ | extreme open-loop schedule, marginal-matched | channel test |
| P1− | exact time-reversal of P1+ | channel test, paired |
| P2 | greedy hindsight oracle on the monitoring partition | strong positive control |

If E01 finds no composition effect, S* does not exist and collapses onto S0 —
in which case, per the freeze conditions, this campaign is not run in this
form at all.

### Stage 2, conditional and derived

Run **only if** the primary contrast or P1+ − P1− exceeds the detectable effect
size. Both arms are derived per block from that block's A1 run:

| arm | construction | what it separates |
|---|---|---|
| R1 | replay A1's realized `q` trajectory with the decision windows **permuted in time** | does the *particular timing* carry value, beyond having that set of mixtures in some order? |
| M1 | static mixture equal to the **time-average realized token composition** of A1 | does anything carry value beyond the run-level average mixture? |

R1 is the run-level analogue of the circular-shift null I0008 used, and it is
the sharpest control available: identical decisions, identical marginal, only
the order changes. M1 is the arm that catches an adaptive controller that
"wins" by drifting to a better static mixture.

### Blocking and run count

`init_seed` × `data_seed`, matching E01's structure so E01's variance estimates
transfer directly: provisionally 2 initialization seeds × 3 data-order seeds =
6 blocks, all arms within each block, run order randomized within block.

**The block count is not settable yet.** It is a function of the noise floor
E01 measures; see Power. At 6 blocks the campaign is 36 stage-1 runs
(≈40.7 run-equivalents, since P2 costs 1.79x) plus 12 stage-2 runs, so about
**53 run-equivalents, $125 and 35 H100-hours** at d12's measured $2.30 and
40 minutes. It scales linearly with blocks: 12 blocks is ~$245, 24 blocks
~$490. If E01's floor forces 24 blocks, this campaign is not affordable in this
form and the design must change rather than be run underpowered.

### Held fixed across all arms

Depth 12, architecture, width, optimizer recipe and every schedule, tokenizer,
dataset snapshot, sequence length 2,048, device batch 32, 8 accumulation
microbatches, 2,520 updates, 524,288 tokens per logical batch, group taxonomy
and version, probe partition manifest, evaluation batches, hardware class,
software revision, telemetry cadence, and the candidate-stream seed within each
block.

**Packing is held fixed at group-pure microbatches for every arm**, including
the static ones. Group purity is what makes `g_k` a free by-product, and a
group-pure microbatch implies group-pure rows within it — so E07 lives entirely
inside E01's packing arm B/D regime. S0 and S* must therefore be run under
group-pure packing too, or packing becomes a co-treatment. This is why E07
needs E01's packing main effect and interaction, not just its composition
effect.

**The mixture is defined by trained tokens, not by documents.** Per-group
token yield can differ under best-fit packing (E01 open question 3), so a
document-share target and a token-share target are different treatments. The
controller targets token shares; the sidecar records both.

## The control lag, and what it does to the cadence

This is a physical constraint of the loader, and it bounds how fast any
controller here can act.

**Prefetch.** `nanochat/dataloader.py` is a generator yielding one microbatch
per `next()`. `scripts/base_train.py:339` pulls microbatch 0 before the loop
starts; line 547 pulls the next microbatch inside the accumulation loop, after
each `backward()`. The loader is therefore always exactly one microbatch ahead.
When θ_{s+1} exists — after `optimizer.step()` at the end of step *s* —
microbatch 0 of step *s+1* has **already been built**. An observation taken at
step *s* cannot reach microbatch 0 of *s+1*; the earliest microbatch it can
influence is microbatch 1 of *s+1*, and the earliest **fully** influenced step
is *s+2*.

**Buffer turnover.** The packer draws from a rolling document buffer of
`buffer_size = 1000`. Measured over all 2,520 steps of d12-s7,
`batch/bos_count` averages **993.5** (sd 26.6, range 776–1,129): a d12 logical
batch consumes almost exactly one bufferful of documents. So a change to the
refill stream reaches full effect only after roughly one further logical batch
— and not uniformly, because best-fit is **length-biased**: it repeatedly takes
the largest document that fits and pops the shortest only on the crop path, so
short documents linger. If group correlates with document length, the realized
mixture is a group-dependent, length-biased low-pass filter of the intended
mixture, with a tail rather than a fixed delay.

**Quantization.** With 8 accumulation microbatches per step and group-pure
microbatches, the per-step mixture lives on a grid of eighths. With 4–8 groups
the nominal proportions are not even representable in a single step. The
mixture must therefore be realized as a **slot schedule over a decision
window**, not per step: a window of K steps gives 8K microbatch slots and a
resolution of 1/(8K).

**Consequences, all of which are design commitments.**

1. **Cadence.** Hard lag ≥2 steps, plus ~1 step of buffer turnover, plus a
   length-biased tail — a per-step controller is meaningless here. Cadence is
   precommitted at **K = 64 steps**, giving about **39 decisions per run** and
   512 microbatch slots per window (resolution 1/512, so quantization is
   negligible). K = 64 is not chosen for convenience: it is the smallest round
   window that also satisfies the I0008 sizing rule late in training, which
   needs ~50 steps of accumulation per group for a resolvable `ĝ_k`. The lag
   constraint and the estimator sizing constraint independently land on tens of
   steps, and 64 satisfies both.
2. **Thirty-nine decisions is the actual control authority.** This is a coarse
   phase schedule, not a fast controller, and every claim must be scoped to
   that.
3. **Three step numbers per decision, recorded, never assumed.** Telemetry v4
   item 2 already provides observation / decision / applied steps in the
   `batch_lineage` sidecar. E07 additionally requires that the applied field be
   an `(step, microbatch)` pair — because application begins mid-step — and
   that the verifier check `observation_step ≤ decision_step ≤ applied_step`
   rather than only the existing "decision precedes applied".
4. **Analysis uses realized composition, never intended `q`.** Because the lag
   is distributional, "applied at *s+2*" is not exact. v4 item 2 records
   realized composition by documents and by trained tokens; those are the
   analysis inputs, and the intended-versus-realized gap is itself a reported
   diagnostic.
5. **The non-perturbation gate applies.** A controller changes host-side loader
   work, so v4's mandatory bitwise A/B gate with step-time and loader-stall
   measurement covers every new loader mode here. The λ̂ probe gradient is the
   controller's own overhead; I0008 costed the val-probe gradient sketch at
   +0.08% over 25 periodic events, and 39 decisions is the same order. The
   overhead budget must be declared and verified, not assumed.

## Controller/evaluator separation

Telemetry v4 item 4 defines three immutable partitions — controller, monitoring
and sealed — bound by a manifest that fixes each partition's *allowed purpose*,
and guarantees the training process never receives the sealed index. E07 is the
first design that actually needs all three, and it adds requirements.

- **The controller reads only the controller partition.** λ̂ is a gradient on
  the controller partition. Nothing in the training process reads monitoring or
  sealed data.
- **The controller partition is not held out.** Because λ̂ steers the data, the
  controller partition is effectively being optimized against. Its loss is a
  diagnostic and must never be quoted as a held-out result.
- **The monitoring partition is for the shakedown, diagnostics, and P2's
  hindsight selection.** P2's monitoring loss is its own objective; reporting
  it as evidence is circular and is precommitted out.
- **Objective leakage through hyperparameter tuning is the real hazard.** The
  sealed partition survives a training process that never reads it and dies to
  an analyst who tunes β against it. Every policy hyperparameter — β, the clip
  bounds, the deadband threshold, cadence K, the EMA schedule, the P1 deviation
  ceiling, P2's L and boundaries — is frozen in the shakedown against the
  monitoring partition **before the first scored run**, and hashed into a
  `policy_version` bound in provenance. Any change to any of them makes a new
  design with a new id, per the standing rule.
- **Partitions must be stratified by group.** If controller, monitoring and
  sealed do not carry the same group proportions, a mixture shift changes the
  train-to-eval distributional relationship differently in each, and the
  measured effect is partly a partition artifact. Require stratified
  partitioning with recorded per-partition group composition.
- **Verification.** Extend v4 item 6's verifier with a structural check that no
  controller decision event references a sealed-partition probe id, and extend
  the tamper suite with an injected sealed read and with a decision whose
  applied step precedes its observation step.

## Propensities: recording the chosen mixture is not enough

The chosen mixture answers "what did we do". It does not support any
counterfactual, which is the entire point of recording a policy. What is
needed:

1. **The full `q` vector over all groups at every decision**, including groups
   with zero realized draws — not just the realized shares.
2. **Everything needed to recompute `q` from recorded state**: `ŝ_k` for all
   *k*, the identity of λ̂ (controller partition id, checkpoint, sketch seed),
   per-group EMA state and effective row count, deadband and clip flags, β, and
   the `policy_version` hash. Without these, `q` is a black box and no
   "what would policy π′ have chosen" question can be asked.
3. **A per-item inclusion propensity, factored so it is actually computable.**
   The design deliberately splits the sampler:
   - the **group slot allocation** over the decision window is *deterministic*
     (largest-remainder from `q`, recorded), which makes the treatment crisp
     and makes R1's time-permuted replay exact;
   - the **within-group document draw** is *randomized*, with a recorded
     `policy_seed`, RNG stream position, and per-document draw probability.

   Then `P(document d drawn) = (slot share of its group) × (within-group draw
   probability)` — a known, design-based quantity with strictly positive
   support everywhere, thanks to the clip. This factorization matters because
   the alternative route to per-item propensities is exact loader replay, and
   v4 defers that to v5.
4. **The candidate set, not only the chosen items.** The buffer's group
   composition and the refill stream position at each decision, so the
   denominator of any propensity is known.
5. **`policy_seed` becomes mandatory.** v4 item 1 makes it conditional on a
   stochastic sampler. E07's sampler is stochastic, so it is required, and it
   is a third seed axis in the design matrix.
6. **Draw-level and trained-token-level are different estimands.** The draw
   propensity above is clean. What a document *contributes* is length-biased by
   the best-fit packer, so the trained-token-level propensity is not recoverable
   without exact packer replay. Record realized trained tokens per document and
   state that off-policy work is valid at the draw level only.
7. **Realized composition by documents and by trained tokens both**, since
   their divergence is a per-group token-yield effect acting as an unmeasured
   co-treatment.

**And an honest limit on what all this buys.** Off-policy evaluation of a whole
training run is a sequential problem with 39 decisions and a horizon-long
outcome; importance weights compound multiplicatively across those decisions
and their variance will be useless. The propensity record is not a route to
"evaluate any policy offline". It is a route to exact reconstruction of what
happened, to short-horizon and single-decision off-policy checks, and to
building R1. Claiming more would be dishonest about the statistics.

## Primary outcome, precommitted

Final validation bits-per-byte on the **sealed** probe partition, using the
same estimator as E01 so the two campaigns share a scale, plus a
validation-trajectory summary as a secondary descriptive channel. The sealed
partition is never read by any training process in any arm.

**Primary estimand:** the mean within-block paired difference **A1 − S\***
(lower is better). One primary, uncorrected.

**Secondary contrasts, precommitted as a family** with Holm correction:
P1+ − P1−; P2 − S\*; A1 − M1; A1 − R1. S0 − (E01's arm A) is a replication
diagnostic, not a hypothesis test.

For scale: d12 final validation bpb is about **0.8478**, and the five
initialization-only d12 seeds span 0.84699–0.84782 (sd 0.00032 bpb, 0.038%
relative) — consistent with I0001's 0.06% loss floor, and a reminder that those
five runs share one data ordering. A depth step from d12 to d14 moves bpb by
−0.036 (−4.3%), which is the scale a genuinely large intervention produces.

## Treatment check, not an outcome

Given the deadband, a controller can run an entire campaign without treating
anything. Before any contrast is interpreted, precommitted treatment checks:

- **Realized deviation.** The time-averaged total variation distance between
  A1's realized token mixture and `p`, and its maximum over decisions, must
  exceed a threshold fixed in the shakedown. If A1's `q` sat inside the deadband
  all run, A1 is not a treatment and its null means only that `ŝ_k` differences
  were unresolvable at the sketch floor — which is a finding about the
  instrument, reported as such.
- **Deadband and clip rates.** Fraction of decisions reverted to `p`, fraction
  of group-decisions at a clip bound. A controller pinned at the clip is a
  different treatment from one operating in the interior.
- **Marginal matching.** P1+, P1−, M1 and R1 must match their intended
  marginals within a stated tolerance on realized trained tokens. If the packer
  breaks marginal matching, those controls do not do their job.
- **Lag realization.** Distribution of the gap between decision step and the
  step at which realized composition reaches its target, per group. This is the
  measurement that validates or refutes the lag model above.

## Decision rule, before the data

1. If **P1+ − P1−** and **P2 − S\*** both fall inside the detectable effect
   size, report: *the composition-timing channel is closed at d12, this horizon
   and this taxonomy.* **A1 is not interpreted at all**, and stage 2 is not run.
2. If either positive control moves the outcome and **A1 − S\*** does not:
   *time-varying composition matters, and the myopic first-order value model
   does not capture it.* This is an informative null about the policy family
   and the direct motivation for A2.
3. If **A1 − S\*** moves the outcome, stage 2 runs and separates the mechanism:
   **A1 − M1** tests whether anything is carried beyond the run-level average
   mixture; **A1 − R1** tests whether the *particular* schedule matters beyond
   the multiset of mixtures. An A1 gain that vanishes against M1 is a static
   finding wearing a controller's clothes.

## Power, honestly — and why it is blocked

**The power calculation cannot be completed and this design cannot be frozen
until E01 reports.** That is a statement of fact about the noise floor, not a
scheduling preference.

The known floor covers **initialization only**: I0001 gives 0.06% relative
standard deviation on loss, and the standing constraint in
[the README](README.md) is explicit that a design varying data order or
batching must establish its own floor. Every E07 arm varies the realized data
order by construction — that *is* the treatment — so the initialization floor
is the wrong reference.

The only anchor for the data-order floor is I0008's finding that batch
selection accounts for 99.1% of step-to-step training-loss variance with a
standard deviation of 0.0232 nats, **10.5x the I0001 seed floor**. That is a
*step-to-step training-loss* estimand. The outcome here is a *run-level final
held-out* loss, and the two are not the same quantity: step-to-step batch noise
partly averages out over 2,520 updates, while order effects can also
accumulate. A working prior of "roughly 10x initialization" is a guess. This
design must not be frozen on a guess.

**What the arithmetic looks like once the number exists.** With *n* paired
blocks, the normal approximation E01 uses gives a detectable difference of
`2.80 · σ_d / √n`, where σ_d is the standard deviation of the **within-block,
arm-to-arm paired difference** in final sealed bpb:

| blocks *n* | detectable difference |
|---:|---|
| 6 | 1.14 σ_d |
| 8 | 0.99 σ_d |
| 12 | 0.81 σ_d |
| 16 | 0.70 σ_d |
| 24 | 0.57 σ_d |

(The exact paired *t* calculation is less generous — 1.43 σ_d at *n* = 6. E01's
1.14 figure is the normal approximation; both should be reported.)

Pairing on `data_seed` shares the **candidate stream**, not the realized order,
because the treatment perturbs the realization. Residual within-block
correlation ρ between arms is therefore unknown, and for planning we assume
ρ = 0, giving `σ_d = √2 · σ_order`.

Three scenarios, against a d12 baseline of 0.8478 bpb:

| σ_d | detectable at *n* = 6 | at *n* = 24 |
|---|---|---|
| 0.0005 bpb (0.06%, data order adds nothing) | 0.0006 bpb (0.07%) | 0.0003 bpb (0.03%) |
| 0.0016 bpb (0.19%, ~3x) | 0.0018 bpb (0.22%) | 0.0009 bpb (0.11%) |
| 0.0050 bpb (0.6%, ~10x) | 0.0057 bpb (0.67%) | 0.0029 bpb (0.34%) |

**The third row is the one to worry about.** A detectable effect of 0.67%
relative is about one sixth of a full d12→d14 depth step. It is not obvious
that any plausible reweighting of coarse groups within one corpus, at a
1.3-billion-token budget, produces something that large. If the floor lands
there, the honest conclusion is that this experiment is not affordable at d12
in this form, and the response is to change the design — a larger effect
channel, a different outcome, or a different scale — not to run it
underpowered and report a null.

**The single number that unblocks this** is σ_d: the standard deviation of
E01's within-block arm-to-arm differences in final sealed bpb. E01 can report
it directly from its 6 blocks × 4 arms, and it plugs straight into the table
above to set the block count.

## What this does not answer

- **Whether the optimal-control theory is right.** No true costate is computed.
  A1 uses the myopic surrogate I0008 rates poor by construction. A null here is
  evidence about a first-order myopic controller, not about adjoint-based data
  selection.
- **Noncommutativity.** `λᵀ[v_i, v_j]` is unavailable: with one loss the
  bracket is identically zero, and `curvature/e_sym_*` must not be mistaken for
  a commutator. Group-restricted HVPs are v5.
- **Whether a quadratic value model would work.** That is A2, and it needs
  instrument capability that does not exist yet.
- **Whether the taxonomy is the right one.** E07 inherits E01's taxonomy. A
  null under a taxonomy that fails to separate materially different data is a
  fact about the taxonomy. This ambiguity is *irreducible within this design*
  and must be stated in any writeup.
- **Whether a faster controller would help.** Cadence is bounded below at tens
  of steps by prefetch, buffer turnover and estimator sizing simultaneously.
  Thirty-nine decisions is what the loader permits.
- **Whether the horizon is long enough for curriculum effects to exist.** 2,520
  updates and about 40 minutes may simply be too short for composition timing
  to matter. This is a live alternative explanation for a null and P1± is the
  only thing that speaks to it.
- **Anything cross-depth, cross-width or cross-architecture.** d12 only; and
  sketch-derived quantities are keyed by parameter schema and cannot cross
  architectures at all.
- **Downstream capability.** Bits-per-byte only. No task evaluations.
- **Off-policy value of arbitrary unrun policies.** Weights compound over 39
  decisions; see the propensity limits.
- **Compute-optimality.** The budget is fixed in trained tokens. The
  controller's wall-clock overhead is charged as overhead and reported, not
  traded against tokens.

## Instrument dependencies

**Telemetry v4, all of:** item 1 (separated `init_seed` / `data_seed`, **plus
`policy_seed`, which this design makes mandatory rather than conditional**);
item 2 (`batch_lineage`, with the applied field as an `(step, microbatch)`
pair, and the full policy fields); item 3 (`batch_id` joins); item 4 (all three
probe partitions, **stratified by group**, with per-partition composition
recorded); item 5 (direction certification fix); item 6 (verifier, extended per
the separation section); item 7 (group-pure microbatches, per-group sketches
and norms, the G×G Gram, exact `g_k · g_j` calibration at a handful of deep
steps, and the controller-partition probe gradient sketch for λ̂); item 8
(independent-draw noise, required because this design quotes noise scale — and
doubly so because group-pure microbatches make the accumulation draws *more*
clustered, further invalidating `b_noise`).

**Plus I0008 Tier 1 in full**, which is what makes `ĝ_k` free (+3.4% wall
clock, under 55 MB per run).

**Plus two items v4 defers to v5, which must be promoted or explicitly scoped
out:**

- **Per-item inclusion propensities.** Partly discharged by the factored
  sampler above — that is why the sampler is designed the way it is — but the
  recorded per-document draw probability is a new sidecar field.
- **Exact loader and controller replay.** Needed for R1 to be an exact replay
  rather than an approximate one, and the only route to trained-token-level
  propensities.

**Not required:** Tier 2 group-restricted HVPs, unless A2 is added.

**Runner and manifest:** the runner automatically accepts flags declared by a
pinned experiment branch, but the schema-v3 trainer does not yet declare or
implement a controller, policy version, third seed axis, or partition
manifests. A v4 experiment branch and immutable manifest must bind
`policy_version`, the partition-manifest hashes, the taxonomy version, β and
all frozen policy hyperparameters, and must verify them post hoc.

## Open questions before freezing

1. Is `q_k ∝ p_k exp(β ŝ_k)` with a myopic λ̂ worth running at all, given that
   I0008 predicts it fails? The alternative is to run only P1± and P2 as a
   channel-and-ceiling screen at roughly a third of the cost, and to defer any
   controller until A2's instrument exists. This is the most consequential open
   question in the document.
2. Should the deadband be set at the sketch floor (0.0086–0.016) or at a
   multiple of it? At the floor the controller acts on marginally resolvable
   differences; at 3x it may never act at all.
3. Is K = 64 the right cadence, or should the window follow normalized progress
   rather than absolute steps? I0006 shows the alignment axis can change
   conclusions and sometimes their sign, and the EMA sizing requirement grows
   over training, which argues for progress-aligned windows.
4. Can P1's deviation ceiling be made wide enough to matter while remaining a
   plausible training recipe? If `[p/4, 4p]` destabilizes training, the channel
   test is weakened exactly where it needs to be strong.
5. Does a 200-step shakedown confirm the lag model — specifically, that the
   realized token mixture reaches its target within one decision window for
   every group, and that group and document length are not so correlated that
   the length-biased packer breaks marginal matching?
6. Should stage 2 be conditional as drafted, or should R1 be run unconditionally
   on the grounds that a *paired* A1−R1 contrast is more informative than
   A1−S\* even when both are near the floor?
7. Is d12 the right scale, or does the horizon question (above) mean this
   experiment only becomes meaningful at a longer run, where the campaign cost
   changes by an order of magnitude?

## What E01 must deliver before this can be frozen

Seven specific results. Without them this document stays a proposal.

1. **A composition main effect on final sealed bpb, with an interval —
   magnitude and sign.** If the interval contains zero and excludes anything
   larger than E01's own detectable effect, static composition does not move the
   outcome and E07 is not run in this form. At most, P1± survives as a
   standalone channel test.
2. **σ_d — the standard deviation of within-block, arm-to-arm paired
   differences in final sealed bpb.** This is *the* number. It plugs directly
   into the power table and sets the block count, and therefore decides whether
   this campaign is affordable at all.
3. **σ_order — the run-level data-order standard deviation**, E01's stated
   deliverable from its three order seeds per block. Needed to size anything
   that is not paired and to report the floor honestly.
4. **A versioned group taxonomy: 4–8 coarse groups, their nominal proportions
   `p_k`, and their stratified assignment across the three probe partitions.**
   E01 open question 1. Every symbol in E07 is undefined without it, and the
   taxonomy must be the *same version*, or the campaigns are not comparable.
5. **Per-group token yield under group-pure packing.** E01 open question 3. E07
   defines its mixture in trained tokens, so the document-to-token conversion
   per group must be measured, not assumed — otherwise a mixture shift is
   confounded with token waste.
6. **Realized-versus-intended mixture fidelity for a static shift** (E01 arms
   C and D): how faithfully a *static* intended mixture survives the best-fit
   packer and the 1,000-document buffer. If a static shift is not realized
   faithfully, an adaptive one certainly will not be, and the lag model here
   needs recalibration before anything is frozen.
7. **The packing × composition interaction.** E07 fixes packing at group-pure
   for every arm. If packing and composition interact strongly, the composition
   effect E07 inherits is regime-specific and S\* must be defined within the
   group-pure regime, not from the marginal.

Items 1 and 2 are the gate. Items 4 through 7 are what make the design
*constructible*. Until all seven exist, freezing this would be pretending to a
power calculation and a treatment definition that we do not have.
