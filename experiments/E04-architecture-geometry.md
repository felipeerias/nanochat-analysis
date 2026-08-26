# E04 — architecture geometry at fixed parameter shape

Status: **draft, not frozen**. Depends on telemetry v4.

## Question

At fixed depth and fixed parameter shape, does a geometric signal measured
early in training rank architectural variants the same way held-out loss ranks
them at the end?

Two results answer it either way. If the arms separate on final sealed-partition
bits-per-byte and the early geometry ranks them in the same order across seeds,
there is a candidate predictor and the next design tests it out of sample. If
the arms separate on loss but the geometry does not track the order, the
geometric families in telemetry v4 item 7 are descriptive of training and not
predictive of outcome — which is worth knowing before more instrument budget
goes into them. If the arms do not separate on loss at all, the design is a
null on the intervention and says nothing about geometry; that failure mode is
why the arm ladder is built around a knob expected to move loss.

## Why shape-preserving interventions come first

A prior design debate settled this and the reasoning is worth restating,
because it is what makes E04 different from E03.

Changing parameter shapes costs three things at once. Matched-initialization
pairing becomes impossible: a different tensor set consumes the RNG stream
differently, so two arms at the same seed do not share θ₀ even in the
parameters that happen to have the same shape. Gradient sketches become
incomparable, because sketches are keyed by parameter schema
([telemetry v4 §3](../telemetry-v4-plan.md)). And what is left is an unpaired
comparison against the curvature noise floor of 25–29%, which needs about 29
seeds per arm to detect a 20% effect — six times this design's budget
([README standing constraints](README.md),
[I0001](../investigations/0001-seed-variation/conclusion.md)).

A shape-preserving intervention gives all three back. That is the entire reason
to start here.

## Which nanochat knobs are shape-preserving

I read `nanochat/nanochat/gpt.py` and verified the shape claim by constructing
each variant on the meta device and hashing the sorted
`(name, shape)` schema of `named_parameters()`.

**Verified shape-preserving** — the parameter schema hash is identical
(`6b882d9b24eb`), 286,261,730 parameters, 110,100,936 matmul parameters, for
every window pattern tested:

| knob | where | why it preserves shape |
|---|---|---|
| `window_pattern` | `GPTConfig.window_pattern`, `_compute_window_sizes` (gpt.py:39, 287) | only builds a list of `(left, right)` tuples handed to FA3; no parameter reads it |
| logit softcap | hardcoded `softcap = 15` (gpt.py:511) | a forward-pass scalar on the logits |
| QK-norm multiplier | `q = q * 1.2`, `k = k * 1.2` (gpt.py:103–104) | a forward-pass scalar on already-normed q/k |
| VE gate range | `3 * torch.sigmoid(...)` (gpt.py:96) | a forward-pass scalar on the gate output |
| backout tap layer | `backout_layer = n_layer // 2` (gpt.py:497) | `backout_lambda` is shape `(1,)` at any tap |
| rotary base | `base=100000` (gpt.py:270) | `cos`/`sin` are non-persistent buffers, not parameters |
| scalar init schedules | `resid_lambdas`, `x0_lambdas` init (gpt.py:238, 241) | shapes fixed at `(n_layer,)`; only the values change |

**Not shape-preserving**, and therefore excluded: `n_embd`, `n_head`,
`n_kv_head`, `ve_gate_channels` (gpt.py:81, sets `ve_gate` to
`Linear(12, n_kv_head)`), the smear gate's 24 input channels (gpt.py:185),
`pad_vocab_size_to`, and depth.

**Shape-preserving but not schema-preserving**, and also excluded: moving
value-embedding placement by editing `has_ve` (gpt.py:53). Keeping the *count*
of VE layers fixed keeps the multiset of shapes fixed, but the parameter
*names* change (`value_embeds.1` becomes `value_embeds.0`), which breaks
per-parameter pairing and breaks sketch keying just as badly as a shape change
would. It is a tempting knob and it does not qualify.

### The two chosen, and why

**Factor 1: the attention window pattern.** It is the only knob in the table
that is structural rather than a constant — it changes *which layers can see
past 512 tokens*, which is a real architectural decision people make, and it is
exactly the kind of change that should redistribute where gradient mass lives.
It requires no code change: `--window-pattern` already exists
(`scripts/base_train.py:55`), it is bound into telemetry provenance
(`telemetry.py:168`), and because `pattern[layer_idx % len(pattern)]` uses the
string verbatim when its length equals `n_layer`, an explicit 12-character
string specifies per-layer placement with no modification whatsoever.

**Factor 2: the logit softcap constant.** Not a crossed factor — a single
**calibration arm**. It is the cleanest intervention available: one scalar, no
parameters, no compute change, and a mechanism that is understood (it bounds
the logits and therefore the output-layer gradient and the cross-entropy
curvature). Its job is to give the geometric signals a reference deflection. If
they cannot separate softcap 15 from softcap 7.5, they will not separate
anything subtler, and the predictive question is answered negatively for cheap.

Rejected from the shortlist: the QK-norm multiplier and the VE gate range are
equally clean but act on the same subsystem as factor 1, which would make the
calibration arm confounded with the treatment mechanism rather than independent
of it. The backout tap layer is genuinely interesting and is the strongest
candidate for E05.

## Design

One factor at d12 with six levels, plus one calibration arm. FLOPs per token
are from `GPT.estimate_flops()`, computed on the built model, not estimated by
hand.

| arm | `--window-pattern` | long layers | softcap | FLOPs/token | vs A2 |
|---|---|---|---|---:|---:|
| A1 | `SSSSSSSSSSSS` | {11} | 15 | 731.4M | −3.7% |
| A2 | `SSSL` (default) | {3, 7, 11} | 15 | 759.7M | — |
| A3 | `SL` | {1,3,5,7,9,11} | 15 | 802.2M | +5.6% |
| A4 | `L` | all 12 | 15 | 887.1M | +16.8% |
| B1 | `LLSSSSSSSSSS` | {0, 1, 11} | 15 | 759.7M | 0.0% |
| B2 | `SSSSSSSSSLLL` | {9, 10, 11} | 15 | 759.7M | 0.0% |
| C1 | `SSSL` | {3, 7, 11} | **7.5** | 759.7M | 0.0% |

A1–A4 are a **budget ladder** (1, 3, 6, 12 long layers). A2/B1/B2 are a
**placement contrast at fixed budget**: three long layers each, and — verified —
*exactly* the same FLOPs per token. That makes the placement contrast the
cleanest comparison in the design: same shapes, same θ₀, same data, same
compute, same parameter count, only the layer indices differ.

`_compute_window_sizes` forces the final layer to L, so layer 11 is long in
every arm and the placement contrast is really about the other two.

Blocking: **5 initialization seeds** crossed with all 7 arms, run order
randomized within seed blocks. One `data_seed`, fixed. **35 runs**, plus 2
replicate runs of the (A2, seed 1) cell — same arm, same seed, run again — to
measure the platform-nondeterminism floor that bounds every paired difference.
**37 runs, about $87 and 25 H100-hours; budget $95**, scaling the known d12 cost
of $2.30 and 40 minutes by the per-arm FLOPs ratios above.

Held fixed: depth 12, width 768, 6 heads, head dimension 128, `n_kv_head` 6,
sequence length 2048, vocabulary, tokenizer, dataset snapshot, data order,
logical batch 524,288 tokens, device batch 32, 2,520 updates (1,321,205,760
tokens), every optimizer hyperparameter and derivation, warmup 40, momentum
ramp 400, warmdown ratio 0.65, final LR fraction 0.05, value-embedding
placement, every initialization constant, probe bank identity, probe partition
indexes, telemetry cadence and anchor steps, hardware class, software revision.
Two strings and one float are the entire treatment space.

Anchors, preregistered on absolute step (at fixed depth, absolute step and
normalized progress are the same axis, so
[I0006](../investigations/0006-warmup-confound/conclusion.md) does not
bite): **40** (end of LR warmup), **400** (end of momentum ramp), **881** (last
step before warmdown onset at 882), **1400**, **2000**, **2519** (final).
"Early" means {400, 881} and is fixed before the first run.

## What pairing buys, and where it stops working

Verified, not assumed: initializing a 12-layer model at seed 42 under six
different window patterns produces a **bitwise identical** parameter hash, and a
different one at seed 43. `init_weights()` never reads `window_sizes`, and the
softcap is not a parameter, so all seven arms in a block start at exactly the
same θ₀.

What that buys:

1. **The treatment is the only difference at step 0.** No initialization
   component in the arm contrast at all — not reduced, absent.
2. **The data stream is identical too.** The loader contains no RNG; the batch
   stream is bitwise identical across seeds at every step
   ([I0001 correction](../investigations/0001-seed-variation/conclusion.md),
   from I0008). With `data_seed` fixed, arms within a block see the same tokens
   in the same order from the same start.
3. **Trajectory-level comparison is licensed.** Step 900 in arm A2 and step 900
   in arm A4 are the same point in the same recipe having consumed the same
   data. "Early geometry predicts late loss" is therefore a within-block
   statement, not a cross-run correlation.
4. **The Muon parameter grouping is identical.** `setup_optimizer` groups matrix
   parameters by shape for stacking; identical schemas mean identical groups, so
   there is no hidden regrouping confound in the orthogonalization.
5. **Variance reduction** on the paired difference, by however much the shared
   θ₀ correlates the two endpoints. That amount is unknown and this design
   measures it.

Where it stops working:

1. **Any shape- or name-changing knob.** This is the boundary of the whole
   design and the reason the excluded list above is excluded.
2. **Divergence.** ρ decays over training. Pairing is strongest at the early
   anchors and weakest at the final outcome, which is exactly the wrong way
   round for the primary estimand. The power table below therefore reports the
   ρ=0 column as the number to plan against.
3. **Platform nondeterminism.** Two bitwise-identical configurations do not
   produce bitwise-identical runs on GPU, and that floor has never been
   measured here. Without it, a null paired result cannot be distinguished from
   "pairing bought nothing". The 2 replicate runs exist for this and nothing
   else.
4. **Across blocks.** Pairing does nothing for treatment-by-initialization
   interaction, which 5 seeds cannot separate from noise.

## Primary outcome

Precommitted before the first run: **final validation bits-per-byte on the
sealed probe partition at step 2519**, one number per run. The sealed partition
is never read by any training process (telemetry v4 item 4).

The primary estimand is the mean paired difference **A4 − A1** — 12 long layers
against 1 — in that quantity.

Why loss and not the alternatives. "Productive" is defined by held-out loss;
everything else is a proxy for the thing actually being asked about. Loss is
also the only channel where 5 seeds gives a usable minimum detectable effect:
0.06% initialization standard deviation on `loss/train_mean` and 0.16% on
`probe/loss`. Curvature at 25–29% would need about 29 seeds per arm for a 20%
effect and is out of budget by a factor of six. `muon/replay_update_relerr` at
3.5% is usable and is retained as a **secondary** outcome and as one of the
geometric signals — but decoherence between a compiled update and its eager
reference is not a definition of productive, and E03 already owns that channel
as a primary.

Preregistered secondaries, in this order: the monotone budget trend
A1 > A2 > A3 > A4 (Page trend test over 4 levels × 5 seeds); the FLOP-matched
placement contrasts B1 − A2 and B2 − A2; the calibration contrast C1 − A2;
`muon/replay_update_relerr` at the six anchors; the four geometric signals.
Everything after the primary is reported with FDR control across this
preregistered list and labelled exploratory.

## The geometric signals

Four scalars, preregistered, one per family. All are computed **within** an arm
and compared across arms as scalars. Raw sketch vectors are not compared across
arms, per telemetry v4 §3 — see the open question below for the one place that
rule may be too strong here.

- **G1 — Gram off-diagonal structure.** Mean of the off-diagonal entries of the
  normalized row-gradient Gram over the probe rows.
- **G2 — effective rank.** Participation ratio of the Gram spectrum,
  (Σλ)²/Σλ². Preregistered in this form rather than the entropy form: it is
  bounded, takes no logarithm of near-zero eigenvalues, and is more robust to
  sketch noise in the off-diagonals.
- **G3 — negative-interference fraction.** Fraction of the 120 distinct row
  pairs (16 rows) with a negative gradient inner product.
- **G4 — role norm shares.** Gradient squared-norm share carried by
  `attn_q + attn_k + attn_v + attn_out` against `mlp_in + mlp_out`, from the
  existing `role_norms` machinery (`telemetry.py:510`), which is already keyed
  by (role, layer). The preregistered scalar is the global attention share; the
  per-layer profile is descriptive.

These compare across arms for a specific reason: the probe rows are the *same
rows* in every arm, so the Gram differs only because the model differs, and the
role groups are identically named and identically shaped.

**The probe bank must be at T = 2048, not T = 256.** This design imposes that
on telemetry v4 item 7, which currently specifies 64 rows at T=256. The short
window is 512 tokens; at T=256 no position ever reaches the window boundary, so
every arm's S layers behave exactly like its L layers and *the treatment is
inert on the probe*. A T=256 probe would measure what the weights became, which
is legitimate but is not the mechanism, and it would measure it with the
treatment switched off. 16 rows at T=2048 is 32,768 probe tokens against 16,384
for the current spec — affordable in tokens; the per-row backward cost is the
real question and is an open item.

Preregistered directional hypothesis, so the result can falsify it: with fewer
long layers, rows whose targets depend on distant context can no longer be
served by attention, so the model leans harder on the residual and embedding
path — G4 attention share falls, G3 negative-interference fraction rises, G2
effective rank rises. This is a mechanism guess, not a prior finding, and it is
recorded here only so that "the geometry moved" cannot be claimed after the
fact in whichever direction it moved.

**The predictive test**, preregistered: within each seed block, compute the
Spearman correlation across the 7 arms between each early-anchor geometric
signal and final sealed bpb. That gives 5 correlations per signal. Test their
mean against zero. The arm-mean Spearman over 7 points is the readable summary
and is reported alongside; it needs |ρ| ≥ 0.786 for nominal significance on its
own, which is a near-perfect monotone relationship, so it is a summary and not
the test.

## Power, honestly

Dispersion statistic: the **standard deviation**, per
[I0001](../investigations/0001-seed-variation/conclusion.md)'s gate.
The floor cited covers **initialization only**; this design varies neither data
order nor batching, so initialization is the seed axis it varies and that floor
is the right reference.

Paired t, α=0.05 two-sided, 80% power, 5 pairs. The multiplier on the paired
difference's standard deviation is (t₀.₉₇₅,₄ + t₀.₈₀,₄)/√5 = 1.66, and
sd_d = sd·√(2(1−ρ)) where ρ is the shared-θ₀ correlation between arms.

| channel | I0001 sd | MDE at ρ=0 | ρ=0.5 | ρ=0.9 |
|---|---:|---:|---:|---:|
| `loss/train_mean` | 0.06% | 0.14% | 0.10% | 0.045% |
| `probe/loss` | 0.16% | 0.38% | 0.27% | 0.12% |
| `muon/replay_update_relerr` | 3.5% | 8.2% | 5.8% | 2.6% |
| `curvature/gHg` | 29% | 68% | 48% | 22% |
| G1–G4 | **unknown** | — | — | — |

Plan against the ρ=0 column. So: **the design detects a 0.38% change in
held-out loss, a 8.2% change in Muon decoherence, and nothing useful in
curvature.** Unpaired, the same 5 runs per arm would need 2.02 sd, so pairing
never costs anything and may buy a lot; how much is a deliverable, not an input.

Is 0.38% the right size to be looking for? For the budget ladder — 1 long layer
against 12 — probably comfortably, though nobody has measured it here. For the
FLOP-matched placement contrast, honestly not: a placement effect could easily
sit under 0.38% and this design would return an uninformative null on the
comparison it most wants to make. If the shakedown or the first block suggests
that, the placement contrast needs its own design at 10–12 seeds, not more
seeds silently added to this one.

**G1–G4 have no established noise floor.** The row-gradient Gram is telemetry
v4 item 7 and did not exist when I0001 was measured, so every geometry claim in
E04 is exploratory by construction. The 5 baseline-arm seeds give the first
5-seed estimate of the initialization spread of each new geometry family, and
the analysis order is preregistered: estimate the floors from the A2 seeds
first, then test arm differences against them. That floor is a deliverable of
this design in the same way the data-order floor is a deliverable of E01.

## Arms carried at no extra cost

- **The initialization floor on new runs.** Five fresh seeds at the default
  configuration under telemetry v4 re-measure I0001's floor on runs it was not
  fitted to. Every finding in this project is a same-data reproduction; this is
  a cheap step toward confirmation.
- **Architecture-invariance of the warmdown lock (I0005).** Sharpening rises
  about 15x starting at progress 0.350. Warmdown is fixed here, so its *onset*
  is not tested — but whether the onset stays at step 882 across seven
  architectures is a new question and the anchors already bracket it.
- **The platform-nondeterminism floor**, from the 2 replicate runs.
- **Gauge drift**, from the sketch cosine between paired arms at each anchor —
  see the open questions.

Not carried: anything data-side (data order is fixed by design), anything
cross-depth (I0006), and the width question (E03 owns it).

## What this does not answer

- **Anything about shape-changing architecture.** Width, head count, GQA ratio,
  gate channel counts, depth. Excluding them is the design's premise, and no
  result here transfers to them.
- **Whether geometry causes the outcome.** The predictive test is a rank
  correlation over 7 arms. Even a clean result is a correlation with n=7.
- **Whether any predictor generalizes.** One structural knob and one scalar knob
  is not a knob space. A perfect correlation within the window ladder could be a
  fact about attention budget rather than a general geometric predictor, and
  this design cannot tell those apart.
- **Whether long attention is worth its compute.** The primary contrast is
  **token-matched, not FLOP-matched**: A4 costs 21.3% more FLOPs per token than
  A1. Any advantage it shows is partly bought. A FLOP-matched version would need
  different token budgets per arm, which destroys the identical-data property
  that pairing depends on — a different design, not a variant of this one.
  The A2/B1/B2 placement contrast is the only FLOP-matched comparison here.
- **Anything about window placement beyond three layouts**, or budget beyond
  four levels.
- **The interaction between window pattern and softcap.** C1 sits only at the
  default pattern; the interaction is unestimated by construction.
- **Anything unconditional on data order.** All 37 runs share one token stream.
  E01 will deliver the data-order floor; until then every effect size here is
  conditional on that ordering, and I0008 puts the batch-selection contribution
  to step-to-step loss variance at roughly 10x the initialization floor.
- **Cross-depth anything.** Fixed d12, deliberately.
- **Downstream capability.** Base pretraining loss only; no midtraining, no SFT,
  no CORE.

## Instrument dependencies

Telemetry v4 items **1** (separate `init_seed` and `data_seed`, and the derived
design matrix that makes "differs only in `window_pattern`" checkable rather
than asserted), **3** (`batch_id` on logical-batch loss and gradient events —
the design's central claim is that all arms saw the same stream, and this is
what verifies it), **4** (sealed probe partition — the primary outcome),
**5** (direction certification in the loader, so gradient-direction curvature is
filtered by the gradient verdict), **6** (sidecar verifier), **7** (geometry
recording: model-only checkpoints at the six anchors, the sampled row-gradient
Gram with per-row norms, spectrum, effective rank and negative-pair fraction,
and `shape` populated).

Item **2** is needed only in reduced form — the batch-identity fields sufficient
to verify stream equality across arms, not the full lineage sidecar, since no
data-side question is asked. Item **8** is not required: this design cites
I0001's published floor and does not size anything from `b_noise`.

Two amendments E04 asks of item 7: the probe bank at **T = 2048** rather than
T=256, for the reason given above; and the row-gradient Gram computed at the
same six anchors as the model-only checkpoints, so geometry and outcome share
an alignment.

Implementation prerequisites beyond v4:

- **None for factor 1.** `--window-pattern` exists, accepts explicit per-layer
  strings, and is already bound into provenance.
- **Factor 2 needs a config-exposed softcap.** It is hardcoded at gpt.py:511 and
  independently duplicated as `TelemetryConfig.softcap = 15.0`
  (`telemetry.py:80`), with `softcap_equiv_err` as a deliberate canary for
  disagreement between the two. One value must feed both, or the C1 arm fires
  its own canary at every checkpoint.
- **FA3 is mandatory.** `base_train.py:120` warns that SDPA has no sliding-window
  support. An SDPA fallback would not just be slow, it would make the treatment's
  realization depend on the kernel path, and only some arms use windows. The
  runner must assert `USE_FA3` and record it in provenance for every run.

## Open questions before freezing

1. **Does the standing rule against cross-arm sketch comparison hold when θ₀ is
   shared?** The rule exists because of permutation symmetry, but matched
   initialization pins the gauge: two arms that start at literally the same
   weights are not permutation-related, they are the same point. The rule is
   obeyed in this draft. The cheap way to settle it is to record the sketch
   cosine between paired arms at each anchor and watch it decay — that measures
   how long the shared gauge survives, and it is a deliverable either way. Should
   it be preregistered as a finding or left as a diagnostic?
2. **Is the placement contrast underpowered enough to be worth splitting out?**
   It is the cleanest comparison in the design — FLOP-matched, shape-matched,
   init-matched — and the most likely to return a null that means nothing. Ten
   seeds on A2/B1/B2 alone might be a better $70 than seven arms at five seeds.
3. **What does 16 row-gradients at T=2048 cost per anchor?** The token count is
   fine; the backward passes are the question, and if it is prohibitive the
   trade is fewer rows at full length against more rows at a length where the
   treatment is inert. Fewer rows also shrinks G3's denominator from 120 pairs.
4. **Does the baseline softcap ever bind?** The calibration arm assumes 7.5
   binds strictly more often than 15. If the existing `saturation_fraction`
   statistic shows 15 essentially never binding, C1 is measuring the onset of a
   previously inactive nonlinearity rather than a strengthening of an active
   one — still a valid calibration, but a different one, and the shakedown must
   check it before the arm is frozen.
5. **Is `L` really full context?** `_compute_window_sizes` maps L to
   `(sequence_len, 0) = (2048, 0)`, while the surrounding comments (gpt.py:166,
   291) describe full context as `(-1, 0)`. For T ≤ 2048 causal attention these
   are equivalent in principle, but they may take different FA3 kernel branches.
   The shakedown should confirm bitwise equality, or the A4 arm should be
   defined as "a 2048-token window" and described as such.
6. **Two documentation bugs to fix before anyone sizes anything from them.**
   gpt.py:302 says the short window is "2048 -> 768"; `base_train.py:55` says
   "S=half context". The code computes a quarter, ceiled to the FA3 tile:
   **512**. Every FLOP figure in this document uses the code, not the comments.
7. **Five seeds, or six?** E03 chose six for `muon/replay_update_relerr`. Five is
   chosen here because it reproduces I0001's population size exactly, which makes
   the baseline-arm spread directly comparable to the published floor. That is a
   real benefit but it is not obviously worth more than the power a sixth seed
   would add.
