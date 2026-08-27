# E03 — width versus Muon decoherence

Status: **draft, not frozen**. Depends on telemetry v4.

## Question

At fixed depth, does increasing model width lower the divergence between the
compiled bf16 Muon update and its eager reference decomposition?

[I0003](../investigations/0003-decoherence-vs-depth/conclusion.md)
found `muon/replay_update_relerr` lower by about 6.6% at width 896 and 11.2%
at width 1024, both relative to the d12 width of 768. Those runs changed depth
at the same time. The within-depth matrix-shape result points to width as the
more plausible mechanism, but it does not identify it.

The width explanation is supported if the 1024-versus-768 primary contrast is
negative and excludes zero, with 896 intermediate in the same direction. It
accounts for the old scale-ray result only if the new effect sizes are also
compatible with roughly −6.6% and −11.2%. If the confidence intervals exclude
effects of those magnitudes toward zero, width is not sufficient to explain
I0003. A smaller or non-monotone change is a partial result, not a binary
refutation.

## Why fixed depth

[I0006](../investigations/0006-warmup-confound/conclusion.md) flags
every `muon/*` family as unsafe for depth claims because absolute-step and
normalized-progress alignment can change the answer. All arms here have depth
12, 2,520 updates, the same token batch at each update, and identical LR,
momentum, weight-decay and telemetry checkpoint schedules. Absolute step and
normalized progress are therefore the same comparison. Width is no longer
confounded with training horizon or schedule position.

## Design

One factor at d12: realized model width.

| arm | `--aspect-ratio` | realized width | attention heads | total parameters | initialization seeds |
|---|---:|---:|---:|---:|---:|
| W768 | 64 | 768 | 6 | 286.3M | 6 |
| W896 | 74 | 896 | 7 | 350.5M | 6 |
| W1024 | 85 | 1024 | 8 | 419.4M | 6 |

These widths reproduce the three widths in I0003 without reproducing its
depth changes. A narrower arm would answer a different extrapolation question
and is not included in this first draft.

`--aspect-ratio` is the available width knob, but the treatment is the
**realized `model_config.n_embd`**, not the argument itself. `base_train.py`
rounds `depth * aspect_ratio` upward to a multiple of `head_dim`. With the
default `head_dim=128`, usable realized widths lie on a 128-channel grid; the
three arms therefore have 6, 7 and 8 heads. Holding head dimension fixed is
preferable to changing attention geometry to preserve the head count, but it
means this design identifies width as implemented by nanochat, including the
mechanical change in head count.

The automatic recipe derivations are disabled or compensated. In the current
`base_train.py`, a fixed-d12 run rebuilds its d12 reference at the requested
aspect ratio. The automatic total batch therefore remains the 524,288-token
d12 reference, its batch-based LR multiplier remains 1, and its Muon
weight-decay scaling cancels back to 0.28. The parameter-count-derived horizon
does change: it would be 2,520 / 3,318 / 4,224 updates at widths 768 / 896 /
1024. Accepting those horizons would recreate the schedule-alignment problem.

Each arm instead uses an explicit total batch of 524,288 tokens, 2,520
updates, device batch 32, sequence length 2,048, matrix LR 0.02, initial Muon
weight decay 0.28, 40-step LR warmup, 400-step momentum ramp and 0.65 warmdown
ratio. This holds training exposure at 1,321,205,760 tokens. It deliberately
changes the tokens-to-scaling-parameters ratio from about 12.0 at W768 to 9.1
at W896 and 7.2 at W1024; that is the cost of identifying width at matched
exposure rather than comparing three automatically scaled recipes.

There is a second LR derivation outside the batch-size scaling:
`GPT.setup_optimizer()` multiplies the embedding, value-embedding and
unembedding LRs by `(width / 768)^−1/2`. To hold the **actual optimizer-group
LRs** fixed, the manifest inputs are compensated by the inverse factor:

| width | `--embedding-lr` | `--unembedding-lr` | resulting embedding / unembedding LR |
|---:|---:|---:|---:|
| 768 | 0.300000 | 0.008000 | 0.300 / 0.008 |
| 896 | 0.324037 | 0.008641 | 0.300 / 0.008 |
| 1024 | 0.346410 | 0.009238 | 0.300 / 0.008 |

The value-embedding LR consequently remains 0.15; scalar-group and smear LRs
also remain at their d12 values. Provenance and the verifier must check the
resulting optimizer-group LRs, not just the CLI inputs.

Blocking: six previously unused initialization seeds, provisionally 101–106,
crossed with all three widths. The one `data_seed` is fixed to the v4 setting
that reproduces the legacy deterministic order. Run order is randomized
within seed blocks. **18 runs, about $51 and 15 H100-hours; budget $55.** The
estimate scales the known 768-wide cost of $2.30 and 40 minutes by the total
parameter counts above.

Held fixed: depth, head dimension, sequence length, window pattern, vocabulary,
dataset snapshot, tokenizer, data order, logical and device batches, optimizer
hyperparameters after all derivations, token budget, every schedule, hardware
class, software revision, telemetry cadence and checkpoint steps. The only
architectural changes are realized width and its necessary head-count and
matrix-shape consequences.

## Primary outcome

Precommitted before the first run: a run-level Muon decoherence summary from
`muon/replay_update_relerr`. Select defined, sparse, post-update rows; exclude
the update-0 checkpoint with structural zeros; at each of the 20 common
uniform-tail checkpoints from 5% through 100% progress, take the median over
all Muon matrices; then take the geometric mean of those 20 checkpoint
medians. Analyze its logarithm.

The primary estimand is the mean log difference W1024−W768. W896−W768 and the
ordered W768 > W896 > W1024 pattern are precommitted secondary checks of dose
response. Per-role and per-shape results are descriptive mechanism checks, not
additional confirmatory outcomes.

Probe partition: **none**. This metric is produced by replaying the actual
optimizer update at sparse post-update events; it is not evaluated on the
controller, monitoring or sealed probe partitions. No loss, curvature or
sketch outcome substitutes for it.

## Power, honestly

[I0001](../investigations/0001-seed-variation/conclusion.md) gives
`muon/replay_update_relerr` a 3.5% initialization sd relative to the median.
That is the canonical standard-deviation figure, not the five-seed range. This
design changes neither data order nor batching, so initialization is the seed
axis it varies and that floor is the relevant available reference.

Against 3.5%, the I0003 effects are standardized differences of 1.9 for 6.6%
and 3.2 for 11.2%. An exact two-sample, two-sided t calculation at α=0.05 and
80% power needs six seeds per width for the smaller effect and three for the
larger one. With six per width the corresponding powers are about 84% and
99.9%. The calculation conservatively gives no credit for seed blocking.

The 3.5% floor was measured only at width 768, on five seeds and one fixed
data order. It is not evidence that the variance is homoscedastic across
widths. The six runs per arm measure that assumption, but the design must not
silently add seeds after seeing them; failure of the variance assumption would
make this campaign underpowered and motivate a new design.

## What this does not answer

- Whether depth affects decoherence. Depth never changes.
- Whether width alone can change while head count and matrix shapes do not.
  Fixed 128-dimensional heads make those consequences part of the treatment.
- Whether the effect persists on each width's automatically scaled,
  compute-optimal horizon or LR recipe. This design instead matches exposure
  and optimizer hyperparameters to identify a fixed-schedule width effect.
- Why compiled Muon decoheres, or whether lower decoherence is better. No
  causal link to loss, capability or optimizer quality is tested.
- Whether the effect generalizes to another data order, GPU/backend, model
  family or width outside 768–1024.
- Cross-architecture sketches or Hessian stability claims. Parameter schemas
  differ, and telemetry v4 explicitly forbids comparing their sketches;
  `eta * lambda_max` is not a valid Muon stability statistic.

## Instrument dependencies

Telemetry v4 item 1 (separate initialization and data seeds) and item 7
(`shape` populated), while retaining the existing sparse
`muon/replay_update_relerr` measurement, parameter roles and common deep-step
schedule. The primary scalar can be compared across parameter schemas; raw
sketches cannot. The new loader/manifest mode must pass v4's non-perturbation
gate and bind the realized model config, derived training quantities,
optimizer-group LRs and exact checkpoint grid into provenance.

The current runner can carry the existing `aspect_ratio`, schedule, and LR
flags in a flat manifest row. It reads their types from the pinned trainer,
asserts the recorded `user_config`, and the verifier recomputes width and head
count from `aspect_ratio`, depth, and `head_dim`. Before freezing, create a new
immutable manifest and decide whether the primary contrast also requires
explicit realized optimizer-group LR provenance. The old sweep manifests
remain unchanged.

## Open questions before freezing

1. Is matching the actual AdamW group LRs the intended width-only estimand, as
   drafted here, or should nanochat's `1/sqrt(width)` AdamW transfer rule be
   included as part of the width treatment? These are different experiments.
2. Should the primary trajectory summary reproduce I0003's full 5%–100% tail,
   or use only checkpoints after the step-400 momentum ramp? Both are cleanly
   aligned here, but they ask about different phases.
3. Will telemetry v4 provide an explicit `data_seed` value that reproduces the
   legacy ordering? If not, the applicability of I0001's 3.5% floor must be
   justified again before the power calculation can be frozen.
4. Does a short, non-scientific shakedown confirm that W1024 fits at device
   batch 32, that all three arms produce identical batch ids and checkpoint
   steps, and that the verifier records the compensated optimizer LRs? A
   device-batch reduction would change batch construction and therefore the
   design.
5. Is the direct 768/896/1024 replication sufficient, or is a 640-wide arm
   worth the additional six runs to test whether the trend extends below the
   characterized baseline?
