---
investigation: I0007
analyst: A0001
design: confirmatory
outcome: supported
saw: >
  investigations/0007-zero-init-wakeup/README.md@e76859c (protocol);
  analysis/README.md@e76859c; telemetry-data/sweep/DATASET.md;
  investigations/0001-seed-variation/conclusion.md@4ac11f3;
  investigations/TEMPLATE-result.md@e76859c; analysis/loader/telemetry_load.py;
  nanochat/nanochat/telemetry.py and nanochat/nanochat/gpt.py on branch
  `telemetry` @5c2fb16 (the instrument and the initializer, read to establish
  what exact zero means mechanically). Did NOT read: the sibling A0002/
  folder, any conclusion.md in I0007, any other investigation's results, or
  analysis/profiles/.
data: >
  sweep; the seven schema-v3 segments
  d12-s7-s0-6c217854f4174ce884dfb2b2dcc13c45,
  d12-s8-s0-2b2e72e4395440029b92226213d137bb,
  d12-s9-s0-b872e698e81b4a8fa0cd08512f43c2c2,
  d12-s10-s0-0ffeb05742bb4154a3f1afc202dc3955,
  d12-s11-s0-5fd1bf807f764954b30658872fc3e2ad,
  d14-s7-s0-b72acf7ac59942dab902e26fa0b2c80d,
  d16-s7-s0-8e0a39fc8f164343bd01ef6b915e849f.
  The legacy v1 segment d12-iter was excluded per the task brief and
  DATASET.md ("do not pool it with the v3 runs").
selection: >
  PRIMARY: metric == "muon/replay_update_relerr", tier sparse, phase
  post_update (its only phase); update_index = step - 1 per the
  pre_update/post_update convention; is_defined applied explicitly (all
  18,044 rows are defined, zero rows dropped, zero honestly-undefined rows).
  No arm filter: this metric carries acceptance_arm = null by design
  (_DEEP_NONARM_SCHEMA). One series per parameter_name per run; every deep
  checkpoint of every run kept.
  CORROBORATION at update index 0 only (periodic tier, step 0, pre_update):
  grad/norm and grad/rms (keyed by param_role x layer), grad/zero_fraction
  and grad/max_abs (role-level aggregates), muon/data_norm,
  muon/u_final_norm_observed, muon/decay_norm, muon/cos_raw_final
  (undefined_reason).
universe: >
  585 per-matrix Muon decoherence series (78 x 5 d12 + 91 d14 + 104 d16) over
  18,044 (matrix, checkpoint) cells tested; all 585 reported, none sampled.
  Plus 8 corroborating channels at update index 0 in all seven runs, all
  reported. The secondary magnitude analysis in section 5 is EXPLORATORY and
  is labelled as such.
code: "coordinator commit pending:investigations/0007-zero-init-wakeup/A0001/wakeup.py"
seed_reference: >
  investigations/0001-seed-variation/conclusion.md@4ac11f3. The primary result
  is a set/partition claim whose five-seed spread is exactly zero, matching the
  behaviour I0001 records for configuration channels. Where a magnitude is
  compared across depths I used the d12 five-seed MIN-MAX RANGE, stated as
  such.
supersedes: none
---

## Result

**The wake-up order has exactly two tiers, and the split is the zero-init
boundary itself. There is no third tier, no layer gradient, and no seed or
depth variation in it.**

### 1. What "exactly zero" means here and how it was tested

`value_scalar` is stored as parquet `double`, so the loader performs no
conversion: the bytes tested are the bytes on disk. Zero was tested on the
IEEE-754 bit pattern (`0x0` or `0x8000000000000000`, i.e. both signed zeros),
which is equivalent to float64 `== 0.0` and admits no tolerance. No epsilon,
no `isclose`, no rounding.

The distinction is not marginal, it is categorical:

| quantity | value |
|---|---|
| exactly-zero decoherence cells, all 7 runs, all 30-33 checkpoints | **405** |
| all of them at update index | **0** (and nowhere else) |
| largest magnitude among the "zero" cells | **0.0** (it is zero) |
| smallest nonzero decoherence at update index 0 | **0.108 - 0.115** across runs |
| smallest nonzero decoherence *anywhere* in the 18,044 cells | **3.23e-3** |

At update index 0 the distribution is bimodal with an *empty* interval between
the modes: a point mass at exactly 0, and a cluster at 0.11-0.15. There is no
value in between in any run. No choice of tolerance could move a cell from one
state to the other. This is what makes the "asleep" state a state rather than
a small number.

The mechanism is legible in the instrument (`nanochat/telemetry.py`,
`muon_stages` / `_emit_muon_reference`): decoherence is
`||delta_reference - delta_applied|| / ||delta_applied||`. When the gradient
and the momentum buffer are both exactly zero, the Newton-Schulz input is zero,
`u_final` is zero, and the applied update reduces to cautious weight decay
`-lr * wd * p` -- a scalar multiply that the compiled kernel and the eager
reference compute bitwise identically. Hence exactly 0.0. When any gradient is
present, Newton-Schulz amplifies rounding placement and decoherence lands in
the 3-10% band that DATASET.md caveat 5 describes. **Exact zero is the
signature of "no gradient reached this matrix at all".**

### 2. The wake-up order

Every Muon matrix falls into one of two tiers, in all seven runs:

| tier | first checkpoint with nonzero decoherence | roles | d12 | d14 | d16 |
|---|---|---|---:|---:|---:|
| A | update index **0** | `attn_out` (= `attn.c_proj`), `mlp_out` (= `mlp.c_proj`) | 24 | 28 | 32 |
| B | update index **1** | `attn_q`, `attn_k`, `attn_v`, `ve_gate`, `mlp_in` | 54 | 63 | 72 |

Tier B is **69.23%** of the matrices at every depth -- and that is arithmetic,
not coincidence: each block contributes 6 Muon matrices plus a `ve_gate` on odd
blocks, of which 4 (+ the gate) are upstream of a zero-init projection, giving
9/13 = 0.6923 exactly for any even layer count.

Tier A is **exactly** the set of zero-initialized matrices. Tested by name
against the model's own initializer (`gpt.py init_weights()` calls `zeros_()`
on `block.attn.c_proj.weight` and `block.mlp.c_proj.weight` and on nothing else
in the Muon set): `tier A == zero-init set` is **True in all 7 runs**
(24/24, 24/24, 24/24, 24/24, 24/24, 28/28, 32/32).

So the order is: the zero-init projections themselves are the only matrices
that are gradient-active at the very first update; every matrix upstream of one
wakes at the very next update; nothing wakes later. **The set of wake-up
indices is {0, 1}. There is no matrix anywhere in the dataset whose wake-up
index is 2 or greater.**

Note the resolution: indices 0 and 1 are *consecutive updates*. The two-tier
structure is therefore not an artefact of the checkpoint grid -- there is no
finer grid to be had. (See Limitations for the one thing this still cannot
see: within-update, per-microbatch structure.)

### 3. Independent corroboration at update index 0

Five further channels agree on the same 54 / 63 / 72 matrices, in all seven
runs, with exact-zero (not small) values:

| channel | tier-B matrices flagged | agreement with tier B |
|---|---|---|
| `grad/norm` exactly 0 | 54/78, 63/91, 72/104 | **exact set match, 7/7 runs** |
| `grad/rms` exactly 0 | 54/78, 63/91, 72/104 | **exact set match, 7/7 runs** |
| `grad/zero_fraction` exactly 1.0 | all 5 tier-B roles | role-level, matches |
| `grad/max_abs` exactly 0 | all 5 tier-B roles | role-level, matches |
| `muon/data_norm` exactly 0 | 54/78, 63/91, 72/104 | **exact set match, 7/7 runs** |
| `muon/u_final_norm_observed` exactly 0 | 54/78, 63/91, 72/104 | **exact set match, 7/7 runs** |
| `muon/cos_raw_final` honestly undefined | 54/78, 63/91, 72/104, reason `degenerate_or_zero` | **exact set match, 7/7 runs** |

The complementary check also lands: `muon/decay_norm` is exactly 0 for exactly
the **24 / 28 / 32 tier-A** matrices -- because their parameters *are* zero at
init, so cautious weight decay has nothing to act on. Tier A and tier B are
each independently identified by a different channel's exact zeros.

For tier A, `grad/zero_fraction` at step 0 is 5e-6 to 1.5e-5, not zero: the
zero-init projections receive a dense, tiny gradient from the very first
backward, because `lm_head` is initialized `normal_(std=0.001)` rather than to
zero, so the residual stream carries gradient at every layer immediately.

### 4. Stability across the five d12 seeds

Quantified, not asserted:

- Matrices keyed identically in all five seeds: **78 of 78**.
- Matrices with the **same wake index in all five seeds: 78/78 = 100.0%**.
  Matrices that disagree anywhere: **none**.
- Pairwise agreement over all **10** seed pairs: min 100.00%, median 100.00%,
  max 100.00%.
- Tier sizes per seed: 24 / 54 in every one of the five.

Because the hard wake index takes only two values, any rank correlation on it
is degenerate (every pair within a tier is a tie) and would report 1.0 by
construction. The honest statement is the set-level one: **the partition is
bit-identical across seeds, so the five-seed spread of the wake-up ordering is
exactly zero.** Against `investigations/0001-seed-variation/conclusion.md@4ac11f3`
this puts wake-up ordering with the configuration channels (`optim/lr`,
`batch/*`), which are the only other quantities in this dataset with exactly
zero seed spread -- and far outside the 3.5% sd-relative / ~8% range-relative
noise floor that the same reference assigns to `muon/replay_update_relerr`'s
*magnitude*.

### 5. Secondary, EXPLORATORY: the graded structure is in the magnitude

Since the hard order is two-valued, I also asked whether the *magnitude* of
decoherence at the wake-up checkpoint orders the matrices reproducibly. This
was not a declared test; it is exploratory.

- Between-seed Spearman rho on all 78 d12 matrices at update index 1, 10 pairs:
  min 0.971, median **0.978**, max 0.985.
- Restricted to tier A at update index 0 (24 matrices): min 0.788, median
  0.870, max 0.948.
- Most of that is role separation -- the d12 role medians at update index 1
  span ~45x: `ve_gate` 0.0040, `attn_q` 0.0066, `attn_k` 0.0068, `mlp_in`
  0.0071, `attn_v` 0.0220, `mlp_out` 0.149, `attn_out` 0.177.
- After removing the per-role median within each seed, the residual per-matrix
  ordering still reproduces: rho min 0.637, median **0.703**, max 0.756. So
  there is a real, seed-stable per-matrix ordering underneath the role effect.
- That residual ordering tracks position in the stack: rho(residual, relative
  depth) = **-0.414 to -0.538** across the five d12 seeds. Deeper matrices
  decohere *less* at the wake-up checkpoint.

### 6. Across depths: governed by neither absolute step nor relative depth

**Normalization stated first.** Depth changes the layer count (12/14/16) and
hence the matrix count (78/91/104), so layer indices are not comparable across
depths; layers are reported as relative depth rho_L = layer/(n_layer-1) in
[0,1]. Steps are not comparable either (horizons are 2520/3759/5376), so
wake-up is reported both as an absolute update index and as normalized
progress.

| run | L | matrices | tier A | tier B | tier B % | wake indices | last wake as normalized progress |
|---|---:|---:|---:|---:|---:|---|---:|
| d12-s7 | 12 | 78 | 24 | 54 | 69.2% | {0, 1} | 3.97e-4 |
| d14-s7 | 14 | 91 | 28 | 63 | 69.2% | {0, 1} | 2.66e-4 |
| d16-s7 | 16 | 104 | 32 | 72 | 69.2% | {0, 1} | 1.86e-4 |

Within every run, the wake index is **constant inside each role** -- so
rho(wake, rho_L) is undefined because there is no variation to correlate. Every
`attn_q` at every relative depth wakes at 1; every `attn_out` at every relative
depth wakes at 0. Relative depth explains nothing.

Answering the protocol's question directly:

- **Not relative position in the network.** Layer 0 and layer L-1 behave
  identically, at all three depths.
- **Not relative progress.** The wake-up point is the same absolute update
  index (1) at all three depths, so in normalized progress it differs by 2.1x
  (3.97e-4 vs 1.86e-4).
- **A fixed absolute count of one update -- but that is a consequence, not a
  cause.** What actually governs wake-up is *graph distance to the nearest
  zero-initialized projection*, and that distance is 1 for every affected
  matrix at every depth: each block's own `c_proj` gates only its own block's
  upstream matrices, so there is no propagation front to travel down the stack.
  Deepening the network adds independent gates in parallel, not gates in
  series. Any recipe change that kept the same block-local zero-init would
  produce the same one-update wake-up at any depth.

The one graded quantity behaves the same way under normalization. The
within-role residual-vs-relative-depth rho is -0.472 at d14 and -0.515 at d16;
the d12 five-seed **range** is [-0.538, -0.414], and **both fall inside it**.
So the depth gradient of wake-up magnitude is present at every depth and is not
distinguishable across depths at n=5 seeds.

## Limitations

1. **The gradient-norm half of Test step 1 was not measured at full resolution.**
   `grad/*` lives in the periodic tier, emitted at pre_update steps 0,
   ceil(N/25), 2*ceil(N/25), ... -- that is steps 0 then 101 (d12), 151 (d14),
   216 (d16). There is no gradient row at update indices 1, 2, 4, 8, 16, 32, 40
   or 64. Directly measured, the gradient-norm wake-up is only bracketed to
   (0, 101]. I closed it to exactly 1 by **inference** from the instrument's own
   reference decomposition: at update 0 the momentum buffer is zero and the
   gradient is exactly zero, so `momentum_buffer.lerp_(g, 1-mu)` leaves it
   exactly zero; at update 1 the Newton-Schulz input is therefore proportional
   to g_1 alone, and a nonzero decoherence at update 1 requires a nonzero
   Newton-Schulz input, hence g_1 != 0. Every tier-B matrix has nonzero
   decoherence at update 1, so every tier-B gradient is nonzero at update 1.
   That argument is sound but it is a derivation from `muon_stages`, not a
   reading of a gradient channel. Flagging it as the one inferential step in
   the result.
2. **Deviation from the protocol's `selection` line.** The line names
   "muon/replay_update_relerr and the Muon stage families at the early deep
   checkpoints". The Muon stage families are periodic-tier and are *not*
   emitted at the early deep checkpoints (only at step 0 in that window), so
   the stage families could only be used at update index 0. I used them there,
   as corroboration, and additionally used `grad/*` (also periodic, also step 0
   only), which `selection` does not name but Test step 1 requires.
3. **Resolution floor.** Update indices 0 and 1 are consecutive, so no finer
   checkpoint could refine the tier-A/tier-B boundary. But `grad_accum_steps`
   is 8, so one update is 8 microbatches, and the instrument has no
   per-microbatch telemetry. If there is an ordering *inside* update 1 it is
   invisible here, and nothing in this dataset can see it.
4. **Interpretation risk on the primary channel.** Exact-zero decoherence
   strictly means "the applied update matched the eager reference bitwise". I
   read that as "no gradient content", justified by the mechanism above, by the
   fact that the nonzero mode never falls below 3.2e-3 anywhere in 18,044
   cells, and by five independent channels flagging the same set at update 0.
   At update 1 the reading is inferential (see 1). If a future recipe made the
   compiled and eager Muon paths agree bitwise on nonzero input, this channel
   would stop being a wake-up detector; nothing in *this* dataset does that.
5. **DATASET.md caveats.** Caveat 1 (size ray, not depth sweep: depth covaries
   with width, heads, batch size, LR, weight decay and horizon -- so "depth
   causes X" is never claimed here; the depth-invariance statement is about the
   nanochat recipe at scale). Caveat 2 (n=3 depths; the cross-depth statement
   is a three-point agreement, not a trend fit). Caveat 3 (absolute warmups)
   applies only weakly: wake-up happens at update 1, far inside the 40-step LR
   warmup at every depth, and the warmup enters as a multiplicative LR factor
   that cannot change a zero into a nonzero. Caveat 5 (Muon decoherence is a
   reference-frame quantity with a recorded error bar) is the basis of the
   magnitude discussion in section 5 and is why that section is exploratory.
   Caveat 8 (compiled training is not bit-reproducible; ~1 ulp from the
   embedding-backward atomic race) is unusually harmless here: the tier-B
   matrices are *structurally* zero, not numerically small, and 1 ulp of
   nondeterminism in an embedding gradient cannot create gradient where the
   chain rule multiplies by an exactly-zero weight matrix.
6. **The seed reference is d12 only.** The d14/d16 comparisons in section 6
   borrow the d12 five-seed range, which I0001's conclusion explicitly says may
   not transfer. This weakens only the section-5/6 magnitude statements; the
   primary partition result needs no error bar because its spread is zero.
7. **Section 5 is exploratory** in the sense of `analysis/README.md`: it was
   found by looking at the data under study, after the declared test returned a
   degenerate (two-valued) ordering. It should be re-tested on new runs before
   being relied on.
8. **Non-Muon parameters were not in the declared universe.** For context only:
   at step 0 `grad/norm` is also exactly zero for all 6 `value_embedding`
   tensors and for `smear_gate`, and nonzero for `token_embedding`,
   `unembedding`, `resid_scalar`, `x0_scalar`, `smear_scalar` and
   `backout_scalar`. Those are AdamW parameters with no decoherence channel and
   were not analysed.

## Files

- `wakeup.py` -- the analysis; regenerates everything below from the raw
  segments.
- `wakeup_tables.txt` -- full printed output, every run and every table.
- `fig_wakeup.png` -- wake-up index vs relative depth, one panel per depth.
- `fig_magnitude.png` -- exactly-zero fraction vs checkpoint (all seven runs
  coincide), and decoherence magnitude at the wake-up checkpoint vs relative
  depth.
