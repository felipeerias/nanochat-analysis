---
investigation: I0008
analyst: A0001
design: exploratory
outcome: refuted            # FEASIBILITY refuted: the dataset cannot support the
                            # proposal as posed. The proposal itself is neither
                            # supported nor refuted here - it is untestable on
                            # this data, for two independent structural reasons.
saw: investigations/0008-adaptive-batching-feasibility/README.md@e76859c;
  analysis/README.md@c97956e; telemetry-data/sweep/DATASET.md;
  investigations/0001-seed-variation/conclusion.md@4ac11f3;
  nanochat/nanochat/telemetry.py@273c8bd (sections 5-7, deep schema, emitters);
  nanochat/nanochat/dataloader.py@273c8bd (read to establish that the loader
  has no RNG); telemetry-spec.md; ../../loader/telemetry_load.py.
  Did NOT read A0002/, conclusion.md, or any other investigation's results.
data: sweep; all seven schema-v3 segments (d12-s7..s11, d14-s7, d16-s7);
  d12-iter excluded as instructed
selection: is_defined == True everywhere via loader defined(); per-arm selection
  via arm(); curvature restricted to acceptance_arm="shadow_fp32" for headline
  numbers with a separate certified subset using curvature/verdict_code_gradient
  == 0; loss analysis drops steps < 40 (LR warmup, caveat 3)
universe: 283 metric families present in the seven v3 segments were inventoried
  (see metric_inventory.csv) and every one was assigned a row in the mapping
  table, including the "no proxy" rows; 56 families were actually computed on
  (41 named + 15 overhead/total/*); 16 distinct empirical tests were run and
  all 16 are reported, including the 4 that were refuted
code: uncommitted (the coordinator commits):
  investigations/0008-adaptive-batching-feasibility/A0001/code/*.py
seed_reference: investigations/0001-seed-variation/conclusion.md@4ac11f3
  (sd-relative statistic). Independently re-derived here: the seed-idiosyncratic
  component of the detrended d12 training loss is 0.071% of the loss level,
  reproducing the reference's 0.06% for loss/train_mean.
supersedes: none
---

## Result

### Headline

The proposal cannot be tested on this dataset, and the reason is worse than the
one the protocol anticipated. There are **two** blocking facts, not one.

1. **No data-group labels.** As stated in the protocol and in `DATASET.md`
   caveat 8. Nothing in the 283 recorded metric families identifies a document,
   a domain, or a source. `g_k` therefore has no `k`.

2. **The dataset contains exactly one data ordering.** `nanochat/dataloader.py`
   contains no RNG: it walks parquet files in filename order, row groups in
   index order, and fills a deterministic best-fit packing buffer. `--seed`
   seeds `torch.manual_seed` only, i.e. initialization. I verified this in the
   data: `batch/bos_count` is **bitwise identical at every step across all
   seven runs and all three depths**, and so are
   `batch/mean_segment_length`, `batch/segments_per_row_{mean,max}` and
   `batch/valid_targets`. The five d12 seeds are five trajectories over the
   *same* batch sequence; d12's sequence is a prefix of d14's, which is a
   prefix of d16's (all three use `total_batch_size` = 524288 and
   `device_batch_size` = 32 with 8 accumulation steps; they differ only in how
   many steps they run).

Fact 2 is the deeper obstruction. Even with perfect group labels, `s_k(t)` is a
statement about a counterfactual — *what the loss would have been had the
mixture been different* — and this dataset has zero variation in the mixture.
No estimator, however clever, recovers a treatment effect from one realization
of the treatment. **Question 2 is therefore unanswerable in principle here, not
just unmeasured**, and any future instrumentation proposal that records more
telemetry but keeps one fixed data order will still be unable to answer it.

Against that, the empirical work below establishes four things that a future
experiment genuinely needs:

- **Data selection is a large effect on this recipe** — batch identity accounts
  for **99.1%** of the step-to-step fluctuation in training loss (ICC 0.991),
  a 0.023-nat sd that is **10.5x** the seed-noise floor from I0001. The premise
  behind the proposal is not idle.
- **But the instrument records nothing about the batch that explains it**: the
  four recorded batch descriptors explain R² = 0.004 of that variance, and the
  effect does not persist past the step it occurs on (lag-1 autocorrelation
  −0.03).
- **The measurement floors are now quantified.** The CountSketch cosine floor
  is 0.0086–0.016; the alignment between two disjoint 4-row data slices decays
  from 0.21 to **0.015** over training and ends *at* that floor; and the
  row-noise scale implied by across-step decorrelation (≥1400 rows) is at
  least 8x what the recorded `noise/b_noise` says (162 rows). Any future
  per-group gradient measurement must be sized against these numbers.
- **A first-order value model is not sufficient.** `s_k = λᵀg_k` is
  first-order. On the one quantity where the dataset can score a first-order
  value prediction against a realized loss change, the first-order term alone
  gives **R² = −0.57** and a median 97% magnitude error; adding the curvature
  term gives **R² = 0.87** and a 2.5% median error.

---

### Part 1 — mapping table

Read "proxy quality" as: how much of the theoretical quantity survives.

| # | theoretical quantity | what the dataset offers | proxy quality | what it cannot do |
|---|---|---|---|---|
| 1 | **g_k** — per-group gradient | `capture_subbatch_gradients` splits each 32-row device batch into K=8 contiguous 4-row slices and takes an eager unscaled-mean-CE gradient per slice, at 25 periodic checkpoints per run. Also: `sketch/probe_grad` (a fixed 1–4-row held-out slice, 113 (role,layer) blocks × 4096 bins, 30 deep checkpoints) and `sketch/grad` (the accumulated logical-batch gradient = g_nominal). | **Weak.** Slices are disjoint data subsets, but positional, not semantic — and drawn from a clustered window (a 1000-document rolling buffer over consecutive parquet row groups), so they are not independent corpus draws. | Identify any `k`. Recover a slice gradient *vector*: the per-slice sketches are computed in memory and **discarded** — only `noise/per_sub_sq_norm` (8 scalars) and `noise/pairwise_cosines` (28 scalars) reach disk. Nothing can be recomputed differently. |
| 2 | **‖g_k‖** | `noise/per_sub_sq_norm` — ‖g_i‖² for 8 disjoint 4-row slices, 175 checkpoints total. Plus `grad/norm`, `grad/rms`, `sketch/grad_sq_norm`, `noise/mean_grad_norm` per (role,layer) for the nominal gradient, and `calib/grad_norm` (exact) at deep steps. | **Fair for a random 4-row slice; none for a semantic group.** | Attribute a norm to any content. Note: the per-(role,layer) squared norms of each slice *are* computed (`role_sq` in `capture_subbatch_gradients`) and then dropped — a free fix. |
| 3 | **g_iᵀg_j** | `noise/pairwise_cosines` — 28 sketched cosines per checkpoint between the 8 disjoint slices. **The single best proxy in the dataset.** Validated below against an exact algebraic identity. | **Good as an estimator, wrong as a grouping.** The sketch is unbiased (median relative error +0.2% against the exact value) with a per-checkpoint sd of 15% and a cosine floor of 0.009–0.016. | Give any semantic `i, j`. Give a corpus-level cosine: both slices come from the same clustered window, so this is a *within-window* alignment. |
| 4 | **g_kᵀg_nominal** | Fully recoverable per slice from the noise records: `g_i·ḡ = (‖g_i‖² + Σ_{j≠i} g_i·g_j)/K`, with `‖ḡ‖² = signal_raw + s²/K`. Computed below. Also cos(fixed-probe gradient, batch gradient), but the two sketches only share a step at 0. | **The best-populated proxy.** 1400 slice-level measurements across 7 runs. | Same grouping problem. The fixed-probe version is unusable away from step 0: by any nonzero step gap the batch gradient has decorrelated to ~0 (see below). |
| 5 | **H_k** — per-group Hessian | HVPs on **one** loss (the frozen short probe), along three directions (random / gradient / actual update): `curvature/vhv_*`, `gHg`, `dhd`, `Hg_norm`, `eta_star`, plus the FD/symmetry/linearity sweeps. Two acceptance arms. | **None for a group.** The machinery *can* apply H to a chosen direction, which is the piece a future instrument reuses. | Produce a group-restricted Hessian: there is no group-restricted loss to differentiate twice. Also, `DATASET.md` caveat 4 and I0001: native bf16 curvature is uncertified everywhere, and in the shadow arm only the **gradient** direction ever passes. |
| 6 | **λ** — the costate | **Nothing.** Nearest surrogate: the *myopic* costate λ̂ = −∇L_probe(θ_s), implicit in `update/p1 = gᵀΔ`, at 30 deep checkpoints per run. | **Poor.** A myopic costate is by construction the object the optimizer already follows; it carries no information about the remaining trajectory. | Be estimated at all. λ is the adjoint of the *remaining* trajectory; recovering it needs either a backward pass over the whole run (checkpoints for that were deliberately excluded from this local copy) or, for validation, variation in the control — which does not exist (fact 2). |
| 7 | **s_k = λᵀg_k** | The one-step *realized* analogue exists and is well populated: `update/actual` = L_probe(θ_s+Δ) − L_probe(θ_s), with `update/p1` (first order), `update/p2` (+ ½ΔᵀHΔ), residuals, and `update/normalized_residual`, per arm, 215 checkpoints per arm. | **Only as a one-step scoring rule for the *applied* update.** It scores Δ, not any g_k. | Score a counterfactual group. `actual` is measured on the update that *was* taken; there is no arm in which a different mixture was taken. |
| 8 | **Lie bracket H_j g_i − H_i g_j** | **Structurally zero-dimensional.** One loss ⇒ one field ⇒ [v,v] = 0. | **None.** | Anything. **Explicit warning:** `curvature/e_sym_{random,gradient,update}` = \|uᵀHv − vᵀHu\|/max(...) is an *arithmetic symmetry-error diagnostic* of the HVP implementation, not a commutator; repurposing it as a bracket proxy would be a category error. The one legitimately related measurement is `update/normalized_residual`, which bounds how much the local quadratic model — on which the bracket approximation itself rests — misses. |
| 9 | **p_k** — nominal corpus proportion | **Nothing.** `provenance.dataset_files_hash` and 170 shard filenames are the entire record of corpus composition. | **None.** | — |
| 10 | **q_k, β** — the policy | **Nothing varies.** Batch size, composition and order are constant across the whole dataset (verified bitwise). | **None.** | Any policy evaluation, at any β, including β = 0 vs β ≠ 0. |
| 11 | *(adjacent)* per-role / per-layer gradient decomposition | `grad/norm`, `grad/rms`, `sketch/grad`, `param/norm` keyed by (`param_role`, `layer`); 113 blocks. | Fine, but it decomposes the gradient by **parameter block**, an axis orthogonal to the proposal's grouping by **data**. | Substitute for a data grouping. Reported below only to show the decomposition machinery works and that block composition drifts strongly. |
| 12 | *(adjacent)* batch composition | `batch/bos_count`, `batch/mean_segment_length`, `batch/segments_per_row_{mean,max}`, `batch/valid_targets` — every step. | The only data-side observable. Describes **packing structure**, not content. | Explain the batch effect: R² = 0.004 (below). |
| 13 | *(adjacent)* per-segment probe loss | `probe/content_loss_sums` + `probe/segment_lengths` (40-vectors), `probe/per_row_loss`, `probe/transition_loss_sum`, `probe/n_segments` — 37–44 fixed documents per probe, 25 checkpoints. | The closest thing to a per-item `L_k` trajectory. Documents are unidentified but *identical across checkpoints*, so relative movement is trackable. | Give `g_k`, or identify what any document is. |

---

### Part 2 — what the proxies actually support (16 tests, all reported)

**T1 · The sketched pairwise inner product is unbiased against an exact
identity.** From the estimator's own algebra, the mean over unordered pairs of
`g_i·g_j` equals `noise/signal_raw` **exactly** (both derive from the fp32
accumulator, not the sketch). Comparing that against the sketch-derived value
`mean_{i<j} cos_ij·‖g_i‖‖g_j‖` over all 175 checkpoints:

| statistic | value |
|---|---|
| median relative error | **+0.20%** |
| mean / sd | +0.35% / 15.1% |
| p05 … p95 | −7.8% … +5.8% |

So `noise/pairwise_cosines` is a trustworthy estimator of a group-gradient
inner product — the estimator is not the problem.

**T2 · The CountSketch cosine floor.** Two different (role, layer) blocks
occupy **disjoint** parameter coordinates, so their true inner product is
exactly zero; any nonzero sketched cosine is pure estimator noise. Over 1326
such pairs per checkpoint × 175 checkpoints:

| construction | floor |
|---|---|
| per-block null, measured sd | **0.0151** (analytic 1/√k = 0.0156, k = 4096) |
| worst \|cos\| observed on a truly-zero pair | 0.468 |
| full-vector analytic floor `√(Σ_B‖g_B‖⁴/k)/‖g‖²` | median **0.0086** (0.0024–0.0156) |
| full-vector empirical half-split null | mean −0.0023, sd 0.0160 |
| effective blocks carrying gradient mass | median ≈ 3 of 113 |

**T3 · Alignment between disjoint data slices collapses over training.**
Mean cosine between two disjoint 4-row slices at fixed θ, by progress quintile,
pooled over 7 runs (35 checkpoints each):

| progress | 0–.2 | .2–.4 | .4–.6 | .6–.8 | .8–1 |
|---|---|---|---|---|---|
| mean cos(g_i, g_j) | 0.207 | 0.229 | 0.193 | **0.084** | **0.015** |
| sketch floor | 0.0033 | 0.0049 | 0.0099 | 0.0107 | 0.0087 |
| signal / floor | 63× | 47× | 20× | 7.8× | **1.7×** |
| implied row-noise scale `b(1−c)/c` | 15 rows | 13 | 17 | 44 | **265** |

Per-run first/last checkpoint: 0.28→0.01 in every one of the seven runs. This
is the single most important number for a future instrument: **by the end of
training, two independent 4-row data slices have gradients that are
indistinguishable from orthogonal at the recorded sketch precision.**

**T4 · The noise machinery is internally consistent.** The exchangeable-slice
prediction cos = 1/(1 + B̂_noise/b) matches the measured mean cosine with
correlation **0.99936** and median relative difference **1.4%** (n = 173).
`noise/b_noise` and `noise/pairwise_cosines` are not independent evidence; they
are two views of the same numbers.

**T5 · g_kᵀg_nominal, at slice level.** Reconstructed exactly from the noise
records (1400 slice-checkpoint values):

| progress | 0–.2 | .2–.4 | .4–.6 | .6–.8 | .8–1 |
|---|---|---|---|---|---|
| median cos(slice, nominal) | 0.553 | 0.570 | 0.540 | 0.454 | 0.373 |
| sd **across slices** within a checkpoint | 0.036 | 0.036 | 0.036 | 0.047 | 0.031 |
| coefficient of variation | 6.5% | 6.3% | 6.7% | 9.5% | 8.3% |

Disjoint slices genuinely differ in how well they align with the nominal
gradient, by 6–9%, well above the 0.9–1.6% sketch floor. This is exactly the
signal an adaptive policy would exploit — and it is **unattributable**, because
nothing records what is in a slice.

**T6 · Slice index carries no persistent structure (refuted).** The K slices
are *contiguous* row blocks and the best-fit packer is deterministic and
length-biased, so exchangeability was worth checking. Pooled, the
between/within variance ratio is 2.48 and slice 5 sits 2.8 s.e. low — but the
seven runs consume identical rows, so those 175 checkpoints are not
independent. Per-run replication kills it: the sign of each slice index's mean
deviation agrees across only **1/7 to 5/7** runs (a real positional artifact
would be 7/7). Verdict: the sub-batch split is exchangeable in practice; a
positive instrument-validity result.

**T7 · Batch identity dominates the loss channel.** Because all five d12 seeds
consume the identical batch sequence, the component of the detrended per-step
loss shared across seeds *is* the data-attributable component. Steps 40–2494,
51-step centred-median detrend, five seeds:

| component | variance | sd | share |
|---|---|---|---|
| total residual | 5.419e−4 | 0.0233 nats | 100% |
| **batch-attributable (shared)** | 5.372e−4 | **0.0232 nats = 0.743% of level** | **99.13%** |
| seed-idiosyncratic | 4.881e−6 | 0.0022 nats = 0.071% of level | 0.90% |

Intraclass correlation **0.991**. The seed-idiosyncratic figure independently
reproduces I0001's 0.06% sd-relative for `loss/train_mean`. The batch effect is
**10.5×** the seed-noise floor — comfortably above the "2–3× the sd-relative
spread" detectability rule in I0001.

**T8 · Recorded batch descriptors explain almost none of it (refuted).** OLS of
the batch-attributable residual on all four recorded batch descriptors:
**R² = 0.0037**, i.e. 0.4% of the batch-attributable variance. Largest single
correlation \|r\| = 0.053.

**T9 · The batch effect does not persist (refuted).** Autocorrelation of the
batch-attributable residual: lag 1 = −0.028, lags 2–20 all within ±0.06.
Cross-correlation of `batch/bos_count` with the residual: +0.053 at lag 0,
\|·\| ≤ 0.04 at every lag 1–20. Within this instrument's resolution, a single
batch's effect on the loss is confined to the step that uses it. Figure:
`fig/batch_effect.png`.

**T10 · A spurious "composition predicts progress" result, and its refutation.**
Regressing the fixed-probe loss change over each 101-step window on the
window-mean `batch/bos_count`, after removing a cubic in normalized progress,
gives **r = −0.66** (d12, all five seeds identically) and −0.75 / −0.74 at d14
and d16, with a circular-shift permutation p < 0.05 in every run. It is an
artifact of inadequate detrending: fitting the trend against **log** progress
drives it to r = −0.03, and dropping the first three windows drives it to
−0.08 with the sign flipping to +0.63 at d14. Reported because it is the exact
failure mode this dataset invites: seven runs that look like seven replications
are one data ordering measured seven times.

**T11 · A first-order value model is not sufficient.** Update-effectiveness on
the frozen short probe, 215 checkpoints per arm:

| arm | predictor | R² vs realized | median \|rel. err\| | p90 |
|---|---|---|---|---|
| native | p1 = gᵀΔ | **−0.607** | 0.961 | 5.05 |
| native | p2 = gᵀΔ + ½ΔᵀHΔ | 0.865 | 0.082 | 0.656 |
| shadow_fp32 | p1 | **−0.574** | 0.974 | 5.17 |
| shadow_fp32 | p2 | **0.871** | **0.025** | 0.358 |
| shadow, certified (verdict_code_gradient == 0, n = 186) | p1 / p2 | −0.706 / 0.884 | — | — |

`s_k = λᵀg_k` is first-order in exactly the way p1 is. At the step sizes this
recipe actually uses, first-order value is worse than predicting the mean.

**T12 · The local *quadratic* model, however, becomes excellent.**
`update/normalized_residual` = (a − p2)/max(\|a\|,\|p2\|), shadow arm:

| progress third | median | p90 of \|·\| |
|---|---|---|
| early | −0.034 | 0.554 |
| mid | −0.028 | 0.243 |
| late | **+0.0002** | **0.023** |

This matters for question 4: the quadratic approximation that the bracket
`[v_i,v_j] ≈ H_j g_i − H_i g_j` relies on is empirically sound late in training
(0.02% median residual). The obstruction to measuring the bracket is the
missing second group, not the validity of the expansion.

Local *scalars* available before the update predict realized value poorly
(n = 215, Pearson / Spearman against the realized probe-loss decrease):
gᵀΔ **+0.783 / +0.814**; ‖Δ‖ −0.297 / −0.221; ΔᵀHΔ +0.236 / −0.014; η*
+0.113 / +0.240; ‖g‖² +0.099 / −0.158; gᵀHg −0.023 / −0.356. Only the
directional inner product carries real signal. Figure: `fig/headline.png`.

**T13 · What changes the gradient is the data, not the parameters.** The
spec's §7 trio, at matched gaps (both channels live on the deep schedule):

| checkpoint gap | 1 | 2 | 4 | 8 | 24 | 126 | 269 |
|---|---|---|---|---|---|---|---|
| `calib/grad_cosine_prev` (logical batch, exact) | 0.969 | 0.875 | 0.246 | 0.094 | 0.451 | −0.012 | −0.036 |
| `sketch/probe_grad_cosine_prev` (fixed probe) | 0.996 | 0.973 | 0.355 | 0.649 | 0.738 | 0.821 | 0.873 |

For gaps ≥ 100 steps, split by phase: batch-gradient cosine −0.048 (early) /
−0.013 (mid) / −0.008 (late) against fixed-probe cosine 0.484 / 0.791 /
**0.955**. Late in training, parameter motion costs the gradient field only ~4%
of its direction over 126 steps, yet the observed logical-batch gradient has
decorrelated completely. Figure: `fig/grad_change.png`.

**T13b · This contradicts the recorded noise scale, and the recorded one is the
optimistic side.** Correcting the late batch-gradient cosine by the
parameter-motion factor gives −0.009 (median) or +0.155 (p95, conservative),
implying a row-noise scale of **≥1400 rows**. `noise/b_noise`, estimated inside
one 32-row device batch, says **162 rows** at the same progress, which would
predict a 256-row batch-to-batch cosine of 0.61. That is not observed. The
mechanism is in `dataloader.py`: rows within a device batch are packed from a
1000-document rolling buffer over consecutive parquet row groups, so the K
slices are a **clustered** sample, not independent corpus draws. This sharpens
`DATASET.md` caveat 8: `noise/b_noise` is a *lower bound* on the corpus-level
noise scale, off by at least 8×, and must not be used to size a per-group
gradient estimator.

**T14 · Parameter-block composition drifts hard (the wrong axis, shown to
work).** Share of ‖g‖² by role, d12 mean over seeds: unembedding 0.999 → 0.055;
mlp_out 0.001 → 0.604; attn_out 0.0002 → 0.075. Largest \|Δ share\| between
step 101 and the last checkpoint: mlp_out 0.377, unembedding 0.230, attn_q
0.156. The decomposition machinery is sound; it simply decomposes by parameter,
not by data. Figure: `fig/sketch_geometry.png`.

**T15 · Per-item loss: heterogeneous improvement, almost no re-ordering.** Over
37–44 fixed probe documents per run:

| quantity | value |
|---|---|
| Spearman(per-segment loss at first checkpoint, at last) | **0.82** (0.78–0.96) |
| Spearman between adjacent checkpoints | 0.993 |
| mean per-document loss reduction | 1.23 nats |
| sd of that reduction across documents | **0.64 nats (52% of the mean)** |

Documents differ enormously in how much they are learned, while their
difficulty *ranking* stays nearly frozen. Figure: `fig/segment_loss.png`.

---

### Part 3a — the five questions

| # | question | verdict | why |
|---|---|---|---|
| 1 | Does the relative usefulness of different data groups change during training? | **unavailable** | No groups. The nearest measurable statements: the alignment of a data slice with the nominal gradient falls 0.55 → 0.37 (T5), the shared signal between disjoint slices falls 0.21 → 0.015 (T3), and individual documents improve at rates that differ by 52% of the mean while their difficulty ranking barely re-orders (ρ = 0.82, T15). All three say the *geometry* of data-value estimation changes over training. None is a statement about groups. |
| 2 | Can we retrospectively estimate the future usefulness of a group's gradient? | **unavailable — and unanswerable in principle on this data** | Not merely unlabelled: there is exactly one data ordering, replicated seven times bitwise. Zero variation in the treatment ⇒ no counterfactual ⇒ no `s_k`. Empirically the loss channel also shows no persistence of a batch's effect past the step that uses it (T9). |
| 3 | Which recorded local quantities best predict future usefulness? | **partially supported**, for *one-step realized* usefulness of the *applied update* only | Among available analogues: the directional inner product gᵀΔ is the only informative scalar (r = +0.78, ρ = +0.81); ‖g‖² (+0.10/−0.16), gᵀHg (−0.02/−0.36), ΔᵀHΔ (+0.24/−0.01) and η* (+0.11/+0.24) are near-useless alone (T12). Crucially, a first-order model of value is worse than the mean (R² = −0.57) while adding curvature reaches R² = 0.87 (T11). The bracket proxy has no analogue at all. |
| 4 | Is there evidence for the noncommutativity hypothesis? | **unavailable** | With one loss there is one gradient field and [v,v] = 0. The dataset applies H to exactly three directions on one probe loss. `curvature/e_sym_*` is an arithmetic diagnostic and must not be mistaken for a commutator. The one positive contribution: the quadratic expansion the bracket approximation rests on is empirically accurate late in training (median normalized residual 2e−4, T12). |
| 5 | What is the smallest set of telemetry that appears sufficient to predict the preferred data mixture? | **unavailable as posed; a concrete answer to "what must be added" is below** | Nothing in the current 283 families can predict a mixture, because no mixture is defined and none varies. The measured floors (T2, T3, T13b) do, however, pin down what a sufficient set must clear. |

---

### Part 3b — instrumentation proposal

Costs are measured from `overhead/total/*` and `step/observed_dt` in the seven
segments (d12 baseline: median training step **979 ms**; existing total
telemetry overhead **21.8%** of wall at d12, 14.1% at d14, 10.5% at d16).
Measured unit costs at d12: one 4-row eager fwd/bwd + sketch + norms **109 ms**
(≈27 ms/row); full-model gradient scan + sketch **430 ms**; probe-gradient
sketch **89 ms**; exact-gradient calibration **4.72 s**; update-effectiveness +
HVP suite **5.21 s**; shadow-fp32 arm **5.11 s**. Storage baseline: 112 MB per
d12 run, of which ~46 MB is `sketch/grad`.

#### Tier 0 — prerequisites (without these nothing else helps)

| what | detail | cost |
|---|---|---|
| **Loader sidecar** (spec §8, already specified, still deferred) | per packed row, per segment: `group_id` (the mixture label — this is the one field the whole proposal turns on), stable source locator (shard, row_group, row_index), original token length, crop flag, per-token segment id. Carried through tokenization, the best-fit buffer, and cropping. | CPU only; ~15M values/run ≈ tens of MB. **~0% wall.** |
| **`p_k` in provenance** | the nominal mixture proportions actually realized, computed from the sidecar per run and per step-window. | negligible |
| **≥2 data orderings** | *not telemetry.* A shuffle seed plumbed into the loader, plus at least one run pair that differs only in mixture (e.g. β = 0 vs β ≠ 0 at fixed `p_k`). Without this, `s_k` stays unidentifiable no matter what is recorded. | **1 extra run per arm.** Unavoidable. |

#### Tier 1 — the minimum viable set (answers Q1, Q3, and the myopic part of Q5)

| # | what to record | cadence | measured/estimated cost |
|---|---|---|---|
| 1 | **Group-pure gradient-accumulation microbatches.** Ask the loader to make each of the 8 accumulation microbatches group-pure. Each microbatch's gradient *is* a group gradient at 32 rows, for free — no extra forward or backward. Declare it as an intervention arm (it changes data order). | every step | **0% wall** (reordering only) |
| 2 | Per-microbatch (i.e. per-group) **CountSketch + exact per-(role,layer) squared norms**, reusing `sketch_named_tensors`. | 25 periodic checkpoints | 8 × ~(sketch part of the 430 ms scan) ≈ **1.5 s/ckpt → +1.4% wall** |
| 3 | Derived and stored instead of raw sketches: the **G×G sketched Gram matrix**, `g_k·ḡ`, `‖g_k‖` per role, and `n_k`. | same | ~50k scalars/run, **< 5 MB** (vs 370 MB if raw per-group sketches were stored — do not store them) |
| 4 | **Fix the existing free losses**: emit the per-(role,layer) `role_sq` already computed per sub-batch, and emit the 8 sub-batch sketches. | 25 periodic | ~0 compute; +46 MB/run if sketches are stored |
| 5 | **Held-out validation-probe gradient sketch** at the periodic cadence (extend `sketch/probe_grad` to the val probe) ⇒ the myopic costate λ̂ and ŝ_k = λ̂ᵀg_k for free from (3). | 25 periodic | 89 ms × 25 = **+0.08%** |
| 6 | **Exact `g_k·g_j` calibration** for the sketched Gram, reusing the existing exact-gradient path. Needed because the sketch floor (0.009–0.016) is within a factor of 2 of the late-training signal (T3). | 6 deep checkpoints | 4.72 s × 6 = **+1.1%** |
| 7 | **Independent-draw noise estimate**: one extra device batch fetched from a *different* loader position at the same θ, to replace the clustered within-device-batch estimate that T13b shows is ≥8× optimistic. Record it as a distinct `estimator_id`. | 25 periodic | 0.9 s × 25 = **+0.8%** |

**Tier 1 total: ≈ +3.4% wall, < 55 MB/run**, on top of the existing 21.8%.

**Sizing constraint that Tier 1 must respect** (from T3, the row-noise scale
implied by the measured slice cosines): rows per group needed for a self-cosine
of at least *c* is `B·c/(1−c)` —

| progress | 0–.2 | .2–.4 | .4–.6 | .6–.8 | .8–1 |
|---|---|---|---|---|---|
| c ≥ 0.3 | 7 rows | 6 | 7 | 19 | **114** |
| c ≥ 0.5 | 15 | 13 | 17 | 44 | **265** |
| c ≥ 0.8 | 61 | 54 | 67 | 176 | **1060** |

and T13b says these are **lower bounds by ≥8×**. Consequences: (a) 32-row
group-pure microbatches are adequate early and marginal after progress ≈ 0.6;
(b) late in training a usable per-group gradient needs on the order of the
whole 256-row logical batch, so `g_k` must be **accumulated over a window of
steps** (an EMA over the per-group sketches, which costs nothing extra) rather
than estimated within one step; (c) G should be small — 4–8 coarse groups — not
tens.

#### Tier 2 — curvature and the bracket (only if Tier 1 shows a signal clearing the floors)

| # | what | cadence | cost |
|---|---|---|---|
| 8 | Extend the HVP entry point to accept a **group-restricted loss** and an arbitrary direction, giving `H_k v`. Record `λ̂ᵀ(H_j g_i − H_i g_j)` as a scalar per designated pair — never the raw vector. | a designated set of ~6 group pairs, 10 deep checkpoints | 2 HVPs/pair ≈ 0.3 s each ⇒ 3.6 s/ckpt ⇒ **+1.4%** native; **+2.8%** if run in the shadow arm |
| 9 | Run 8 in the **shadow_fp32 arm with per-direction verdicts**. I0001 shows only the shadow gradient direction ever passes; a bracket measured in the native bf16 arm would be uncertified by construction. | same | included above |
| 10 | Per-group `η*`, `g_kᵀH_k g_k` at the same checkpoints. | 10 deep | folded into 8 |

**Tier 2 total: ≈ +3%.** Combined Tier 1 + Tier 2 ≈ **+6.5% wall** — the
existing budget would rise from ~22% to ~28% at d12, which is within the
spec's own tiering philosophy if the deep-tier cadence is trimmed to
compensate.

#### What Tier 1 + Tier 2 still cannot do

They make Q1, Q3 and Q4 measurable. They do **not** make Q2 answerable, and
therefore do not validate `q_k ∝ p_k e^{βs_k}`. That needs Tier 0's third row:
runs that differ in the mixture. The cheapest decisive design is a
three-arm bridge at d12 — nominal `p_k`, a fixed non-nominal mixture, and a
telemetry-driven adaptive mixture — with the five-seed d12 spread from I0001 as
the noise floor. I0001's rule (an effect must clear 2–3× the sd-relative
spread) plus T7 (the batch effect is 10.5× the seed floor on
`loss/train_mean`) says such an arm would be detectable on the loss channel
with **five seeds per arm**, and on nothing else: curvature channels need
50–75% effects, which no plausible mixture change will produce.

---

## Limitations

- **The seven runs are not seven replications for anything data-related.** They
  consume one bitwise-identical batch sequence. Every result here that involves
  the data stream (T3–T5, T7–T10, T13) has an effective n of one ordering.
  T10 is the worked example of how badly this can mislead.
- **T7's decomposition assumes the shared component is data-driven.** It is,
  given that the five d12 runs differ only in initialization and the schedule
  is removed by detrending — but the 51-step centred-median detrend also
  suppresses any *slow* batch effect, so T9's "no persistence" applies to
  lags 1–20 at that filter, not to accumulation over hundreds of steps.
- **The "fixed probe" behind every update-effectiveness and curvature number
  (T11, T12) is the short probe: 4 rows × 256 tokens = 1024 target positions**,
  drawn from the held-out val shard. The myopic costate λ̂ = −∇L_short(θ_s) is
  therefore a gradient on ~1k tokens. Its *direction* is stable (T13:
  `sketch/probe_grad_cosine_prev` = 0.96 over 126 steps late), but it is a very
  small sample of the objective, and R² values in T11 are R² against that
  probe's loss change, not against validation loss.
- **`DATASET.md` caveats that apply directly**: 8 (batch construction is
  under-instrumented — this whole result is an elaboration of it, and T13b
  sharpens it); 4 and the I0001 finding that only the shadow gradient direction
  is certified (all headline curvature statements in T11/T12 are shadow-arm;
  the native numbers are quoted alongside as uncertified); 6 (probe-derived
  quantities carry sampling variance — T15's documents are one frozen draw per
  run; note the val and short probes come from the held-out val shard, but the
  `train_stream` probe used in T15's second row is literally the first 16 rows
  of the training stream and is *not* held out); 3 (the absolute 40-step
  warmup, handled by dropping steps < 40 in T7 and by the drop-first-windows
  robustness in T10); 9 (multiple comparisons — 16 tests, all reported, all
  exploratory).
- **T13's "data vs parameter motion" split uses the fixed 1–4-row probe
  gradient as a proxy for the stability of the mean gradient field.** A 1–4-row
  gradient is not the mean gradient; if the probe's direction happens to be more
  stable than the corpus mean's, T13b's ≥8× understatement is an overestimate.
  The conclusion (that the recorded noise scale is optimistic) survives at the
  conservative p95 bound, but the factor is uncertain.
- **The cost estimates for Tier 2 extrapolate** a bare HVP from the acceptance
  suite's `fwd_equiv` work model rather than measuring one; the suite bundles
  ~50 forward-equivalents per checkpoint and I attributed ~3 to a single HVP.
  Treat Tier 2's +3% as an order of magnitude.
- **`d12-iter` was excluded** as instructed; no v1 data was read.
- Anomaly noted in passing, not claimed: at the warmdown-start landmark in both
  d14 (step 1315→1316) and d16 (1881→1882), two logical-batch gradients one
  step apart have `calib/grad_cosine_prev` of −0.83 and −0.87 while the fixed
  probe gradient stays at 0.95 and 0.91. That is a two-run, two-point
  observation at a schedule kink and deserves its own investigation.
