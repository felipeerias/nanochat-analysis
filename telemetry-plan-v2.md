# nanochat Telemetry Plan v2 (converged review)

> **Status update (2026-08-24).** After the H100 shakedown and a four-way
> uncorrelated strategic review, the sweep plan changed: the first collection sweep is
> d12 x 3 seeds + d14 + d16 at upstream defaults (head_dim 128 - the hd64
> pin is reverted), with d18/d20 conditional and d6-d10 dropped (absolute
> Muon momentum warmup dominates small depths). Deep telemetry uses
> Pythia-shaped per-run step sets aligned in normalized progress; periodic
> cadence is per-run (ceil(N/25)). Lineage checkpoints (theta_0 + interior)
> are saved into each segment. Manifests are immutable, machine-readable
> run tables embedded in each segment. The acceptance suite runs two arms
> (native + IEEE-fp32 shadow) with per-arm verdicts, and eta* uses a
> relative reliable-sign gate. The freeze gate is an analysis notebook
> reproducing a result from parquet alone. Details: telemetry-spec.md and
> the telemetry branch commit history.



Original proposal by GPT-5.6 Sol (chat context, no repo access); reviewed and
revised against the actual nanochat code by Claude (Fable 5) and Codex running
GPT-5.6 Sol at xhigh effort — i.e. the same base model as the proposal author,
but code-grounded — 2026-08-22. Three review rounds; both reviewers signed
off. Claude independently verified all load-bearing code claims.

## Execution model (updated 2026-08-22)

The local workstation is NOT a design constraint. Its role is limited to
developing the telemetry code itself: the d2 smoke loop for correctness, unit
tests, and the HVP acceptance suite at small depth. All actual measurement
runs happen on rented GPUs. Default measurement platform: a SINGLE rented
GPU (H100-class) — single-device keeps telemetry simple (no DDP gradient
all-reduce to work around) and a compute-optimal d12 costs roughly $2-3 and
under an hour per run. Multi-GPU nodes only when wall-clock demands it, and
only after telemetry is made DDP-aware (per-rank pre-all-reduce capture is
extra work — deferred). Depth sweeps should target the real miniseries range
*(superseded 2026-08-24: the first collection sweep is d12-d16 with d12 seed replicates;
d6-d10 dropped - see the status note at the top)*.

Note for GPU measurement runs: single-GPU training at the standard total
batch size runs gradient accumulation naturally (e.g. ~16 microbatches at
d12), so natural sub-batches exist for noise statistics — but the eager
separate-diagnostic design below still applies; do not hook the compiled
training backward. Scope: **observational phase only** —
intervention experiments (batch construction, adaptive LR rules) need their own
design review later, including loader sidecars, trajectory isolation, and
compute-matched controls.

## Corrections to the original proposal

1. **Muon update norms are analytic, not emergent.** nanochat's Muon
   (`nanochat/optim.py`) runs Nesterov momentum → row equilibration → Polar
   Express orthogonalization → Muon+ renormalization (Frobenius norm snapped to
   √min(m,n)) → norm-preserving factored variance scaling → cautious weight
   decay, and the group LR is multiplied by √max(1, m/n) (`optim.py:393`).
   Net result: `‖ΔW_data‖_F = group_lr·√m` by construction. Update norms,
   relative update sizes, and update effective-rank for Muon matrices are
   optimizer invariants (useful as sanity checks), not research observables.
   Measure instead: gradient/momentum/final-direction alignments, polar
   orthogonality residual, factored-scale dispersion, cautious-mask fraction,
   data-vs-decay update norm split, observed-vs-analytic norm deviation.
2. **cos(g_t, g_{t−1}) across steps conflates data noise with parameter
   motion.** Replace with three separate diagnostics: fixed-probe gradient
   cosine across checkpoints (parameter-motion effect), independent
   sub-batch gradients at fixed parameters (data-noise effect), and the
   successive-batch cosine kept only under its narrow label.
3. **Sign convention:** with Δ = θ_after − θ_before, descent alignment is
   cos(−g, Δ); the proposal's cos(g, Δ) is normally negative.
4. **Step-0 metrics are undefined for zero-initialized matrices** (attention
   and MLP output projections, `gpt.py:229`): ‖g‖/‖θ‖, stable rank, and
   residual contributions must carry an explicit `undefined` flag, not an
   epsilon-fudged value.
5. **Attention statistics require a separate manual recomputation.** Training
   uses FA3 (Hopper bf16) or SDPA — attention probabilities never exist.
   Entropy/distance stats must be recomputed with manual QK-softmax that
   reproduces rotary, QK RMS-norm ×1.2, 1/√d scale, causal mask, and the
   layer's sliding window; normalized as H/log(n_allowed), stratified by
   query position, short vs full-window layers separated (`gpt.py:287`).
6. **HVPs need an eager math-SDPA path.** Verified on torch 2.9.1: default CPU
   SDPA fails double-backward (`aten::_scaled_dot_product_flash_attention_for_cpu_backward
   not implemented`); forcing SDPBackend.MATH succeeds. The model is
   torch.compile'd (`base_train.py:246`); all probes run on the eager
   `orig_model` reference (shared parameters), hooks registered only around
   probe forwards and removed after.
7. **η·λ_max is not a Muon stability statistic** (no single scalar LR / fixed
   preconditioner). Use curvature along the actual update, ΔᵀHΔ.
8. **"Examples" are packed rows, not documents.** The loader best-fit-packs
   whole BOS-prefixed documents into fixed-T rows, cropping to fill
   (`nanochat/dataloader.py`). Segment stats are recoverable from BOS
   positions; document IDs/sources are not (loader resume state is not a doc
   ID). Sequence-length distribution is degenerate (always T) — dropped.
9. **η* = gᵀg/gᵀHg is meaningful only where gᵀHg is sufficiently positive**
   (indefinite Hessian); power iteration finds largest-magnitude, not largest
   algebraic eigenvalue — use Lanczos with an explicit convention.
10. **train/loss as currently logged is the last microbatch, EMA-smoothed**
    (`base_train.py:511`) — fix to a correctly weighted logical-batch mean.

## Key protocol decisions

- **Pin `--head-dim=64` for the primary cross-depth sweep** (d2…d12)
  *(SUPERSEDED 2026-08-24 — see the status note at the top: upstream
  head_dim 128, sweep d12–d16, small depths dropped)*: width
  then equals 64·depth exactly at every depth (default hd128 rounds d3/d4 both
  to width 256, aliasing depth regularities). Note hd64 means more, narrower
  heads than stock. Label results `nanochat@hd64`; optionally run a small
  stock-hd128 sensitivity panel at d4/d6/d12. Pin sequence length, window
  pattern, batch construction, and schedule across depths.
- **Gradient noise scale is first-sweep batch** (batch-construction agenda).
  On diagnostic steps: clone the current x,y BEFORE the prefetch at
  `base_train.py:518` overwrites the persistent buffers, split into K=4/8/16
  sub-batches, run eager fwd/bwd per sub-batch at pre-update parameters with
  temporary hooks reducing to per-role squared norms + sketches (nothing
  retained), clear grads/hooks/RNG, then run the normal compiled step
  untouched. (Single-GPU measurement runs already accumulate ~16 natural
  microbatches at d12, but the separate eager diagnostic is still the design —
  never hook the compiled training backward.) Cost ≈ one extra logical-batch
  fwd/bwd (1.5–3× wall on diagnostic steps), NOT K×. Scaling: unscaled
  sub-batch losses for noise stats (undo any 1/K); log s² = (Σ‖g_i‖² −
  K‖ḡ‖²)/(K−1), bias-corrected signal ‖ḡ‖² − s²/K, and components separately.
  Sampling unit is the packed row.
- **Sketches: CountSketch/feature hashing, O(P),** k≈4096 (calibrate at d2/d4
  against exact inner products). Dense k×P Rademacher projection is
  computationally unsuitable. Fixed telemetry seed INDEPENDENT of run seed;
  mapping keyed by (schema_version, canonical param name, shape, flat coord).
  Separate sketches per role/layer. Cross-depth comparisons only on derived
  scalars; cross-run raw-gradient cosines are weak evidence anyway
  (permutation symmetries).
- **Probes:** frozen validation probe + held-out train-domain probe (reserve a
  probe shard excluded from training — otherwise it's a "frozen
  training-stream probe", not held out), 16 rows × T=512 each, naturally
  packed by the real loader algorithm on reserved documents. Separately packed
  T=256 short probe (not truncated rows) for manual attention (2–4 rows) and
  HVPs (start 1 row × 128–256). Per-segment loss comes from unreduced CE in
  the probe forwards (both training-objective and content-only variants;
  final segment's crop status is unknowable from tokens).
- **HVP acceptance suite before any Lanczos:** central finite difference
  Hv ≈ [g(θ+εv) − g(θ−εv)]/2ε with an ε-sweep (find the plateau between
  truncation and fp32 cancellation); symmetry uᵀHv ≈ vᵀHu; linearity;
  vᵀHv vs second-difference of the loss; random + gradient + actual-update
  directions; fp64 dot-product accumulation. True-fp64 model spot check needs
  a small diagnostic dtype path (COMPUTE_DTYPE only supports bf16/fp16/fp32;
  model.double() alone is insufficient).
- **Update effectiveness:** log primitives separately — p1 = gᵀΔ,
  p2 = gᵀΔ + ½ΔᵀHΔ, actual a = L(θ+Δ) − L(θ), residuals a−p1, a−p2,
  normalized by max(|a|,|p2|,ε). Ratios formed offline. Fixed probe batch,
  exact actual Δ including weight decay.
- **Storage:** partitioned Parquet dataset (atomic chunk files, not one
  append-file), dictionary-encoded strings, sketches as fixed-size arrays.
  Fields: run_id, schema_version, telemetry_config_hash, event_id, step,
  tokens_seen, normalized_progress, phase(pre/post), checkpoint_id, probe_id,
  param_role, parameter_name+shape, optimizer_kind, layer, head, segment_id
  (nullable, separate), metric, value, estimator_id, aggregation, sample_count,
  sketch_seed, units, dtype, backend (event-level), wall_time,
  telemetry_elapsed. W&B receives only the cheap tier for live monitoring.
- **Methodological safeguards:** frozen probe tensors (never advance the
  training loader for probes); autograd.grad or rigorous grad clearing so
  probes never leak into optimizer state; RNG save/restore; explicit pre/post
  update phase on every record; telemetry logs its own time/FLOPs/memory
  overhead; telemetry-disabled baseline runs to confirm trajectory and
  throughput are unperturbed; multiple seeds (current seed is hardcoded 42,
  `common.py:183` — plumb a --seed). Measurement runs are uniformly GPU
  (bf16, FA3/SDPA-CUDA), which removes the CPU-vs-GPU backend confound from
  the sweep itself; deconfound precision instead by repeating one depth with
  NANOCHAT_DTYPE=float32 on the same GPU. Align runs on tokens/normalized
  progress, not raw step.

## Converged collection tiers

### Continuous (every step; scalar or O(BT) work only)
Correctly weighted logical-batch mean loss; step/tokens/progress/phase; actual
per-group LR, Muon momentum, weight decay, Adam hyperparameters; train-step vs
data-loading vs telemetry timing; finite/nonfinite + skipped-step flags;
learned scalars (resid_lambdas, x0_lambdas, smear_lambda, backout_lambda);
BOS/segment count and length summaries; run/backend/dtype identifiers;
telemetry overhead and memory watermark. (Full-gradient sketches are NOT in
this tier unless later fused into optimizer reads and benchmarked.)

### Periodic (cadence set by an overhead budget, e.g. ≤5-10%)
Per-role/layer parameter and accumulated-gradient norm/RMS (explicit zero
handling); CountSketch of accumulated gradient + successive-checkpoint
sketched cosine; both fixed probes: loss, per-row/token/segment loss, logit
max/LSE/entropy/softcap saturation (softcap=15, `gpt.py:510-511`),
residual/attention/MLP activation moments, ReLU² sparsity/tails,
value-embedding gate and contribution stats; Muon internals (alignments,
orthogonality residual, factored-scale dispersion, cautious-mask stats,
data/decay norm split, expected-vs-observed norm); K-way sub-batch noise
diagnostic; short-probe manual attention entropy/distance for selected
layers, normalized by allowed context.

### Sparse deep probes
Exact previous-gradient cosine at calibration checkpoints only; exact
parameter delta for selected updates; fixed-probe before/after update loss
with p1/p2/actual primitives; HVP acceptance suite (validated at d2 locally,
re-validated once on the GPU backend); HVP along gradient and actual update
at the measurement depths after memory/backend validation;
Lanczos only after acceptance tests pass; limited per-packed-row gradient
sketches on tiny diagnostic batches; opt-in loader sidecar experiments
(source locator (shard, row_group, row_index) + content hash, domain,
original length, crop flag, per-token segment ID) — prerequisite for the
intervention phase.

### Offline from checkpoints
Distance/angle from initialization; inter-checkpoint displacement and path
length; parameter/optimizer-state quantiles and distributions; stable/
effective rank; randomized spectral estimates for a predefined matrix set;
cross-run/depth aggregation; overhead and backend-confound analysis;
hd64-vs-hd128 sensitivity comparison.

### Dropped or replaced
Muon polar-stage update norm/rank as research metrics; dense Rademacher
projections; continuous exact gradient retention; successive-batch cosine as
"temporal gradient evolution"; full attention-map storage and all-pairs
attention comparisons; full SVDs of all matrices; full per-example gradient
tensors; layerwise Hessian eigenspectra; scalar η·λ_max for Muon; ρ as the
primary stored quantity; document IDs inferred from loader resume state;
sequence-length distribution (degenerate at fixed T).

## Existing-logging gaps to fix first
Logical-batch loss averaging (`base_train.py:511`); unambiguous pre/post-step
semantics (current step 0 mixes validation and the first update); provenance
(git SHA + dirty flag, derived model/batch/schedule values, compute dtype,
attention backend, tokenizer/dataset identity, software versions, telemetry
config); actual per-group LRs rather than only the LR multiplier; peak memory
logged rather than printed at shutdown; note MFU is meaningless on CPU (peak
FLOPS set to infinity, `base_train.py:95`).
