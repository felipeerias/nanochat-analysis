# E09 — how much of the curvature record is a 256-token artifact

Status: **draft, not frozen**. Runs offline on saved checkpoints. Needs no new
training and no instrument change.

## Question

At a fixed model state, how much do curvature quantities depend on the **probe
sequence length**, does the sharpening trajectory have the same shape at each
length, and do the acceptance verdicts change?

An answer either way is useful and immediately actionable. If the T=256 surface
and the T=2048 surface agree to within the initialization noise floor, the
entire certified curvature record stands as written, `DATASET.md` caveat 4 gets
downgraded from a warning to a measurement, and E06 can size its probe in rows
without a second thought. If they disagree by more than that, every published
curvature magnitude is a T=256 quantity that must be relabelled, and E06's
sizing grid is calibrating a number nobody should quote.

## Why this is the question

`DATASET.md` caveat 4, restated exactly: the `short` probe holds four rows, but
the HVP path takes only the first —

```python
telemetry.py:2484:    self._hvp_x, self._hvp_y = sx[:1], sy[:1]
```

— so every Hessian-vector product, curvature value, acceptance verdict and eta*
in this dataset describes the loss surface of **one 256-token sequence**.
Training optimizes a mean over 524,288 tokens arranged as 256 rows of 2,048.
These are different functions, and nothing in the project has ever measured the
gap.

The record that rests on it is not small: I0005's sharpening trajectory (gHg
flat under constant LR, then ~15x through warmdown, then plateau), I0001's seed
reference (gHg 29% sd-relative, eta* 25%, dhd 13%), I0004's depth verdict
(decided on `e_sym_gradient`), and every acceptance verdict in 2.69M records.
`telemetry-v4-plan.md` §3 already names enlarging the probe "the single cheapest
variance reduction available" — but it names it in **rows**. This design is
about the other axis.

### Sequence length is not just a size knob — the architecture makes it special

`window_pattern = "SSSL"` is tiled across layers with the final layer forced to
long, and the short window is `ceil(sequence_len/4 / 128)·128` = **512 tokens**
at `sequence_len = 2048`. At d12 that gives 9 short-window layers and 3
long-window ones; at d14, 10 and 4; at d16, 12 and 4.

A probe at T never reaches a boundary the window does not impose, so the
effective attention span per layer is `min(window, T)`:

| T | short-window layers span | long-window layers span | Σ_l min(w_l, T), d12 | what the model *is* on this probe |
|---:|---:|---:|---:|---|
| 256 | 256 | 256 | 3,072 | a plain 12-layer full-context transformer |
| 512 | 512 | 512 | 6,144 | still full-context — the last T at which windowing is inert |
| 1024 | 512 | 1024 | 7,680 | windowing first bites, for positions ≥ 513 |
| 2048 | 512 | 2048 | 10,752 | the training configuration |

**On the existing probe, 9 of d12's 12 layers — 75% of the model — are measured
outside their attention regime**, and short-window and long-window layers are
literally the same function. The same fraction holds at d16. Any curvature
statement about attention structure, and any future intervention on the window
pattern, is inert on a T=256 probe. `telemetry-v4-plan.md` item 7 already flags
this for the Gram probe and defers the decision; this design is what settles it.

Note also that the sensitivity is *not* smooth in T. There is a qualitative
change crossing T=512. A two-level design at {256, 2048} could not tell "longer
context" from "windowing turned on", which is why the ladder below has four
levels and why 512 is one of them.

## What is already frozen, hashed, and free

This design generates no new probe data. Everything it needs exists:

| artifact | contents | status |
|---|---|---|
| `probe_val.pt` | **16 rows × 2048 tokens**, val split | id `066a03cc…`, identical in **all seven runs at all three depths** (verified from provenance) |
| `probe_short.pt` | 4 rows × 256 tokens, val split, packed natively at T=256 | id `2d9baec3…`, identical in all seven runs |
| `probe_train_stream.pt` | 16 rows × 2048, training stream | id `5c54f16c…`, identical in all seven runs |
| lineage checkpoints | θ₀ + 3 interiors + final triplet, hash-inventoried | Runpod volume `nanochat_experiment` (AP-JP-1, ~70 GB) |

The probe identity across depths matters more than it looks: it means the T
ladder is evaluated on the **same 16,384 tokens** at d12, d14 and d16, so the
Stage 2 depth comparison is a comparison of operators and not of samples.

As with E06, the checkpoints are **not** in the local copy; this design runs on
a pod with the volume mounted.

## Anchors

Lineage labels sit at the same normalized progress at every depth, which is what
makes them usable here:

| anchor | d12 | d14 | d16 | progress | regime (I0005) | role |
|---|---:|---:|---:|---:|---|---|
| A0 | 0 | 0 | 0 | 0.000 | initialization | stress case only (54 structurally zero-gradient matrices, I0007) |
| A1 | 631 | 941 | 1345 | 0.250 | constant LR, gHg flat | **the "before" end of the sharpening ratio** |
| A2 | 1261 | 1880 | 2689 | 0.500 | mid-warmdown, gHg rising | secondary |
| A3 | 1890 | 2820 | 4032 | 0.750 | late warmdown | secondary |
| A4 | 2520 | 3759 | 5376 | 1.000 | plateau | **primary** |

Warmdown begins at progress 0.350 at every depth, so A1 is the only anchor
before it. The sharpening ratio is therefore defined as R = Q(A4)/Q(A1), a
within-run quantity, exactly as I0005 treats it.

Four usable anchors is a real limitation and §"What this does not answer" says
so: this design samples the trajectory, it does not trace it.

## Factors, levels, what is held fixed

**Factor 1 — probe sequence length T ∈ {256, 512, 1024, 2048}, nested by
truncation.** Row *r* at length T is the first T tokens of row *r* at 2048. The
nesting is exact, so T is the only thing that varies between levels and every
contrast is paired within row.

**Factor 2 — row, 8 levels.** Rows 0–7 of the frozen `probe_val`. Rows are the
replication unit for the between-row spread and the pairing unit for the T
contrast. They are *not* the unit of inference for the primary test (see
"Power").

Held fixed: depth 12 (Stage 1), the shadow fp32 arithmetic and its construction
(`build_shadow_model` under `shadow_precision(torch.float32)`, TF32 off, matmul
precision "highest", rotary rebuilt in shadow arithmetic, `estimator_id =
"hvp-shadow-fp32-ieee-v1"`), the direction set, the FD and curvature epsilon
sweeps, `tolerance_version = 1`, `telemetry_config_hash`, the reduction order,
and **the chunk size at one row at every T**.

That last one is deliberate and it is a deviation from E06, which fixes c = 4
rows. Holding the chunk at one row means the arithmetic path is bit-for-bit the
same shape at every T, so nothing in the T contrast is an artifact of a
different reduction tree. If E06 freezes c = 4 at T=256 first, this design adds
one bridge cell at c = 4 and reports the difference as a pure arithmetic-path
term with no probe-sampling content.

**Loss reduction, precommitted: mean over valid targets**, matching the training
loss. `n_valid_targets` and the BOS-position count are recorded per bank so any
re-weighting is reconstructible after the fact.

## Three extra banks, each answering something the ladder cannot

**Bank L — the legacy bridge.** The four rows of the frozen `probe_short` at
T=256, plus row 0 alone. Row 0 of `probe_short` is *the* object every published
curvature number in this project was computed on. This bank is the tie between
the new measurements and the existing record, and it also isolates a confound
the truncation ladder deliberately suppresses: `probe_short` was packed natively
at T=256 by its own loader (`device_batch_size = 4`, `T = 256`), so its rows are
a different document composition from the first 256 tokens of a 2048-token
best-fit packing. Bank L quantifies that difference at one point; the ladder
holds it fixed everywhere else.

**Bank I — the iso-token cell**, which is the joint measurement with E06. Take
the same 8 primary rows — **exactly 16,384 tokens** — and cut them four ways:

| bank | shape | tokens | attention regime |
|---|---|---:|---|
| I-256 | 64 rows × 256 | 16,384 | windowing inert |
| I-512 | 32 rows × 512 | 16,384 | windowing inert |
| I-1024 | 16 rows × 1024 | 16,384 | windowing partial |
| I-2048 | 8 rows × 2048 | 16,384 | training configuration |

Same tokens, same order, different cuts. This is the one measurement that says
whether a probe should be specified as **m·T tokens** or as **m rows** — which
is E06's open question 1, asked in a form that has an answer. Segments cut from
the middle of a packed row begin without a BOS and without context; that is
precisely the property under test, and the per-segment BOS count is recorded so
the effect can be attributed.

**Bank R — platform repeatability.** Eight reruns of one (state, row, T=2048)
cell with identical inputs. The compiled embedding backward contains an
atomic-accumulation race (DATASET caveat 8) and an HVP passes through that
backward twice, so without this control platform jitter would be silently
attributed to sequence length.

## Primary outcome, precommitted

**Δ = log(vhv_gradient at T=2048) − log(vhv_gradient at T=256)**, on
`acceptance_arm = "shadow_fp32"`, restricted to cells where the
**gradient-direction** verdict passes at both lengths, averaged over the 8 rows
within each (seed, anchor) cell.

`vhv_gradient` is the primary rather than `gHg` because it is the curvature
along a *normalized* direction and so is insensitive to the gradient magnitude,
which itself changes with T for reasons that have nothing to do with the
landscape (more target tokens, lower gradient noise). Per I0005, eta* is its
exact reciprocal to 1.2e-15, so the two must never be reported as corroborating
each other; eta* is reported as a derived convenience only.

The gradient direction is the only choice available: I0001 and I0004 together
establish that the random and update directions certify at **0 of 215
checkpoints across every depth**.

Secondary outcomes, all precommitted:

- **Verdict discordance**: the fraction of (state, row) pairs where
  `curvature/verdict_code_gradient` differs between T=256 and T=2048.
- **Sharpening shape**: R(T) = vhv_gradient(A4)/vhv_gradient(A1) at each T, and
  the ratio R(2048)/R(256) per seed.
- **The window step**: Δ across each rung of the ladder separately —
  256→512 (length only, windowing inert), 512→1024 (windowing onset),
  1024→2048 (full windowing).
- `gHg`, `‖∇L‖`, `dhd` and `dhd/‖δ‖² = vhv_update`, reported for continuity with
  I0005 and flagged uncertified where the update verdict does not pass — which,
  per I0001, is everywhere.
- **Between-row spread** of vhv_gradient at each T, which is σ_row and is a free
  deliverable for E06 (see "Sequencing").
- **The seed floor for curvature at T=2048**, which nobody has, from the 5 seeds
  at A4.
- `e_sym_gradient`, `e_lin_gradient`, `fd_cos_gradient`, `curv_snr_gradient` and
  the conclusiveness reasons at every T — the instrument-side view.
- Native bf16 arm at every cell, recorded, never quoted (v4 §3).

### Decision rule, precommitted

Let **m = median over the 20 (seed, anchor) cells of |Δ|**, and let the seed
floor be I0001's 29% sd-relative for gHg (ln 1.29 = 0.255), the closest
established channel.

1. **The record stands.** m ≤ 0.255 (one seed sd) **and** verdict discordance
   ≤ 5% **and** the paired test on log R(2048)/log R(256) does not reject at
   α = 0.05. Consequence: `DATASET.md` caveat 4 is rewritten as a measurement
   with a number, E06 proceeds at T=256 unchanged, and v4 item 7 may size the
   Gram probe in rows.
2. **The record needs a caveat.** 0.255 < m ≤ 0.507 (two seed sd). Curvature
   results remain usable, but every quoted magnitude carries a stated
   T-sensitivity, and E06 records the T-sensitivity beside its chosen m*.
3. **The record must be re-cut.** m > 0.507, **or** verdict discordance > 20%,
   **or** the shape test rejects. Consequence: T becomes a first-class probe
   parameter; E06's sizing grid is re-run at the T chosen here; v4 item 7
   provides the Gram probe at T=2048 only, not at both; and every published
   curvature number is relabelled as a T=256 quantity in `DATASET.md`,
   I0001's table and I0005's trajectory.

The rule fires on the **d12 Stage 1 data alone**. Stage 2 is descriptive.

### Precommitted predictions, so the design can be wrong

- **Verdict pass rates should RISE with T.** More target tokens means more
  signal against the same arithmetic floor, and the suite's conclusiveness rules
  are SNR-gated (`hvp_min_snr = 10`, `hvp_snr_safety = 5`). If pass rates *fall*
  with T, the tolerance thresholds (`e_sym = e_lin = 1e-4`, `fd_cos = 0.999`,
  `tolerance_version 1`) are mis-cut for long sequences, and that is an
  instrument finding rather than a landscape one.
- **The largest single rung should be 512→1024**, where windowing first bites.
  If the largest rung is instead 256→512 — where the architecture is
  windowing-inert — then the effect is about context length and not about
  attention structure, and the "windowing makes T special" argument above is
  wrong in its emphasis.
- **`e_sym_gradient` should fall with T.** It is I0004's decisive channel and it
  is dominated by arithmetic noise relative to signal.

## Power, honestly, against the known floors

**Unit of inference.** The 8 rows within a (seed, anchor) cell are paired
replicates of the same operator, not independent samples of the estimand.
Treating 160 row-level pairs as the sample size would be pseudo-replication. The
test uses the **20 (seed, anchor) cell means** as the unit and reports the
row spread separately.

- **Primary.** A paired test at n = 20 detects 0.661 within-pair standard
  deviations at α = 0.05 two-sided, 80% power. Taking sd(Δ) ≤ σ_log(gHg) ≈ 0.25
  as a conservative bound — the paired contrast shares θ and shares rows, so it
  cannot be noisier than the level itself — the design detects **an 18% median
  shift** in vhv_gradient between T=256 and T=2048. That is comfortably below
  the 29% seed floor, so the design can support the *negative* conclusion, which
  is the one that would vindicate the existing record. sd(Δ) is itself a
  deliverable, so the power statement is re-derivable after the fact.
- **Verdict discordance.** 160 paired cells. By the rule of three, observing
  zero discordant pairs bounds the discordance rate below **1.9%** at 95%
  confidence.
- **Shape.** The test on log R(2048) − log R(256) has n = 5 (seeds), detecting
  1.66 sd. I0005 measured plateau levels differing 15–19% between seeds, so
  taking sd(log R) ≈ 0.17 gives a **detectable change of ~33% in the sharpening
  ratio** — enough to distinguish 15x from 20x or 11x, not enough to distinguish
  15x from 17x. Say so when reporting: a subtle shape change is not detectable
  with five seeds, and this design does not pretend otherwise.
- **What no floor covers.** I0001's floor is initialization-only (corrected
  2026-08-25 after I0008) and this design changes neither data order nor
  batching, so it is the right floor. But it was measured *on the T=256 probe*.
  If the T=2048 seed floor comes back much wider, the 0.255 threshold in the
  decision rule was cut against the wrong yardstick — which is why the T=2048
  seed floor is measured here and the rule is re-derivable.

## Sequencing with E06 — the two designs together determine the probe

E06 sizes the probe in **rows** at fixed T=256. This design sizes it in
**sequence length** at fixed rows. Run independently, each can converge on a
point of the wrong slice, and E06's own open question 1 says exactly this:
"Is a short-context spectrum the quantity we want, or should the sizing axis be
*tokens* (m·T) rather than rows?"

**Precommitted ordering.**

1. **E09 Stage 0 runs first**, before E06 freezes. It is 1 seed (s7), 2 anchors
   (A1, A4), 4 rows, the full 4-rung ladder, plus one iso-token cell at A4.
   About **0.5 GPU-hours and $2**.
2. E06 then proceeds under one of three outcomes:
   - **m ≤ 0.255 and Bank I shows rows are the right axis** → E06 proceeds
     exactly as drafted.
   - **m ≤ 0.255 but Bank I shows tokens are the right axis** → E06 proceeds at
     T=256, restating its grid as {1024, 4096, 16384} tokens rather than
     {4, 16, 64} rows. Same cells, honest label.
   - **m > 0.507** → E06's grid is re-cut at the T chosen here, and its cost
     model changes materially: a T=2048 HVP at m=64 is 8.8x a T=256 one, taking
     E06 Stage 1 from ~2.5 to roughly 20 GPU-hours. Its **57.3 GB Lanczos basis
     contract must also be rechecked**, because at T=2048 the math-SDPA
     double-backward working set is no longer negligible beside the basis, and
     the k=50 cap may have to fall — which E06 itself notes has no answer except
     a different algorithm and a new id.
3. E09 Stages 1 and 2 run next, as the full record.
4. E06 Stage 2 (its confirmation bank and seed floor) runs last, at the settled
   probe.

One free hand-off in the other direction: this design's **between-row spread at
T=256 on 8 rows is σ_row at m=1**, from which E06's `CV(m) ≈ σ_row/√m` follows
under exchangeability. E06 can pre-check its own nested-scaling prediction
before spending its sizing grid, and if σ_row is heavy-tailed here, E06 learns
that its rows are not exchangeable for about $2.

## Cost

No training runs. The unit is one acceptance suite at (θ, bank, T).

**Cost per suite is an outcome, not an input.** Two bounds bracket it. The
FLOP-bound estimate: a suite is roughly 85 forward+backward equivalents across
its three directions and its epsilon sweeps, so at d12 and one row it is
0.65 s at T=256 and ~9 s at T=2048 (with a 1.6x derating for materialized
math-SDPA attention at long T). The latency-bound estimate: at one row of 256
tokens the GPU is nowhere near saturated and the suite is dominated by ~hundreds
of small sequential kernel launches, which is why the recorded per-deep-checkpoint
sparse overhead is 16.9 s at d12 for both arms plus update effectiveness plus
Muon replay calibration. Planning figure: **5 s per row-suite at T=256, rising
to ~12 s at T=2048** — i.e. cost grows far more slowly than FLOPs until the GPU
saturates, which is the pleasant surprise in this design.

| block | cells | est. |
|---|---:|---:|
| Stage 0 screen (1 seed, A1+A4, 4 rows, 4 lengths, 1 iso-token cell) | 36 | 0.5 h |
| Stage 1 ladder: 5 seeds × 4 anchors × 8 rows × 4 lengths | 640 | 1.4 h |
| Stage 1 Bank I: 5 seeds × 2 anchors × 4 cuts | 40 | 0.7 h |
| Stage 1 Bank L: 20 states × (4 rows + row 0 alone) | 100 | 0.2 h |
| Stage 1 θ₀ stress: 1 seed × 4 rows × 4 lengths | 16 | 0.1 h |
| Stage 1 Bank R: 8 platform reruns at T=2048 | 8 | 0.1 h |
| Stage 2: d14 and d16, 1 seed each, 4 anchors × 8 rows × 4 lengths | 512 | 1.0 h |
| checkpoint loading + shadow construction, 28 model states | — | 0.6 h |

**About 4.5 GPU-hours of compute at the planning figure; budget 8 GPU-hours and
~$30 at $3.29/h against the latency-bound case, ~6–9 h of pod wall-clock, and
zero training GPU time.** Stage 0 alone, which is the piece that gates E06, is
about $2.

## What this design does not answer

- **Whether even T=2048 is the right function.** The training loss is a mean
  over 524,288 tokens — 256 rows of 2,048. The largest bank here is 16,384
  tokens, **3.1% of one batch**. Sequence length is one axis of the probe and
  row count is the other; this design fixes one and varies the other, and E06
  does the reverse. Neither reaches the training loss, and the pair of them
  together still does not.
- **λ_max, the top-k subspace, or any spectral quantity.** No Lanczos here. This
  design measures the existing scalar directional channels only.
- **Whether curvature responds to anything.** Like E06, this is calibration: it
  says what the number describes, not whether it moves under treatment.
- **The trajectory between anchors.** Only four usable saved checkpoints per run
  exist (progress 0.25, 0.50, 0.75, 1.00), all but the first inside the
  warmdown. The design tests whether the *ratio* across the warmdown has the
  same size at each T; it cannot locate the change point, and it cannot
  reproduce I0005's 30-point trajectory at any length but 256.
- **Length versus packing composition, in general.** The truncation ladder holds
  document composition fixed by construction, which is what makes T identifiable
  — but it means the design measures "the same rows, seen shorter", not "how a
  natively-packed short probe differs". Bank L quantifies that at one point and
  nowhere else.
- **Whether attention windowing matters for training.** Only for the probe. A
  window-pattern intervention is an architecture experiment (E04's territory),
  and this design's only contribution to it is to establish that such an
  intervention is *unmeasurable* on a T ≤ 512 probe.
- **Whether the acceptance tolerances are correctly cut at any T.** The design
  can flag that they are wrong; re-cutting them is a separate instrument change
  with its own `tolerance_version`, and doing it here would make the verdicts
  incomparable with all 2.69M existing records.
- **Anything cross-depth beyond within-run ratios at matched progress.** Stage 2
  compares Δ and R across depths at anchors of progress 0.25–1.00, all above
  I0006's 0.159 threshold, on a byte-identical probe. That is about as safe as a
  cross-depth statement gets on this dataset, and it is still n=1 per depth.
- **fp32 correctness.** `_SHADOW_DTYPES` supports fp32 only, because the forward
  pins rotary and logits to fp32, so there is no higher-precision reference at
  any T. fp32 correctness is assumed, backed by the acceptance suite — which is
  itself one of the things under test here. Say so wherever these numbers are
  quoted.

## Instrument dependencies

**None new.** Like E06, this design deliberately consumes only frozen-instrument
output, at current versions: `hvp` (`telemetry.py:1102`), `hvp_acceptance`
(`:1307`), `build_shadow_model` (`:1988`), `shadow_precision` (`:1963`),
`save_probe`/`load_probe` (`:1841`/`:1849`), `collect_probe_rows` (`:2306`),
`canonical_named_parameters`, the `Record` schema (v3) and `TelemetryWriter`.
The truncation, bank construction and driver are new **offline** code that never
touches the training path, so the instrument stays frozen and the
non-perturbation contract is not in play.

Records are written at tier `offline` under a new `seqlen/*` namespace, which
adds no required columns; `probe_id`, `parent_probe_id`, `probe_T`,
`checkpoint_id`, `acceptance_arm`, `estimator_id` and `dtype` are populated
voluntarily. Truncated banks are materialized with `save_probe()` so each
carries its own content-derived `probe_id`, with the parent id and truncation
length recorded — a truncated bank is a new artifact and must not inherit its
parent's identity. Per-direction verdicts are written as
`curvature/verdict_code_{random,gradient,update}` under the existing
`tolerance_version = 1`, so they are directly comparable with the 2.69M records
already collected.

**E09 produces the input to telemetry v4 item 7**, whose text currently defers
this exact decision: "Provide the Gram probe at the training sequence length
(T=2048), or maintain banks at both lengths and record which was used." It also
supplies the T-axis of the cost model that v5's deferred "online Lanczos and
spectral density" item needs.

## Open questions before freezing

1. **Is truncation the right nesting?** It is the only nesting that makes T
   identifiable, but it means the design never sees a natively packed long
   probe versus a natively packed short one — which is what a future instrument
   would actually build. Bank L covers one point. Should a full native-repacking
   arm be added at each T, at roughly 4x the bank-construction cost and with the
   loss of exact pairing?
2. **Four anchors is thin for a shape claim.** A fresh d12 run with
   `--telemetry-checkpoints 12` costs about one d12 run (~0.72 GPU-h, ~$2.40)
   and would give 13 anchors instead of 5, turning the shape test from four
   points into a curve. That is cheap enough that it may be the better use of
   this design's budget than Stage 2. Should Stage 2 (d14/d16) be dropped in
   favour of it?
3. **Mean or per-token?** A truncated row has fewer valid targets and possibly a
   different BOS structure, and the mean-over-valid-targets reduction makes
   curvature a per-token quantity while the gradient norm is not. Does the
   decision rule need a per-token normalization, and if the answer changes with
   the normalization, which one is the claim about?
4. **Is 8 rows enough for σ_row?** The CV of a CV at 8 samples carries ~27%
   relative uncertainty (E06's own note). Sixteen rows are available in
   `probe_val` at no extra data cost, at double the Stage 1 ladder time.
5. **Which anchor is primary?** A4 is chosen because architecture studies
   compare final models and because E06 chose the equivalent. I0005 puts the
   interesting dynamics in the warmdown. If m differs between A2 and A4, the
   design has a decision that depends on regime and no rule for that case —
   the same gap E06 has, and the two should resolve it the same way.
6. **What if the native bf16 arm behaves differently in T from the shadow arm?**
   I0002 measured native curvature as accurate to ~0.3% and unbiased but
   uncertified at T=256. If the native-versus-shadow gap widens with T, the
   training surface and the measurement surface diverge with sequence length,
   which would be a genuine finding — but this design records it without a
   precommitted rule for it, and probably should have one.
7. **If outcome 3 fires, is caveat 4 a caveat or a correction?** `DATASET.md`
   caveat 4 is currently a warning about scope. If m > 0.507, I0001's spread
   table and I0005's trajectory are measurements of a function nobody trains on,
   and the honest handling may be a correction block in each conclusion — as
   I0001 already carries for I0008 — rather than a relabelling. That should be
   agreed before the first run, not after the result.
