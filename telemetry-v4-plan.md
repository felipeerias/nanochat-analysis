# Telemetry v4 — instrument plan

What the instrument should record next, and why. **This document is about the
instrument only.** Experiment designs live in their own documents; the
instrument is justified by the class of questions it makes answerable, not by
any single campaign.

Agreed 2026-08-25 between Claude Code and Codex over three rounds of debate,
after eight completed investigations on the d12–d16 dataset.

The governing principle from that debate: **make identity, selection, packing
and policy decisions reconstructible first.** More curvature telemetry cannot
rescue an experiment whose treatment and sampling process are not recoverable.

The cut line: *can we identify the treatment, reconstruct its realization,
attach the outcome, and verify the join?* Everything else waits for v5.

## Capability gaps this closes

| gap | consequence today | what unblocks it |
|---|---|---|
| No data-group or document identity | no data-side question is answerable at all (I0008) | the lineage sidecar |
| One data ordering, one seed axis | data-order variance is unmeasurable; "would emphasizing X have helped" is unanswerable in principle | separate seeds |
| No join between a batch and its outcome | cannot attribute any measurement to the data that produced it | `batch_id` on loss and gradient events |
| Certification applied at the wrong granularity | analyses needing certified HVP quantities had to select direction verdicts by hand | loader direction-certification fix (completed 2026-08-27) |
| No example-level interference measurement | interference and alignment questions have no instrument | sampled row-gradient Gram |
| Spectral quantities absent | landscape geometry has no λ_max, no subspace structure | probe bank plus model-only checkpoints for offline work |
| Noise estimator invalid | ≥8x optimistic; unusable for sizing anything (I0008) | independent-draw estimator, old one renamed |

## 1. What v4 must contain

1. **Separate the seeds.** Split `--seed` into `init_seed` and `data_seed`
   (add `policy_seed` only if a sampler is stochastic; the packer stays
   deterministic given its input stream). Record the values and derive an
   experimental design matrix automatically, so any "differs only in X" claim
   is checkable rather than assumed — the failure that produced the wrong
   statement in the first dataset card.

2. **One `batch_lineage` sidecar table.** One row per logical batch, nested
   row and segment structs: `batch_id`, update, microbatch and row positions,
   ordered document locators, group id and taxonomy version, original length,
   used span, document offset, crop flag, ordered segment spans, intended
   mixture, realized composition by documents and by trained tokens, and where
   a policy exists the observation/decision/applied steps, the chosen `q`, and
   the policy version. Relational normalization is deferred.

3. **`batch_id` on the logical-batch loss and sampled gradient events.** The
   cross-model join. The existing aggregate `batch/*` fields stay as redundant
   integrity checks.

4. **Probe partitions as immutable, precommitted artifacts.** Controller,
   monitoring and sealed evaluation indexes, produced by deterministic
   partition of stable locators and materialized as index files. The manifest
   binds their hashes, the dataset and tokenizer identity, the partitioning
   algorithm and salt, and each partition's allowed purpose. A training
   process never receives the sealed index. (This is an instrument capability;
   whether a given experiment uses a controller is not the instrument's
   concern.)

5. **Fix direction certification in the loader**, without a new schema column.
   Completed 2026-08-27 in analysis commit `8950b04`: gradient quantities use
   the gradient verdict, update curvature uses the update verdict, and `p1`
   and the realized loss change use no HVP verdict. Unknown future deep metrics
   fail closed until their dependency is declared. Definedness remains a
   separate, explicit filter.

6. **Sidecar verifier.** Four layers — physical (atomic chunks, per-chunk
   SHA-256, ordered inventory bound into provenance), structural (counts,
   unique keys, ordered non-overlapping spans covering each row, BOS
   positions, mixture sums, decision step precedes applied step), cross-model
   (every logical-batch loss has exactly one sidecar batch; aggregate `batch/*`
   recomputes from nested segments), and a sampled semantic audit (reread the
   source documents, retokenize with the frozen tokenizer, reconstruct the
   rows, compare hashes). Extend the tamper suite with deletion, duplication,
   row reorder, altered group id, altered span or crop, mismatched decision
   id, wrong batch hash, and a chunk swapped between runs.

7. **Geometry recording.** A frozen probe bank of at least 64 rows — **and
   the sequence length is not free.** The draft E04 design found that at
   T=256 no position reaches the sliding-attention boundary (the short window
   is 512 tokens), so short-window and long-window layers behave identically
   on the probe and any attention-pattern intervention is inert on it. A probe
   bank at T=256 would make that class of experiment unmeasurable. Provide the
   Gram probe at the training sequence length (T=2048), or maintain banks at
   both lengths and record which was used. Also:
   model-only checkpoints at about six preregistered anchors; a sampled
   row-gradient Gram (8–16 rows: exact per-row norms, sketched pairwise inner
   products, stored normalized Gram, trace, effective rank, negative-pair
   fraction, eigenvalue spectrum, occasional exact calibration); and populate
   `shape`, which is nearly free.

8. **Independent noise estimation, gated on use.** Add independent draws at
   about five anchors only when a protocol quotes noise scale or sizes an
   estimator from it; otherwise stop quoting `b_noise` and defer. Retain the
   existing statistic under a name describing what it measures (within-buffer
   clustered device batch), since it remains valid for that narrower estimand.

Sections 1-8 are the agreed list. Section 6 holds four smaller items added
afterwards; nothing above it changed.

### Non-perturbation constraint

The sidecar must never read CUDA tensors. `batch_id` is a loader-issued key.
Token binding uses **one hash per microbatch**, computed on the contiguous CPU
buffer immediately before the existing host-to-device copy — not per row,
which costs 20.6 MB of incompressible data per run against 0.65 MB for no
material gain in corruption detection.

Host-side work still changes, so the bitwise A/B gate stays mandatory for every
new loader mode, with step-time and loader-stall measurements. If hashing
causes stalls, hash deterministic audit samples instead (first and last
batches, measurement checkpoints, about 1 in 64 otherwise); joins and
structure remain exactly verified and only the token-to-lineage binding of
unsampled batches weakens.

### Size

About 20–50 MB compressed per d12 run. If a pilot exceeds 50–64 MB, drop in
order: per-row hashes, repeated policy fields, string encoding, derivable
fields, oversized integer types. Never drop ordered document locators, segment
spans, group ids, original and used lengths, or crop indicators — those are
the scientific content. Confirm with a 100–200 step shakedown, since segment
density is the dominant uncertainty.

## 2. Deferred to v5

First-class `direction` column; derived-metric and metric-to-certifier
registries; eta*/Rayleigh storage deduplication; per-family cadence framework;
sidecar normalization; exact loader and controller replay; per-item inclusion
propensities; per-segment losses every step; a comprehensive assertion
framework (a small preflight suffices now); online Lanczos and spectral
density; group HVP and bracket instrumentation; Muon geometry on the deep
schedule.

Three items drafted for section 6 were cut to here after review, because each
needs several missing pieces at once rather than one added field:

- **The two-by-two projected Hessian** on the probe loss (T01 equation 26),
  which is what would connect the certified probe-gradient direction to the
  update direction. Try it offline first on the existing lineage checkpoints.
  That path is not guaranteed: no update vector is stored, and interior
  checkpoints omit loader state, so exact batch and update reconstruction has
  to certify before any number from it means anything. Record it in-run only
  if the offline attempt works.
- **A mechanism for the warmdown bend.** `r_s` is fitted, not derived. Getting
  a mechanism needs projected geometry, valid independent noise and the
  projection weights together, not a denser measurement cadence.
- **Production-versus-reference mask comparison**, per item 1 in section 6.

**Official runs do not resume.** A failed run restarts from zero at about
$2.30. Exact buffer and controller restoration is poor leverage.

## 3. Instrument facts that constrain any experiment using it

These are properties of the measurement, not of any campaign, and analyses
must respect them.

- **ηλ_max is not a valid stability statistic for Muon.** Matrix-specific
  effective learning rates, momentum, nonlinear polar orthogonalization,
  factored scaling and cautious decay mean no fixed scalar preconditioner
  makes it meaningful. The Hessian spectrum is a landscape descriptor only.
- **Three distinct gradient estimands** must not be conflated: the row
  gradient (the realized causal unit), the document-target gradient in mixed
  context (well defined but conditional on preceding packed context), and the
  document-pure gradient (intrinsic, only in document-pure rows).
- **Averaging λ_max across small probes is wrong** — the mean of λ_max is not
  λ_max of the mean Hessian. Average the HVP across probe chunks inside each
  Lanczos iteration instead.
- **Sketches are keyed by parameter schema** and cannot be compared across
  architectures. Scalar structure derived within each architecture can be:
  row-gradient cosines, Gram spectra, effective rank, negative-interference
  fraction, role norm shares. The common space is function space, not a
  parameter projection.
- **Certified curvature exists along the gradient direction only**, at every
  depth measured (I0001, I0004).
- **The HVP probe is one row of 256 tokens** (`telemetry.py:2484` takes
  `sx[:1]` from a four-row probe). Every curvature result to date describes a
  256-token loss surface, not the 2048-token training loss. Enlarging it is
  the single cheapest variance reduction available, and E06 exists to size it
  before any spectral work.
- **Native bf16 curvature values are accurate to about 0.3% and unbiased, but
  uncertified**; update-effectiveness records degrade late through
  catastrophic cancellation (I0002).

## 4. Verification and canaries

New canaries must be **calibrated by fault injection** — wrong scaling, stale
θ, a dropped parameter block, permuted flattening, asymmetric perturbation —
and kept only if they detect those mutations. On that criterion the
random-direction and native finite-difference suites are dropped from ordinary
runs: across seven runs and three depths the random direction certified 0 of
215 times, and native finite differences at bf16 add no independent
sensitivity beyond the paired native-versus-shadow comparison.

Run a full suite once when dtype, backend, model family or loss implementation
changes — keyed off configuration, not model name.

## 5. Requirements discovered by the experiment drafts

Nine designs have now been drafted against this plan. Reading across them
surfaced one cross-cutting operations requirement.

**Runner status, updated 2026-08-27.** A manifest row is now flat and contains
the arguments for that run. The runner discovers valid names and types from
the pinned checkout's `base_train` and telemetry parsers, instead of carrying
a fixed allowlist or a nested `recipe` block. It rejects runner-owned flags,
passes the resolved argument vector directly, records the exact manifest, and
asks the verifier to assert every resolved input in `provenance.user_config`.
The verifier also recomputes model width and head count from the recorded
geometry inputs.

This removes the generic runner block for E02 and E03, and for E08's existing
trainer overrides. Each campaign still needs a new immutable manifest. A knob
that does not yet exist in the pinned trainer still belongs on an experiment
branch; this is the remaining situation for E01, E04, E05, and E07.

Draft designs do not need a universal realized-treatment schema before their
questions are settled. Before a design freezes, it should decide whether its
primary contrast is adequately bound by recorded inputs and model config. If
not, add only the realized derived quantities that matter for that treatment:
for example optimizer-group learning rates, the effective warmdown landmarks,
the window pattern per layer, or the Muon execution mode. E02 found that
`--warmdown-ratio` moves both the learning-rate and Muon-momentum warmdowns,
so it is a concrete case where a realized schedule may be worth recording.

Other specific requirements from the drafts:

- **Spectral work is memory-bound to d12.** A 50-vector fp32 Lanczos basis is
  57.3 GB at d12, 80 GB at d14 and 107 GB at d16 (E06). This is a hard
  constraint, not a preference.
- **A reference-locked Muon execution mode** is needed for E05: apply the
  eager reference operation sequence as the actual update, at unchanged
  precision. This turns the reference decomposition from a measurement into an
  intervention and is the only way to separate divergence-from-reference from
  arithmetic precision.
- **The softcap constant is duplicated** between the model and
  `TelemetryConfig.softcap`, with `softcap_equiv_err` as a canary that will
  fire if they diverge (E04). Any experiment changing it must change both.
- **Probe banks need a declared sequence length**, recorded per bank, since
  E04 and E09 need different lengths and E06 sizes the row count separately.
## 6. Added after section 1 was agreed

Four items, from the theory notes and from reading the version machinery.
Reviewed with Codex, which refuted three larger items that were drafted here
first; those are recorded in section 2 as deferred.

1. **Record the cautious-mask margin.** For each entry, how close
   `u_ij * theta_ij` is to zero, as a distribution. The cautious-decay term
   is discontinuous at that boundary and has no derivative bound, so it is
   the one decoherence source T02 cannot bound
   (`exploratory/T02-muon-decoherence.md` §4, equation 9).
   `cautious_mask_fraction` counts masked entries but not their margin.
   Take the margin against the reference path: it needs no extra model
   evaluation, though the reduction is not free. Comparing the production
   mask against the reference is a separate and harder change, because it
   means exposing the mask from the compiled optimizer and passing the
   non-perturbation gate. Cross-arm disagreement is **not** available at all:
   the native and shadow acceptance arms do not execute Muon.

2. **Version the sketch projection separately from the schema.**
   `name_key` (`nanochat/telemetry.py:404`) takes `schema_version` and seeds
   `sketch_coefficients`, which produces the polynomial coefficients behind
   every sketch bin and sign. Any schema bump therefore silently changes the
   projection and makes new sketches unpoolable with old ones. Give the
   projection its own version, so the schema can bump when it needs to and
   sketch comparability is unaffected.

3. **Make the loader read the version stamp.** `telemetry_load.py` loads
   `schema_version` into its frames but never inspects it; it infers v1 from
   the absence of `acceptance_arm`. Column-presence inference cannot see a
   change that alters values while keeping every column name, which is
   exactly what re-seeding the sketches does. Refuse mixed versions by
   default. Raw sketch operations should also require a matching parameter
   hash, sketch seed and estimator id.

4. **Populate `shape`.** Already listed in item 7 and repeated here only
   because it is nearly free and keeps being deferred.
