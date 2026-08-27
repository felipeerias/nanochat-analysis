# nanochat training-dynamics telemetry — d12–d16 dataset

Densely instrumented pretraining runs of [nanochat](https://github.com/karpathy/nanochat),
collected 2026-08-24/25 on a single H100 SXM. Produced by the `telemetry`
branch of https://github.com/felipeerias/nanochat (instrument:
`nanochat/telemetry.py`). Operations live in this repository, in
`operations/`. **Measure-first dataset**: the
instrument is general-purpose (~90 metric families), not built around a
particular hypothesis.

## Inventory

| run | schema | depth | width | seed | steps | deep ckpts | rows |
|---|---|---|---|---|---|---|---|
| `d12-s7` … `d12-s11` (5 runs) | 3 | 12 | 768 | 7,8,9,10,11 | 2520 | 30 | 292k each |
| `d14-s7` | 3 | 14 | 896 | 7 | 3759 | 32 | 403k |
| `d16-s7` | 3 | 16 | 1024 | 7 | 5376 | 33 | 542k |
| `d12-iter` | **1** | 12 | 768 | 7 | 2520 | 3 | 281k |

2.69M records total. `d12-iter` is the earlier shakedown run kept for
history: **schema v1, hd64, no shadow arm, no lineage checkpoints** — do not
pool it with the v3 runs.

All v3 runs use upstream defaults (`head_dim=128`, so `n_head = depth/2`),
`aspect_ratio=64` (width = 64·depth), and nanochat's automatic
depth-dependent derivations for batch size, LR, weight decay, and horizon
(~12:1 tokens:params). The five d12 runs differ **only** in seed and exist to
characterize seed-to-seed variance.

**Important (established 2026-08-25, investigation I0008): the seed changes
initialization only.** `nanochat/dataloader.py` contains no RNG; it walks the
parquet files in index order into a deterministic packer, and `--seed` seeds
initialization. Verified: `batch/bos_count`, `batch/valid_targets` and
`batch/mean_segment_length` are bitwise identical at all 2,520 steps across
the five d12 seeds, and the three frozen probes have identical ids in all
five runs. **The dataset therefore contains exactly one data ordering**, and
the seed-variance reference measures initialization variance alone. Any
experiment that changes data order or batching faces a different, larger
noise floor — I0008 estimates the batch-selection effect on step-to-step loss
at about 10x the initialization effect.

## Segment layout

```
<run_id>-s0-<32hex>/           # the "-s0-" is the segment start step
  provenance.json              # complete run identity (see below)
  manifest.json                # exact sweep-manifest bytes used (v3 runs)
  continuous/ periodic/ sparse/ offline/    # chunk_NNNNNN.parquet
  probe_val.pt probe_train_stream.pt probe_short.pt   # frozen probes
  checkpoints/inventory.json   # hashes/sizes of lineage checkpoints
  checkpoints/*.pt             # NOT in this local copy — see Caveats
```

Tiers: **continuous** = every step (loss, timing, memory, learned scalars);
**periodic** = every `ceil(N/25)` steps (~25 per run: gradient/parameter
norms, sketches, gradient-noise scale, Muon stage geometry, probe forwards,
attention statistics, optimizer state); **sparse** = the deep checkpoints
(curvature/HVP acceptance, update effectiveness, Muon reference calibration);
**offline** = written once at close (overhead totals).

Deep checkpoints follow a Pythia-shaped schedule with the same shape in
normalized progress at every depth: geometric prefix `{0,1,2,4,8,…}`, a
~20-point uniform tail, plus landmarks at the recipe's transitions (LR warmup
end = 40, Muon momentum ramp end = 400, warmdown start). The concrete step
set is in `provenance.telemetry_deep_steps`, exactly recomputable from
`telemetry_deep_steps`/`deep_schedule_landmarks`.

## Record schema

One row per measurement. Key columns: `metric`, `tier`, `phase`, `step`,
`tokens_seen`, `normalized_progress`, `value_scalar` **or** `value_vector`,
`is_defined`, `undefined_reason`, `param_role`, `parameter_name`, `layer`,
`head`, `acceptance_arm`, `acceptance_status`, `estimator_id`, `dtype`,
`backend`, `aggregation`, `run_id`, `schema_version`,
`telemetry_config_hash`, `parameter_schema_hash`.

**Conventions that matter:**
- `pre_update` rows carry step *s*; `post_update` rows carry *s+1*. Deep
  checkpoint at update index *s* therefore appears at `step == s+1` in sparse.
- `is_defined == False` means *honestly undefined*, with a reason — never
  silently zero. **Always filter explicitly.**
- `normalized_progress` is the cross-depth x-axis. Step numbers are not
  comparable across depths.
- Sketches are comparable only within a matching `parameter_schema_hash`.

## Loading

```python
from loader.telemetry_load import DEFAULT_DATA_ROOT, load_segment

run = load_segment(DEFAULT_DATA_ROOT, SEGMENT)
df = run["tiers"]["sparse"]
```

A loader that encodes the filtering discipline (defined-row handling, per-arm
selection, verdict-aware `certified()`) lives at
`loader/telemetry_load.py`. `NANOCHAT_REPO` and
`NANOCHAT_TELEMETRY_DATA_ROOT` override its default sibling layout. Reading the
parquet raw with pyarrow works equally well; several independent analyses have
done so and agreed with the loader.

**Findings from this dataset are in `investigations/`**, one folder
per investigation, each with a frozen protocol, two independent blind analyses,
and a conclusion. Read those before starting new work — several correct
statements made earlier in this card.

## Two acceptance arms — read this before using curvature data

Curvature is measured twice at each deep checkpoint:

- `acceptance_arm="native"` — the **bf16 training surface** as the optimizer
  experiences it.
- `acceptance_arm="shadow_fp32"` — a disposable IEEE-fp32 upcast copy of the
  model at θ_s (TF32 off, math-SDPA, rotary rebuilt), measured along the
  *actual* applied update (upcast endpoints). `estimator_id =
  "hvp-shadow-fp32-ieee-v1"`.

Verdicts are **per arm, never merged**: `curvature/native_verdict_code` and
`curvature/shadow_verdict_code` (0=passed, 1=inconclusive, 2=failed).

**Checkpoint-level verdicts are the worst across the three probe directions
(random / gradient / update) and are therefore pessimistic.** Observed:
native = `failed` everywhere (bf16 cannot meet fp32-era symmetry/linearity
thresholds — this is a property of the training arithmetic, not an instrument
defect, and is itself data); shadow = mostly `inconclusive`, driven by the
random direction's near-null curvature failing its noise floor. The
per-direction records are where usable data lives: e.g. in `d12-s7` the
shadow **gradient** direction `passed` at 26 of 30 checkpoints. Use
`curvature/verdict_code_{random,gradient,update}` to select.

## Verification status

Every segment passed `../nanochat/runs/verify_telemetry_run.py` on the pod before its pod
stopped (structure, chunk contiguity, per-arm schema completeness at every
deep step, provenance/manifest identity, probe re-hashing, lineage checkpoint
hashes, cadence coverage). A 7-mode tamper suite proves the verifier detects
edited manifests, missing/corrupt checkpoints, and stripped columns.

Re-running the verifier on **this local copy will fail** the lineage-hash
check, because `checkpoints/*.pt` were deliberately excluded from the
transfer (70 GB → 999 MB). That is expected, not corruption.

## Caveats (please carry these into any analysis)

These numbers are referenced throughout the repository. Keep existing numbers
stable and append new caveats rather than renumbering this list.

1. **This is a size ray, not a depth sweep.** Depth co-varies with width,
   head count, batch size, LR, weight decay, and horizon by design (nanochat's
   own scaling rules). Conclusions are about *the nanochat recipe at scale*,
   never "depth causes X".
2. **n=3 depths.** Trend-fitting across d12/d14/d16 is weak; the five d12
   seeds exist so that any cross-depth claim can be checked against the
   seed-noise floor first.
3. **Warmups are absolute, not proportional**: 40-step LR warmup and a
   hard-coded 400-step Muon momentum ramp are ~16% of d12 but ~7% of d16.
   Early-training comparisons across depths are confounded by this.
4. **All curvature is measured on a SINGLE 256-token sequence.** The `short`
   probe holds four rows, but the HVP path takes only the first
   (`telemetry.py:2484`, `sx[:1]`), so every Hessian-vector product, curvature
   value, acceptance verdict and eta* in this dataset describes the loss
   surface of one 256-token sequence — not the 2048-token training loss, which
   is a different function. This is the likely explanation for curvature's wide
   seed spread (25-29%), and it means curvature results are local in a stronger
   sense than "evaluated at a checkpoint": they are local in the data too.
5. **Muon updates decohere from any reference**: the compiled bf16 optimizer
   applies updates ~3–10% (per-matrix relative L2) away from the eager
   reference decomposition, because Newton-Schulz amplifies rounding
   placement. `muon/replay_update_relerr` records this per matrix per deep
   checkpoint; Muon stage metrics should be read as reference-frame
   quantities with that recorded error bar.
6. **Native bf16 curvature is uncertified everywhere** (see above). Do not
   quote native curvature numbers as measurements without the shadow arm or
   an explicit statement that they are uncertified.
7. **Probes are identical across all seven schema-v3 runs** (verified: the
   same three probe ids), because probe selection inherits the deterministic
   loader. Probe-derived spread across the five d12 seeds is therefore
   *not* contaminated by probe sampling — an earlier version of this card said
   otherwise and was wrong. A run with a different data ordering would draw
   different probes.
8. **Compiled GPU training is not bit-reproducible**: the embedding backward
   contains an atomic-accumulation race, so identical configs can differ by
   ~1 ulp in optimizer moments. Quantified by the A/B control gate.
9. **Batch construction is under-instrumented** relative to the other areas:
   gradient-noise scale is estimated on a device batch (below the logical
   batch that drives the update), there is no loader sidecar (no document
   identity/domain/crop metadata), and batch size never varies within this
   dataset. Treat noise-scale numbers as descriptive, not as critical-batch
   claims.
10. **Multiple comparisons**: with ~90 metric families and hundreds of
   channels, declare confirmatory tests before looking. Exploratory findings
   should be labeled as such and re-tested on new runs.

## Reproducing / extending

```bash
bash operations/telemetry_pod_setup.sh                       # pod bootstrap
bash operations/telemetry_run.sh \
     operations/manifests/sweep-d12-d16-v1.json d12-s7
```

Manifests live in `operations/manifests/` and pin the training code with
`nanochat_commit`; the runner refuses to start against a different commit.
Every segment embeds its own copy, so the external file may change later.
Official runs accept no extra training arguments: an official run is exactly
the resolved flat manifest row. Full model+optimizer
lineage checkpoints (θ₀ + 3 interiors + the final triplet, hash-inventoried)
remain on the Runpod network volume `nanochat_experiment` (AP-JP-1, ~70 GB)
for offline-tier work.

Instrument design and rationale: `../nanochat/docs/telemetry-spec.md`;
research framing: `telemetry-plan-v2.md`; engineering history is narrated honestly in the
branch's commit messages.
