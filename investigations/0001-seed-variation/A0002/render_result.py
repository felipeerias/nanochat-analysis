#!/usr/bin/env python3
"""Render result.md from the validated A0002 ranking artifacts."""

from pathlib import Path


HERE = Path(__file__).resolve().parent

PREAMBLE = r"""---
id: A0002
investigation: I0001
kind: result
status: submitted
analyst: Codex (GPT-5)
date: 2026-08-25
evidence: exploratory
supersedes: null
protocol: investigations/0001-seed-variation/README.md@2a460b5b156819edeb098578bd6704dae7affa5d
data: sweep-d12-d16-v1; d12-s7, d12-s8, d12-s9, d12-s10, d12-s11; continuous, periodic, sparse parquet only
profiles: five d12 profiles@0c44574fd2310f059fbacb24b941da2c8c47b79c
universe: tested 268/268 base tier/metric families and report 268/268; acceptance-arm splitting produces 366/366 reported series, covering 27,469,549 fully aligned scalar/vector comparison elements
saw: frozen I0001 README; analysis/README.md; telemetry-data/sweep/DATASET.md; analysis/loader/telemetry_load.py; the five d12 profile.md files; profile and sweep directory/file-name listings; git commit metadata for the permitted protocol/profiles/procedure/loader; and continuous/periodic/sparse parquet chunks for only the five named d12 segments. Did not read A0001, conclusion.md, TEMPLATE-result.md, other-run parquet/content, profile figures, provenance, manifests, logs, checkpoints, or offline data.
---

# A0002 — d12 seed variation

## Result

Seed sensitivity spans essentially zero for fixed controls to several hundred percent for zero-centered/sketch, noise, attention, curvature, and update-effect channels. The most useful trajectory headline is `loss/train_mean`: its median five-seed range is **0.004668 loss units, or 0.1557%**, while its worst aligned step is **0.420982, or 5.6627%**. The early median is **1.1423%** and the late median is **0.1407%**, so a single pooled number materially overstates late loss noise and understates early loss noise.

`probe/loss` is also tight by the declared rule: median absolute/relative ranges are **0.014268 / 0.4532%**, worst **0.148396 / 3.2306%**, with **1.0911% early** and **0.4281% late**. This is a combined trajectory-plus-probe-sampling floor, not pure trajectory variance.

At the other end, representative median relative ranges are **68.11%** for `noise/b_noise`, **93.90%** for `noise/signal_raw`, **110.59%** for probe-based attention mean distance, **59.23%** for passing shadow `curvature/eta_star`, **70.58%** for passing shadow `curvature/gHg`, **171.81%** for shadow `update/actual`, **407.73%** for shadow `update/normalized_residual`, and **378–382%** for gradient-sketch components. Those are unsuitable for detecting modest changes from one run per condition.

Of the 366 arm-specific series, 302 have a numerical five-seed relative spread and 64 are unavailable under the selection. All 99 native-arm series (90 curvature and nine update families) are shown only as uncertified raw diagnostics; 90 have a numerical common spread and nine lack a common defined comparison. Excluding the entire native arm leaves **212 numerical series** classified mechanically as **64 tight, 87 intermediate, and 61 noisy**, plus **55 unavailable** shadow direction-conditioned series. The tight count includes fixed controls and categorical/status families, so it is not a count of scientifically useful response variables.

## Method

- I explicitly retained only `is_defined == True` rows and matched the five runs at exact `normalized_progress` (rounded only to 12 decimal digits for a stable key; the schedules are identical).
- A comparison element is a scalar or one vector component with the same metric, phase, aggregation, layer/head/role/parameter, probe type, optimizer metadata, dtype/backend, and other stable semantic metadata. Per-seed probe hashes were mapped to `train_stream`, `val`, and `short`; the two long probes were identified from the permitted profiles/final `probe/loss` ordering, and all other non-null hashes are the short probe. No duplicate semantic keys or vector-length mismatches occurred.
- At each comparison element, absolute spread is `max(seed values) - min(seed values)`. Relative spread is absolute spread divided by `abs(median(seed values))`. For an exact zero median, `0/0` is recorded as zero and a positive range divided by zero as infinity.
- “Typical” is the median over all aligned channel-progress elements in a family. “Worst” is their maximum. Ranking is ascending by typical relative spread; unavailable families are placed last. The table also gives typical/worst absolute spreads because ratios near zero can be pathological.
- Early is `normalized_progress <= 400/2520 = 0.158730`, covering both the 40-step LR warmup and the full 400-step Muon momentum ramp. Late is everything after that boundary.
- Any metric with an `acceptance_arm` is split. Shadow curvature quantities with an identifiable direction use only steps where that seed's corresponding `curvature/verdict_code_{direction}` is zero. The shadow gradient direction passes at 26, 26, 25, 26, and 26 checkpoints (s7–s11); their common intersection is 25 points. Shadow random and update pass at zero checkpoints, so 55 such direction-conditioned series are unavailable and reported at the bottom rather than dropped.
- The protocol simultaneously requires native-arm spread and says native is uncertified everywhere. I therefore compute native spread from defined raw rows without pass conditioning, mark all 99 native-arm results `uncertified`, and exclude them from change-detection recommendations. Generic curvature diagnostics with no governing direction (`arith_eps` and checkpoint verdict codes) are reported unconditioned.

The exploratory utility bands were declared before the spread results were inspected: **tight** requires overall, early, and late typical relative spread at most 5% and worst at most 20%; **noisy** means typical spread at least 50%, non-finite typical spread, or at least half of element-wise ratios infinite; everything else is intermediate. These bands correspond roughly to whether a persistent 10%-scale effect, or even an order-one effect, could clear the observed five-seed range. They are not inferential hypothesis tests.

## Channels tight enough to detect a change

Representative numerical measurement families meeting the tight rule (native curvature and fixed/categorical controls excluded):

| family | typical relative | worst relative | early | late | interpretation |
|---|---:|---:|---:|---:|---|
| `loss/train_mean` | 0.1557% | 5.6627% | 1.1423% | 0.1407% | Best general trajectory channel; late training is especially tight. |
| `optim/muon_second_momentum_norm` | 0.2159% | 6.6457% | 0.6337% | 0.1758% | Tight optimizer-state magnitude. |
| `probe/loss` | 0.4532% | 3.2306% | 1.0911% | 0.4281% | Tight, but includes probe sampling variance. |
| `probe/logit_lse_mean` | 0.8955% | 2.1136% | 0.8948% | 0.8994% | Tight probe/logit summary; probe-sampling caveat applies. |
| `probe/per_row_loss` | 0.9936% | 7.7286% | 1.4019% | 0.9435% | Still tight after retaining individual probe rows. |
| `update/direction_norm [shadow_fp32]` | 1.6579% | 12.7347% | 4.4414% | 1.2516% | Useful update-magnitude channel; short-probe lineage flag applies. |
| `update/loss_before [shadow_fp32]` | 2.9967% | 8.4201% | 0.7720% | 3.6742% | Useful baseline probe loss at deep checkpoints. |
| `update/loss_after [shadow_fp32]` | 3.6151% | 9.1161% | 2.1486% | 3.9149% | Useful for persistent effects larger than roughly 10%. |

`muon/data_norm` and `muon/u_final_norm_observed` are even tighter (typical relative range about **0.0000159%**), but they are nearly normalization invariants; I treat them as excellent instrument/control checks rather than broad training-response channels.

## Channels too noisy to be useful

Representative families failing even the 50% typical-spread criterion:

| family | typical relative | worst relative | early | late | reason/caveat |
|---|---:|---:|---:|---:|---|
| `noise/b_noise` | 68.11% | 246.30% | 72.99% | 67.62% | Noisy and one aligned point is lost; batch/noise caveat applies. |
| `noise/pairwise_cosines` | 79.03% | 5,343.32% | 83.51% | 76.79% | Near-zero denominators create extreme point noise. |
| `noise/signal_raw` | 93.90% | 420.15% | 125.07% | 92.36% | Too variable for modest effects. |
| `attn/per_head_norm_entropy` | 61.28% | 171.19% | 23.07% | 69.02% | Probe sampling and late noise dominate. |
| `attn/per_head_mean_distance` | 110.59% | 659.82% | 74.12% | 114.17% | Probe-dependent and order-one seed spread. |
| `curvature/eta_star [shadow_fp32]` | 59.23% | 189.57% | 31.55% | 69.85% | Certified only on the 25 common passing gradient points; short-probe variance applies. |
| `curvature/gHg [shadow_fp32]` | 70.58% | 248.44% | 37.94% | 82.15% | Same pass/probe restriction; order-one noise. |
| `update/actual [shadow_fp32]` | 171.81% | 28,991.40% | 78.48% | 320.46% | Sign/near-zero sensitivity makes the relative channel unusable. |
| `update/normalized_residual [shadow_fp32]` | 407.73% | 1,996.23% | 237.64% | 640.05% | Several-fold seed range. |
| `sketch/grad` | 381.81% | infinity | 279.01% | 399.04% | Signed sketch components cross zero; relative spread is intrinsically ill-conditioned. |
| `sketch/probe_grad` | 378.47% | infinity | 360.62% | 391.70% | Same, plus per-seed short probes. |

The full table is the decision surface: intermediate families can still detect effects larger than their observed spread, and “noisy” does not imply an implementation defect.

## Probe and certification asymmetries

The `probe` flag marks every family with a non-null `probe_id`: all 20 periodic `attn/*` series, all 50 periodic `probe/*` series, 180 arm-specific sparse `curvature/*` series, three sparse `sketch/probe_grad*` series, and 18 arm-specific sparse `update/*` series—**271 of 366** reported series. Their spread combines trajectory differences with independent probe draws. A single-run comparison cannot separate those components.

Every native-arm family is flagged `uncertified`; its raw variability is a diagnostic about the failed bf16 acceptance arm, never a measurement recommendation. Shadow random/update direction families with no passing records are `NA`, not zero. Only passing shadow gradient quantities may be read as curvature measurements.

## Limitations and deviations

1. The strict allowed-read list omits `../../TEMPLATE-result.md` while separately requiring its header schema. I did not read it, prioritizing strict isolation, and reconstructed the header from the fields explicitly named in the request and working procedure. This is an explicit protocol-administration deviation, not a data-analysis reinterpretation.
2. The protocol does not define “spread,” “typical,” the denominator sign convention, family-level channel pooling, the early/late boundary, or thresholds for “tight” and “noisy.” I used range, median, absolute median denominator, equal weighting of channel-progress elements, the end of the 400-step ramp, and the declared 5%/20%/50% rules. Other defensible choices can change ranks near cutoffs.
3. Relative spread is not a reliable scale measure for signed or near-zero families. The absolute-spread columns are essential for those rows; infinity and very large ratios should not be interpreted as infinite absolute instability.
4. Vector components are treated as channels. This makes comparisons exhaustive and preserves head/row/token/sketch indices, but “typical family” weights every component-progress element equally. It may hide a small subset of unstable layers/heads; the worst column partially exposes that.
5. Probe hashes are per seed and do not encode human-readable probe type in parquet. I mapped the two `probe/loss` hashes via the final train-stream/validation ordering documented in the permitted profiles; the remaining hash is `short`. This is deterministic here but would need provenance metadata in a general loader.
6. The native raw-spread exception is required to satisfy “report its spread” despite zero passing native directions. All native-arm rows remain explicitly uncertified. Nine native curvature series still have no five-seed common defined comparison and are reported unavailable.
7. This is a descriptive range over five seeds on one platform, not an estimated population variance or confidence interval. Worst values are especially sensitive to n, multiple channels, and the number of training points. Threshold classifications are exploratory and effect-size-dependent.
8. Configuration, categorical verdict, finite/skip, and timing families are mechanically ranked because the protocol requires every family. A zero range for a fixed control does not make it a scientifically useful outcome variable, and categorical code ratios have no interval-scale interpretation.
9. Gradient-noise results inherit the data card's device-batch/loader-underinstrumentation caveat. Muon stage quantities remain reference-frame quantities with `muon/replay_update_relerr` as their separate error bar.

## Complete family ranking

Sorted by typical relative seed spread. `n` is the number of fully aligned scalar/vector elements; key coverage is common semantic row keys divided by their five-run union. Absolute values retain each metric's native units. Flags: `probe` includes independent probe sampling, `uncertified` is native bf16-arm raw data, `pass-only` is shadow per-direction-passing data, and `zero-median` means at least one positive range had a zero median. `NA` families are included at the end.

"""


def main() -> None:
    table = (HERE / "family_ranking.md").read_text()
    artifacts = r"""
## Reproduction artifacts

- `analyze.py` — full parquet loading, alignment, filtering, ranking, and artifact generation.
- `validate.py` — independent direct-parquet spot checks for headline families and structural assertions.
- `family_ranking.csv` — complete machine-readable results with additional coverage/certification columns.
- `summary.json` — universe counts, verdict counts, thresholds, and conventions.
"""
    (HERE / "result.md").write_text(PREAMBLE + table + artifacts)


if __name__ == "__main__":
    main()
