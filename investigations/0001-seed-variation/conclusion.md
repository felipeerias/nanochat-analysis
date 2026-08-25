# I0001 — conclusion

Status: **closed**. Evidence level: **reproduced** (two blind analyses of the
same data agree). Not confirmed — confirmation would need new runs.

Runs: [A0001](A0001/result.md) (Claude Code, Fable 5, commit `ee85a2e`) and
[A0002](A0002/result.md) (Codex, GPT-5, commit `b82e155`, written blind
against protocol commit `2a460b5`).

## What the reference says

Seed variation is **not one number**. Across the five d12 runs it spans five
orders of magnitude between metric families. Both analyses agree on the
ordering: **Spearman rank correlation 0.997 across the 83 families keyed
identically by both**.

Use these as the reference spreads. The first column is the standard
deviation across five seeds relative to the median; the second is the
five-seed range relative to the median. Cite which one you used.

| family | sd-relative | range-relative | usable? |
|---|---:|---:|---|
| `loss/train_mean` | 0.06% | 0.16% | best detector in the dataset |
| `probe/loss` | 0.16% | 0.45% | tight; includes probe sampling |
| `param/norm` | 1.0% | ~2.5% | tight |
| `muon/replay_update_relerr` | 3.5% | ~8% | tight; a real response channel |
| `muon/cos_raw_final` | 7.3% | ~18% | usable for large effects |
| `curvature/dhd` | 13% | 32% | large effects only |
| `noise/s2` | 14% | ~33% | large effects only |
| `curvature/eta_star` | 25% | 59% | large effects only |
| `noise/b_noise` | 26% | 68% | large effects only |
| `curvature/gHg` | 29% | 71% | large effects only |
| `optim/adamw_m_rms` | 65% | — | not usable |
| `update/p2` | 79% (142% late) | — | not usable |
| `curvature/c_fd_*`, `curv_floor_*` | 100x+ | — | instrument noise diagnostics, not observables |

Configuration channels (`optim/lr`, `optim/beta1`, `batch/*`, `mem/*`) have
**exactly zero** spread in both analyses. That is the correct answer and it
confirms neither pipeline manufactures variance.

## Practical rule

An effect must clear roughly **two to three times** the sd-relative spread of
the channel it is measured on before five runs can distinguish it from seed
noise. Concretely: a change is detectable on training loss at well under 1%,
on Muon decoherence at around 10%, and on curvature only above roughly 50-75%.
Subtle curvature effects are not measurable with five runs at d12.

## Where the two analyses differ, and why

The magnitudes differ by a constant factor with a known cause. Everything
else is a disclosed methodological choice, not a contradiction.

1. **Dispersion statistic.** A0001 used the standard deviation; A0002 used the
   range (max − min). Empirically the ratio is **2.45** (IQR 2.34–2.54)
   against a theoretical 2.33 for five Gaussian samples — so the difference is
   explained almost entirely by the choice, and the small excess indicates
   mildly heavier tails than Gaussian. Prefer the **standard deviation** as the
   canonical figure, because it does not grow with sample size and so remains
   comparable when a future experiment uses a different number of runs.
2. **Verdict conditioning.** The protocol said curvature would be restricted to
   passing per-direction verdicts. A0002 applied it; A0001 did not, arguing
   that a literal reading deletes every arm-level curvature channel. Both
   disclosed the choice. A0002's restricted figures are the ones to cite for
   *certified* curvature; A0001's unrestricted figures describe the channel's
   variability regardless of certification. They answer different questions.
3. **Vectors.** A0001 excluded all 69 vector-valued families as requiring an
   unauthorized summary choice. A0002 expanded vector components into
   comparison elements (27.5M of them) and reports them — finding gradient
   sketches at roughly 380% relative range. A0002's coverage is better here.
4. **Universe accounting.** A0001 counted 250 metric-by-arm families over 3,392
   channels; A0002 counted 268 base families over 366 arm-specific series with
   a finer semantic key. Both reported their entire universe; neither selected
   a subset.

## A finding neither analysis was looking for

A0002's verdict conditioning surfaced something operationally important:
**among shadow-arm curvature checkpoints, only the gradient direction ever
passes.** The random and update directions pass at zero checkpoints in all
five runs, and the gradient direction passes at 25–26 of 30 per run, with 25
common to all five.

Certified curvature therefore exists only along the gradient direction. Any
future work using `curvature/*_random` or `curvature/*_update` is working with
uncertified numbers and must say so. This was invisible to A0001 because it
did not apply the filter.

## Consequences for the instrument and for future work

- The d12 configuration is usable as an intervention testbed, but only for
  channels in the top half of the table. Loss and Muon decoherence are the
  channels most likely to register a change.
- Curvature is a weak response variable at n=5. An intervention study aimed at
  curvature would need more seeds, and the required number scales with the
  square of the effect size you hope to see.
- The protocol's curvature-selection line was underspecified: it admitted two
  reasonable readings that produce different populations. Future protocols
  should state selection in terms that survive the case where nothing passes.

## Gate

This investigation is closed, so the gate in `../../README.md` is lifted.
Comparative and effect claims may now cite
`investigations/0001-seed-variation/conclusion.md@<commit>` and must state
which dispersion statistic they used.

The reference is **d12 only**. Caveats 1 and 3 in `DATASET.md` apply: depth
covaries with width, batch size and learning rate, and warmup is a fixed step
count, so this spread should not be assumed to transfer to d14 or d16.
