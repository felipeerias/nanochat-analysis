---
source: d12-s11-s0-5fd1bf807f764954b30658872fc3e2ad
profile_spec: 1
generator: cc5ecea:profiles/generate_profiles.py
generated: 2026-08-25
author: Claude Code (Fable 5)
---

# Profile — d12-s11

## Inventory

- schema v3, seed 11, depth 12, width 768, heads 6
- 2520 steps; periodic every 101; 30 deep checkpoints
- backend fa3, compute dtype torch.bfloat16, shadow arm fp32
- lineage checkpoint labels: [0, 631, 1261, 1890]

| tier | rows | undefined | metric families |
|---|---|---|---|
| continuous | 194,094 | 0 | 32 |
| periodic | 82,387 | 737 | 118 |
| sparse | 15,483 | 449 | 118 |
| offline | 15 | 0 | 15 |

## Summaries

- train loss: 10.3980 at the first step, 2.7733 at the last (2,520 points)
- final probe loss, train_stream: 2.8035
- final probe loss, val: 2.9264
- telemetry overhead: 649.3 s total; update_effectiveness 193.2 s, shadow_acceptance 184.2 s, calibration 147.0 s, noise 24.2 s, lineage_inventory 23.9 s

### Acceptance verdicts

Checkpoint level (worst of the three probe directions):

- native: {'failed': 30}
- shadow_fp32: {'inconclusive': 28, 'failed': 2}

Per direction:

| arm | direction | verdicts |
|---|---|---|
| native | random | {'failed': 30} |
| native | gradient | {'failed': 30} |
| native | update | {'failed': 30} |
| shadow_fp32 | random | {'inconclusive': 28, 'failed': 2} |
| shadow_fp32 | gradient | {'inconclusive': 4, 'passed': 26} |
| shadow_fp32 | update | {'inconclusive': 30} |

![loss](figures/loss.png)

## Data quality notes

- Undefined rows are present and carry reasons: 737 periodic, 449 sparse. They are honest 'not measurable here' records, not missing data.
- Every native (bf16) checkpoint verdict is `failed`. This is a property of bf16 arithmetic against fp32-era thresholds, documented in DATASET.md caveat 4. Native curvature values are not certified measurements.
- Shadow (IEEE fp32) checkpoint verdicts are the worst of three directions, so they understate what is usable; see the per-direction table above.
- Lineage checkpoint tensors (`checkpoints/*.pt`) are absent from this local copy by design; they remain on the GPU volume. Verification of their hashes is not possible here.
