---
source: all schema-v3 segments in telemetry-data/sweep
profile_spec: 1
generator: cc5ecea:profiles/generate_profiles.py
generated: 2026-08-25
author: Claude Code (Fable 5)
---

# Profile — the d12-d16 collection

## Inventory

| run | seed | depth | width | heads | steps | periodic every | deep ckpts | train loss last |
|---|---|---|---|---|---|---|---|---|
| d12-s7 | 7 | 12 | 768 | 6 | 2520 | 101 | 30 | 2.7761 |
| d12-s8 | 8 | 12 | 768 | 6 | 2520 | 101 | 30 | 2.7713 |
| d12-s9 | 9 | 12 | 768 | 6 | 2520 | 101 | 30 | 2.7744 |
| d12-s10 | 10 | 12 | 768 | 6 | 2520 | 101 | 30 | 2.7741 |
| d12-s11 | 11 | 12 | 768 | 6 | 2520 | 101 | 30 | 2.7733 |
| d14-s7 | 7 | 14 | 896 | 7 | 3759 | 151 | 32 | 2.6847 |
| d16-s7 | 7 | 16 | 1024 | 8 | 5376 | 216 | 33 | 2.5885 |

## Step grids

Runs that share a step grid can be compared point by point with no interpolation. Runs that do not must be aligned on `normalized_progress`.

- grid 1: 2520 steps, periodic every 101, 30 deep — d12-s10, d12-s11, d12-s7, d12-s8, d12-s9
- grid 2: 3759 steps, periodic every 151, 32 deep — d14-s7
- grid 3: 5376 steps, periodic every 216, 33 deep — d16-s7

## Data quality notes

- The five d12 runs come from two manifests (`sweep-d12-d16-v1` and `sweep-d12-seeds-v1`) but are configuration-identical apart from the seed.
- One legacy segment (`d12-iter`, schema v1, head_dim 64, no shadow arm) is present in the data folder and is not profiled here. Do not pool it with these runs.
