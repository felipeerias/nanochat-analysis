# Cross-implementation answer sheet — d12-iter segment

Two INDEPENDENT implementations (Claude: `analysis/claude/`, Codex:
`analysis-codex/`) each produce a `results.json` with exactly the keys
below, computed from the segment

    ~/Igalia/nanochat/telemetry-data/runpod/telemetry-data/
        d12-iter-s0-0a3f5527067944708caeb7e1ff638b76/

using ONLY the parquet chunk files and `provenance.json`. Every quantity is
defined on raw table rows (columns: `metric`, `step`, `phase`, `value_scalar`,
`is_defined`, `param_role`, `layer`, `parameter_name`, `probe_id`, tiers =
subdirectory names), so no interpretation layer is shared. Counts must match
exactly; floats to a relative tolerance of 1e-9 (both read the same stored
doubles). `null` where a value is undefined.

```json
{
  "segment": "<segment directory name>",
  "row_counts": {"continuous": N, "periodic": N, "sparse": N, "offline": N},
  "undefined_counts": {"<tier>": "rows with is_defined == false", "...": 0},
  "native_verdict_counts": {
    "passed|inconclusive|failed": "count of DEFINED rows of metric curvature/native_verdict_code with value 0.0|1.0|2.0"
  },
  "deep": {
    "<step in 1, 1001, 2001>": {
      "gHg":    "value_scalar of curvature/gHg at this step",
      "gg":     "value_scalar of curvature/gg at this step",
      "eta_star": "value_scalar of curvature/eta_star at this step, null if undefined",
      "dhd":    "value_scalar of curvature/dhd at this step",
      "update_p1":     "value_scalar of update/p1 at this step",
      "update_p2":     "value_scalar of update/p2 at this step",
      "update_actual": "value_scalar of update/actual at this step"
    }
  },
  "relerr": {
    "<step in 1, 1001, 2001>": {
      "n":      "count of DEFINED muon/replay_update_relerr rows at this step",
      "median": "median of their value_scalar",
      "max":    "max of their value_scalar",
      "zeros":  "count with value_scalar < 1e-12"
    }
  },
  "train_loss_first": "value_scalar of loss/train_mean at step 0 (exactly one row)",
  "train_loss_last":  "value_scalar of loss/train_mean at step 2519 (exactly one row)",
  "probe_val_loss_last": "value_scalar of the max-step probe/loss row whose probe_id equals provenance probe_ids.val",
  "overhead_total_seconds": "sum of value_scalar over offline rows whose metric starts with overhead/total/",
  "max_grad_norm_at_1000": "max value_scalar over rows with metric grad/norm, step 1000, phase pre_update"
}
```

Notes:
- All `deep` metrics above appear exactly once per step in this (schema v1)
  segment; if an implementation finds a different multiplicity it must say so
  instead of silently picking one.
- Median: for even n, the mean of the two central order statistics
  (`numpy`/`pandas` default).
