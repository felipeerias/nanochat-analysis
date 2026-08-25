#!/usr/bin/env python3
"""Compute the cross-validation answer sheet directly from raw Parquet rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


TIERS = ("continuous", "periodic", "sparse", "offline")
STEPS = (1, 1001, 2001)
READ_COLUMNS = (
    "metric",
    "tier",
    "phase",
    "step",
    "value_scalar",
    "is_defined",
    "probe_id",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEGMENT = (
    REPO_ROOT
    / "telemetry-data"
    / "runpod"
    / "telemetry-data"
    / "d12-iter-s0-0a3f5527067944708caeb7e1ff638b76"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_tiers(segment: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for tier in TIERS:
        files = sorted((segment / tier).glob("*.parquet"))
        require(bool(files), f"No Parquet files found for tier {tier!r}")
        table = pq.read_table([str(path) for path in files], columns=list(READ_COLUMNS))
        frame = table.to_pandas()
        require(
            frame["tier"].notna().all() and frame["tier"].eq(tier).all(),
            f"Tier column disagrees with directory {tier!r}",
        )
        require(
            frame["is_defined"].notna().all(),
            f"Null is_defined value found in tier {tier!r}",
        )
        frames[tier] = frame
    return frames


def one_row(rows: pd.DataFrame, description: str) -> pd.Series:
    require(len(rows) == 1, f"Expected exactly one {description} row, found {len(rows)}")
    return rows.iloc[0]


def defined_scalar(row: pd.Series, description: str) -> float:
    require(bool(row["is_defined"]), f"{description} is unexpectedly undefined")
    require(pd.notna(row["value_scalar"]), f"{description} has a null scalar")
    return float(row["value_scalar"])


def compute(segment: Path) -> dict[str, Any]:
    provenance_path = segment / "provenance.json"
    require(provenance_path.is_file(), f"Missing provenance: {provenance_path}")
    with provenance_path.open("r", encoding="utf-8") as handle:
        provenance = json.load(handle)
    val_probe_id = provenance["probe_ids"]["val"]

    by_tier = read_tiers(segment)
    all_rows = pd.concat(by_tier.values(), ignore_index=True)

    row_counts = {tier: int(len(by_tier[tier])) for tier in TIERS}
    undefined_counts = {
        tier: int(by_tier[tier]["is_defined"].eq(False).sum()) for tier in TIERS
    }

    verdict_rows = all_rows.loc[
        all_rows["metric"].eq("curvature/native_verdict_code")
        & all_rows["is_defined"]
    ]
    require(
        verdict_rows["value_scalar"].notna().all(),
        "A defined native verdict has a null scalar",
    )
    verdict_values = verdict_rows["value_scalar"].astype(float)
    unexpected_verdicts = set(verdict_values.unique()) - {0.0, 1.0, 2.0}
    require(not unexpected_verdicts, f"Unexpected native verdict codes: {unexpected_verdicts}")
    native_verdict_counts = {
        "passed": int(verdict_values.eq(0.0).sum()),
        "inconclusive": int(verdict_values.eq(1.0).sum()),
        "failed": int(verdict_values.eq(2.0).sum()),
    }

    deep_metrics = {
        "gHg": "curvature/gHg",
        "gg": "curvature/gg",
        "eta_star": "curvature/eta_star",
        "dhd": "curvature/dhd",
        "update_p1": "update/p1",
        "update_p2": "update/p2",
        "update_actual": "update/actual",
    }
    deep: dict[str, dict[str, float | None]] = {}
    for step in STEPS:
        values: dict[str, float | None] = {}
        for output_name, metric in deep_metrics.items():
            row = one_row(
                all_rows.loc[
                    all_rows["metric"].eq(metric) & all_rows["step"].eq(step)
                ],
                f"{metric} at step {step}",
            )
            if output_name == "eta_star" and not bool(row["is_defined"]):
                values[output_name] = None
            else:
                values[output_name] = defined_scalar(row, f"{metric} at step {step}")
        deep[str(step)] = values

    relerr: dict[str, dict[str, int | float | None]] = {}
    for step in STEPS:
        rows = all_rows.loc[
            all_rows["metric"].eq("muon/replay_update_relerr")
            & all_rows["step"].eq(step)
            & all_rows["is_defined"]
        ]
        require(
            rows["value_scalar"].notna().all(),
            f"A defined replay relerr at step {step} has a null scalar",
        )
        samples = rows["value_scalar"].to_numpy(dtype=np.float64)
        relerr[str(step)] = {
            "n": int(samples.size),
            "median": float(np.median(samples)) if samples.size else None,
            "max": float(np.max(samples)) if samples.size else None,
            "zeros": int(np.count_nonzero(samples < 1e-12)),
        }

    train_loss_first_row = one_row(
        all_rows.loc[
            all_rows["metric"].eq("loss/train_mean") & all_rows["step"].eq(0)
        ],
        "loss/train_mean at step 0",
    )
    train_loss_last_row = one_row(
        all_rows.loc[
            all_rows["metric"].eq("loss/train_mean") & all_rows["step"].eq(2519)
        ],
        "loss/train_mean at step 2519",
    )

    val_probe_rows = all_rows.loc[
        all_rows["metric"].eq("probe/loss")
        & all_rows["probe_id"].eq(val_probe_id)
    ]
    require(not val_probe_rows.empty, "No probe/loss rows matched provenance probe_ids.val")
    max_probe_step = int(val_probe_rows["step"].max())
    probe_val_last_row = one_row(
        val_probe_rows.loc[val_probe_rows["step"].eq(max_probe_step)],
        f"validation probe/loss at max step {max_probe_step}",
    )

    overhead_rows = by_tier["offline"].loc[
        by_tier["offline"]["metric"].str.startswith("overhead/total/", na=False)
    ]
    require(
        overhead_rows["value_scalar"].notna().all(),
        "An overhead/total row has a null scalar",
    )
    overhead_total_seconds = float(
        overhead_rows["value_scalar"].to_numpy(dtype=np.float64).sum()
    )

    grad_rows = all_rows.loc[
        all_rows["metric"].eq("grad/norm")
        & all_rows["step"].eq(1000)
        & all_rows["phase"].eq("pre_update")
    ]
    require(not grad_rows.empty, "No grad/norm rows at step 1000, phase pre_update")
    require(
        grad_rows["value_scalar"].notna().all(),
        "A selected grad/norm row has a null scalar",
    )

    return {
        "segment": segment.name,
        "row_counts": row_counts,
        "undefined_counts": undefined_counts,
        "native_verdict_counts": native_verdict_counts,
        "deep": deep,
        "relerr": relerr,
        "train_loss_first": defined_scalar(train_loss_first_row, "first train loss"),
        "train_loss_last": defined_scalar(train_loss_last_row, "last train loss"),
        "probe_val_loss_last": defined_scalar(probe_val_last_row, "last validation probe loss"),
        "overhead_total_seconds": overhead_total_seconds,
        "max_grad_norm_at_1000": float(
            grad_rows["value_scalar"].to_numpy(dtype=np.float64).max()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", type=Path, default=DEFAULT_SEGMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = compute(args.segment.resolve())
    rendered = json.dumps(results, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
