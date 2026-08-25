#!/usr/bin/env python3
"""Independent spot checks for the A0002 family ranking."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


HERE = Path(__file__).resolve().parent
ROOT = Path("/home/felipe/Igalia/nanochat/telemetry-data/sweep/telemetry-data")
SEGMENTS = {
    "d12-s7": "d12-s7-s0-6c217854f4174ce884dfb2b2dcc13c45",
    "d12-s8": "d12-s8-s0-2b2e72e4395440029b92226213d137bb",
    "d12-s9": "d12-s9-s0-b872e698e81b4a8fa0cd08512f43c2c2",
    "d12-s10": "d12-s10-s0-0ffeb05742bb4154a3f1afc202dc3955",
    "d12-s11": "d12-s11-s0-5fd1bf807f764954b30658872fc3e2ad",
}


def read(run: str, tier: str, columns: list[str]) -> pd.DataFrame:
    return ds.dataset(ROOT / SEGMENTS[run] / tier, format="parquet").to_table(
        columns=columns
    ).to_pandas()


def direct_scalar(metric: str, tier: str, arm: str | None = None) -> tuple[float, float, int]:
    columns = [
        "metric",
        "normalized_progress",
        "value_scalar",
        "is_defined",
        "acceptance_arm",
    ]
    pieces = []
    for run in SEGMENTS:
        frame = read(run, tier, columns)
        rows = frame[(frame.metric == metric) & frame.is_defined].copy()
        if arm is None:
            rows = rows[rows.acceptance_arm.isna()]
        else:
            rows = rows[rows.acceptance_arm == arm]
        # This validator is intentionally only for one-scalar-per-progress families.
        assert not rows.normalized_progress.duplicated().any(), (run, metric)
        rows["run"] = run
        pieces.append(rows[["normalized_progress", "run", "value_scalar"]])
    wide = pd.concat(pieces).pivot(
        index="normalized_progress", columns="run", values="value_scalar"
    ).dropna()
    values = wide.to_numpy()
    spread = values.max(axis=1) - values.min(axis=1)
    relative = spread / np.abs(np.median(values, axis=1))
    return float(np.median(relative)), float(np.max(relative)), len(relative)


def check_scalar(
    ranking: pd.DataFrame, family: str, metric: str, tier: str, arm: str | None = None
) -> None:
    typical, worst, n = direct_scalar(metric, tier, arm)
    row = ranking[ranking.family == family].iloc[0]
    np.testing.assert_allclose(typical, row.typical_relative_spread, rtol=1e-12)
    np.testing.assert_allclose(worst, row.worst_relative_spread, rtol=1e-12)
    assert n == row.aligned_elements
    print(f"PASS {family}: typical={typical:.12g}, worst={worst:.12g}, n={n}")


def direct_probe_loss() -> tuple[float, float, int]:
    pieces = []
    columns = [
        "metric",
        "normalized_progress",
        "probe_id",
        "value_scalar",
        "is_defined",
    ]
    for run in SEGMENTS:
        frame = read(run, "periodic", columns)
        rows = frame[(frame.metric == "probe/loss") & frame.is_defined].copy()
        finals = (
            rows.sort_values("normalized_progress")
            .groupby("probe_id")
            .tail(1)
            .sort_values("value_scalar")
        )
        labels = {
            finals.iloc[0].probe_id: "train_stream",
            finals.iloc[1].probe_id: "val",
        }
        rows["probe_label"] = rows.probe_id.map(labels)
        rows["run"] = run
        pieces.append(rows[["normalized_progress", "probe_label", "run", "value_scalar"]])
    wide = pd.concat(pieces).pivot(
        index=["normalized_progress", "probe_label"], columns="run", values="value_scalar"
    ).dropna()
    values = wide.to_numpy()
    spread = values.max(axis=1) - values.min(axis=1)
    relative = spread / np.abs(np.median(values, axis=1))
    return float(np.median(relative)), float(np.max(relative)), len(relative)


def direct_passing_eta_star() -> tuple[float, float, int]:
    pieces = []
    columns = [
        "metric",
        "step",
        "normalized_progress",
        "value_scalar",
        "is_defined",
        "acceptance_arm",
    ]
    for run in SEGMENTS:
        frame = read(run, "sparse", columns)
        passed = set(
            frame[
                (frame.metric == "curvature/verdict_code_gradient")
                & (frame.acceptance_arm == "shadow_fp32")
                & frame.is_defined
                & (frame.value_scalar == 0)
            ].step
        )
        rows = frame[
            (frame.metric == "curvature/eta_star")
            & (frame.acceptance_arm == "shadow_fp32")
            & frame.is_defined
            & frame.step.isin(passed)
        ].copy()
        rows["run"] = run
        pieces.append(rows[["normalized_progress", "run", "value_scalar"]])
    wide = pd.concat(pieces).pivot(
        index="normalized_progress", columns="run", values="value_scalar"
    ).dropna()
    values = wide.to_numpy()
    spread = values.max(axis=1) - values.min(axis=1)
    relative = spread / np.abs(np.median(values, axis=1))
    return float(np.median(relative)), float(np.max(relative)), len(relative)


def main() -> None:
    ranking = pd.read_csv(HERE / "family_ranking.csv")
    finite = ranking.typical_relative_spread.dropna().to_numpy()
    assert np.all(finite[:-1] <= finite[1:])
    assert ranking.typical_relative_spread.notna().sum() == 302
    assert ranking.typical_relative_spread.isna().sum() == 64
    assert len(ranking) == 366

    for tier, expected in (("continuous", 32), ("periodic", 118), ("sparse", 118)):
        metric_sets = []
        for run in SEGMENTS:
            frame = read(run, tier, ["metric"])
            metric_sets.append(set(frame.metric))
        assert all(metrics == metric_sets[0] for metrics in metric_sets[1:])
        assert len(metric_sets[0]) == expected
        print(f"PASS universe: {tier} has the same {expected} base families in all runs")

    check_scalar(
        ranking,
        "continuous:loss/train_mean",
        "loss/train_mean",
        "continuous",
    )
    check_scalar(ranking, "periodic:noise/b_noise", "noise/b_noise", "periodic")
    check_scalar(
        ranking,
        "sparse:update/actual [shadow_fp32]",
        "update/actual",
        "sparse",
        "shadow_fp32",
    )
    for family, direct in (
        ("periodic:probe/loss", direct_probe_loss),
        ("sparse:curvature/eta_star [shadow_fp32]", direct_passing_eta_star),
    ):
        typical, worst, n = direct()
        row = ranking[ranking.family == family].iloc[0]
        np.testing.assert_allclose(typical, row.typical_relative_spread, rtol=1e-12)
        np.testing.assert_allclose(worst, row.worst_relative_spread, rtol=1e-12)
        assert n == row.aligned_elements
        print(f"PASS {family}: typical={typical:.12g}, worst={worst:.12g}, n={n}")

    # Check the documented zero convention on a constant-zero family.
    zero = ranking[ranking.family == "continuous:step/skipped"].iloc[0]
    assert zero.typical_absolute_spread == 0
    assert zero.typical_relative_spread == 0
    assert zero.zero_median_fraction == 1
    assert zero.infinite_relative_fraction == 0
    print("PASS zero convention: continuous:step/skipped uses 0/0 -> 0")

    # All full-coverage families must have every selected row key in all runs.
    full = ranking[ranking.row_key_coverage == 1]
    assert (full.aligned_row_keys == full.union_row_keys).all()
    assert not (ranking.vector_length_mismatch_keys > 0).any()
    print(f"PASS structural checks: {len(full)} full-coverage series; no vector mismatches")


if __name__ == "__main__":
    main()
