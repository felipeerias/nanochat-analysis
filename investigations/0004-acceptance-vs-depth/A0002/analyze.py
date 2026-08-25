#!/usr/bin/env python3
"""Prespecified analysis for I0004, analyst run A0002.

The unit summarized for the decision is a run: for each error channel and
acceptance arm, take the median across all defined sparse/deep checkpoints.
At d12, summarize the five run medians by their median and full min--max seed
band.  Apply the frozen decision rule independently to all six shadow-fp32
channels.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-I0004-A0002")
)

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HERE = Path(__file__).resolve().parent
ANALYSIS_ROOT = HERE.parents[2]
DATA_ROOT = ANALYSIS_ROOT.parent / "telemetry-data" / "sweep" / "telemetry-data"
sys.path.insert(0, str(ANALYSIS_ROOT))

from loader import telemetry_load as tl  # noqa: E402


SEGMENTS = (
    "d12-s7-s0-6c217854f4174ce884dfb2b2dcc13c45",
    "d12-s8-s0-2b2e72e4395440029b92226213d137bb",
    "d12-s9-s0-b872e698e81b4a8fa0cd08512f43c2c2",
    "d12-s10-s0-0ffeb05742bb4154a3f1afc202dc3955",
    "d12-s11-s0-5fd1bf807f764954b30658872fc3e2ad",
    "d14-s7-s0-b72acf7ac59942dab902e26fa0b2c80d",
    "d16-s7-s0-8e0a39fc8f164343bd01ef6b915e849f",
)
ARMS = ("native", "shadow_fp32")
DIRECTIONS = ("random", "gradient", "update")
ERROR_KINDS = ("e_sym", "e_lin")
METRICS = tuple(
    f"curvature/{kind}_{direction}"
    for kind in ERROR_KINDS
    for direction in DIRECTIONS
)
THRESHOLD = 1e-4


def load_targets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate exactly the seven schema-v3 sparse segments."""
    selected_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []

    for segment in SEGMENTS:
        segment_root = DATA_ROOT / segment
        with (segment_root / "provenance.json").open() as f:
            provenance = json.load(f)
        depth = int(provenance["model_config"]["n_layer"])
        seed = int(provenance["seed"])
        expected_steps = {int(step) + 1 for step in provenance["telemetry_deep_steps"]}

        sparse = tl.read_telemetry(str(DATA_ROOT), segment, "sparse").to_pandas()
        schema_versions = set(sparse["schema_version"].dropna().astype(str))
        if schema_versions != {"3"}:
            raise AssertionError(f"{segment}: expected schema 3, got {schema_versions}")

        target_all = sparse[sparse["metric"].isin(METRICS)].copy()
        target = tl.defined(target_all).copy()
        target = target[target["acceptance_arm"].isin(ARMS)].copy()
        if set(target["metric"].unique()) != set(METRICS):
            raise AssertionError(f"{segment}: target metric mismatch")
        if set(target["acceptance_arm"].unique()) != set(ARMS):
            raise AssertionError(f"{segment}: target arm mismatch")
        if set(target["step"].astype(int).unique()) != expected_steps:
            raise AssertionError(f"{segment}: target steps do not match deep schedule")
        if target.duplicated(["metric", "acceptance_arm", "step"]).any():
            raise AssertionError(f"{segment}: duplicate metric/arm/deep-step rows")

        per_series_counts = target.groupby(["metric", "acceptance_arm"]).size()
        if not (per_series_counts == len(expected_steps)).all():
            raise AssertionError(f"{segment}: incomplete defined target series")

        target["segment"] = segment
        target["depth"] = depth
        target["seed"] = seed
        selected_parts.append(target)
        audit_rows.append(
            {
                "segment": segment,
                "depth": depth,
                "seed": seed,
                "deep_checkpoints": len(expected_steps),
                "target_rows_total": len(target_all),
                "target_rows_undefined": int((~target_all["is_defined"]).sum()),
                "target_rows_selected": len(target),
            }
        )

    return pd.concat(selected_parts, ignore_index=True), pd.DataFrame(audit_rows)


def summarize(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["segment", "depth", "seed", "acceptance_arm", "metric"]
    run = (
        selected.groupby(keys, as_index=False)
        .agg(
            n_checkpoints=("value_scalar", "size"),
            median=("value_scalar", "median"),
            minimum=("value_scalar", "min"),
            maximum=("value_scalar", "max"),
            q25=("value_scalar", lambda x: x.quantile(0.25)),
            q75=("value_scalar", lambda x: x.quantile(0.75)),
            n_above_threshold=("value_scalar", lambda x: int((x > THRESHOLD).sum())),
        )
        .sort_values(["acceptance_arm", "metric", "depth", "seed"])
        .reset_index(drop=True)
    )

    depth = (
        run.groupby(["acceptance_arm", "metric", "depth"], as_index=False)
        .agg(
            n_runs=("median", "size"),
            depth_median=("median", "median"),
            seed_min=("median", "min"),
            seed_max=("median", "max"),
        )
        .sort_values(["acceptance_arm", "metric", "depth"])
        .reset_index(drop=True)
    )
    return run, depth


def decisions(run: pd.DataFrame, depth: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    shadow_run = run[run["acceptance_arm"] == "shadow_fp32"]
    shadow_depth = depth[depth["acceptance_arm"] == "shadow_fp32"]

    for metric in METRICS:
        values = shadow_depth[shadow_depth["metric"] == metric].set_index("depth")
        d12 = float(values.loc[12, "depth_median"])
        d14 = float(values.loc[14, "depth_median"])
        d16 = float(values.loc[16, "depth_median"])
        d12_runs = shadow_run[
            (shadow_run["metric"] == metric) & (shadow_run["depth"] == 12)
        ]["median"]
        seed_min = float(d12_runs.min())
        seed_max = float(d12_runs.max())
        monotone = d12 < d14 < d16

        if d16 > seed_max and monotone:
            outcome = "supported"
        elif seed_min <= d16 <= seed_max:
            outcome = "refuted"
        else:
            outcome = "inconclusive"

        rows.append(
            {
                "metric": metric,
                "d12_median": d12,
                "d12_seed_min": seed_min,
                "d12_seed_max": seed_max,
                "d14_median": d14,
                "d16_median": d16,
                "monotone_d12_d14_d16": monotone,
                "d16_exceeds_d12_seed_max": d16 > seed_max,
                "outcome": outcome,
            }
        )
    return pd.DataFrame(rows)


def crossing_sensitivity(depth: pd.DataFrame) -> pd.DataFrame:
    """Compute transparent model-sensitivity diagnostics, not a CI.

    With three depth points and one seed at d14/d16, these projections cannot
    support an inferential crossing estimate.  They are retained to show how
    much an answer depends on arbitrary raw-vs-log and all-points-vs-last-two
    choices.
    """
    q = depth[
        (depth["acceptance_arm"] == "shadow_fp32")
        & (depth["metric"] == "curvature/e_sym_gradient")
    ].sort_values("depth")
    x = q["depth"].to_numpy(dtype=float)
    y = q["depth_median"].to_numpy(dtype=float)

    fits: list[dict[str, object]] = []

    def add_fit(name: str, xx: np.ndarray, yy: np.ndarray, log_y: bool) -> None:
        response = np.log(yy) if log_y else yy
        slope, intercept = np.polyfit(xx, response, 1)
        target = np.log(THRESHOLD) if log_y else THRESHOLD
        crossing = (target - intercept) / slope if slope > 0 else np.nan
        fitted = slope * xx + intercept
        if log_y:
            fitted = np.exp(fitted)
        fits.append(
            {
                "model": name,
                "slope_positive": bool(slope > 0),
                "crossing_depth": float(crossing),
                "max_abs_residual": float(np.max(np.abs(yy - fitted))),
            }
        )

    add_fit("raw-linear, all 3 depths", x, y, False)
    add_fit("log-linear, all 3 depths", x, y, True)
    add_fit("raw-linear, d14-d16", x[-2:], y[-2:], False)
    add_fit("log-linear, d14-d16", x[-2:], y[-2:], True)
    return pd.DataFrame(fits)


def plot_summary(run: pd.DataFrame, depth: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8), sharex=True)
    colors = {"native": "#b44b4b", "shadow_fp32": "#286f9b"}
    labels = {"native": "native bf16", "shadow_fp32": "shadow fp32"}

    for row_index, kind in enumerate(ERROR_KINDS):
        for col_index, direction in enumerate(DIRECTIONS):
            ax = axes[row_index, col_index]
            metric = f"curvature/{kind}_{direction}"
            for arm in ARMS:
                qd = depth[(depth["metric"] == metric) & (depth["acceptance_arm"] == arm)]
                qr = run[(run["metric"] == metric) & (run["acceptance_arm"] == arm)]
                color = colors[arm]
                ax.plot(
                    qd["depth"], qd["depth_median"], marker="o", color=color,
                    linewidth=1.7, label=labels[arm],
                )
                d12_runs = qr[qr["depth"] == 12]
                ax.scatter(
                    np.full(len(d12_runs), 12.0) + np.linspace(-0.13, 0.13, len(d12_runs)),
                    d12_runs["median"], s=24, facecolors="none", edgecolors=color,
                    linewidths=1.0, zorder=3,
                )
            ax.axhline(THRESHOLD, color="black", linestyle="--", linewidth=1.0)
            ax.set_yscale("log")
            ax.set_title(f"{kind}: {direction}")
            ax.set_xticks([12, 14, 16])
            ax.grid(True, which="both", alpha=0.22)
            if col_index == 0:
                ax.set_ylabel("run median error")
            if row_index == 1:
                ax.set_xlabel("recipe depth")

    axes[0, 0].legend(loc="best", fontsize=9)
    fig.suptitle(
        "Acceptance self-consistency errors vs nanochat recipe depth\n"
        "open circles: five d12 seed medians; dashed line: 1e-4 threshold",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "acceptance_medians_by_depth.png", dpi=180)
    plt.close(fig)


def plot_gradient_checkpoints(selected: pd.DataFrame) -> None:
    q = selected[
        (selected["acceptance_arm"] == "shadow_fp32")
        & (selected["metric"] == "curvature/e_sym_gradient")
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    colors = {12: "#6a8f3f", 14: "#b9801d", 16: "#73559a"}
    for (depth, seed), group in q.groupby(["depth", "seed"]):
        label = f"d{depth}, s{seed}" if depth != 12 else ("d12, five seeds" if seed == 7 else None)
        alpha = 0.35 if depth == 12 else 0.9
        linewidth = 0.8 if depth == 12 else 1.4
        ax.plot(
            group["normalized_progress"], group["value_scalar"],
            marker=".", markersize=3.5, linewidth=linewidth,
            alpha=alpha, color=colors[int(depth)], label=label,
        )
    ax.axhline(THRESHOLD, color="black", linestyle="--", linewidth=1.1, label="1e-4 threshold")
    ax.set_yscale("log")
    ax.set_xlabel("normalized training progress")
    ax.set_ylabel("shadow fp32 e_sym_gradient")
    ax.set_title("Defined deep-checkpoint values (not only the two early d16 failures)")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "e_sym_gradient_checkpoints.png", dpi=180)
    plt.close(fig)


def main() -> None:
    selected, audit = load_targets()
    run, depth = summarize(selected)
    decision = decisions(run, depth)
    crossing = crossing_sensitivity(depth)

    outputs = {
        "segment_audit.csv": audit,
        "run_medians.csv": run,
        "depth_summary.csv": depth,
        "decision.csv": decision,
        "crossing_sensitivity.csv": crossing,
    }
    for filename, frame in outputs.items():
        frame.to_csv(HERE / filename, index=False, float_format="%.17g")
    plot_summary(run, depth)
    plot_gradient_checkpoints(selected)

    print("SEGMENT AUDIT")
    print(audit.to_string(index=False))
    print("\nSHADOW DECISIONS")
    print(decision.to_string(index=False))
    print("\nE_SYM_GRADIENT CROSSING SENSITIVITY")
    print(crossing.to_string(index=False))
    print("\nPHASES", sorted(selected["phase"].dropna().unique()))
    print("SELECTED ROWS", len(selected))


if __name__ == "__main__":
    main()
