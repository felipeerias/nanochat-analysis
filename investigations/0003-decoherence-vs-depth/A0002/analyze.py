#!/usr/bin/env python3
"""Blind A0002 analysis for I0003: Muon decoherence versus model depth."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-mpl-i0003-a0002")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HERE = Path(__file__).resolve().parent
ANALYSIS_ROOT = HERE.parents[2]
DATA_ROOT = ANALYSIS_ROOT.parent / "telemetry-data" / "sweep" / "telemetry-data"
sys.path.insert(0, str(ANALYSIS_ROOT))

from loader.telemetry_load import defined, metric, read_telemetry  # noqa: E402


METRIC = "muon/replay_update_relerr"
RUN_IDS = (
    "d12-s7",
    "d12-s8",
    "d12-s9",
    "d12-s10",
    "d12-s11",
    "d14-s7",
    "d16-s7",
)
ROLES = ("attn_q", "attn_k", "attn_v", "attn_out", "mlp_in", "mlp_out", "ve_gate")
ROLE_COLORS = dict(zip(ROLES, plt.get_cmap("tab10").colors[: len(ROLES)]))


def segment_for(run_id: str) -> Path:
    matches = sorted(DATA_ROOT.glob(f"{run_id}-s0-*"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one segment for {run_id}, got {matches}")
    return matches[0]


def load_selected() -> tuple[pd.DataFrame, list[dict]]:
    frames: list[pd.DataFrame] = []
    provenance: list[dict] = []
    for expected_run_id in RUN_IDS:
        segment = segment_for(expected_run_id)
        prov = json.loads((segment / "provenance.json").read_text())
        sparse = read_telemetry(str(DATA_ROOT), segment.name, "sparse").to_pandas()
        selected_before_defined = metric(sparse, METRIC)
        selected = defined(selected_before_defined).copy()

        depth = int(prov["model_config"]["n_layer"])
        seed = int(prov["seed"])
        assert prov["manifest_run_id"] == expected_run_id
        assert prov["run_id"] == segment.name
        assert set(selected_before_defined["schema_version"].astype(str)) == {"3"}
        assert len(selected) == len(selected_before_defined)  # no undefined rows for this family
        assert selected["value_scalar"].notna().all()
        assert set(selected["phase"]) == {"post_update"}
        assert selected["acceptance_arm"].isna().all()

        selected["segment"] = segment.name
        selected["depth"] = depth
        selected["seed"] = seed
        # Sparse post_update step s+1 corresponds to deep update index s.
        selected["update_index"] = selected["step"].astype(int) - 1
        # Frozen protocol's relative-depth convention: zero-based layer index / depth.
        selected["relative_layer"] = selected["layer"].astype(float) / depth

        per_checkpoint = selected.groupby(["step", "parameter_name"], dropna=False).size()
        assert per_checkpoint.eq(1).all()
        expected_matrices = 13 * depth // 2
        assert selected.groupby("step").size().eq(expected_matrices).all()
        assert set(selected["param_role"]) == set(ROLES)

        frames.append(selected)
        provenance.append(prov)

    out = pd.concat(frames, ignore_index=True)
    assert len(out) == 18_044
    assert out["shape"].notna().sum() == 0
    assert (out.loc[out["update_index"] != 0, "value_scalar"] > 0).all()
    return out, provenance


def checkpoint_medians(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(
            ["segment", "run_id", "depth", "seed", "step", "normalized_progress"],
            as_index=False,
        )["value_scalar"]
        .median()
        .sort_values(["depth", "seed", "normalized_progress"])
    )


def classify(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.where(values < low, "below", np.where(values > high, "above", "inside"))


def decision_analysis(checkpoints: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d12_rows = checkpoints.query("depth == 12")
    grid = np.sort(d12_rows["normalized_progress"].unique())
    assert len(grid) == 30
    for _, seed_rows in d12_rows.groupby("seed"):
        np.testing.assert_array_equal(np.sort(seed_rows["normalized_progress"].unique()), grid)

    d12 = d12_rows.pivot(
        index="normalized_progress", columns="seed", values="value_scalar"
    ).loc[grid]
    low = d12.min(axis=1).to_numpy()
    high = d12.max(axis=1).to_numpy()
    center = d12.median(axis=1).to_numpy()

    result = pd.DataFrame(
        {
            "normalized_progress": grid,
            "d12_low": low,
            "d12_median": center,
            "d12_high": high,
        }
    )
    nearest_sides: dict[int, np.ndarray] = {}
    reused_nearest: dict[int, int] = {}
    max_nearest_distance: dict[int, float] = {}

    for depth in (14, 16):
        observed = checkpoints.query("depth == @depth").sort_values("normalized_progress")
        observed_progress = observed["normalized_progress"].to_numpy()
        observed_values = observed["value_scalar"].to_numpy()
        interpolated = np.interp(grid, observed_progress, observed_values)
        result[f"d{depth}"] = interpolated
        result[f"d{depth}_side"] = classify(interpolated, low, high)

        nearest_indices = np.abs(observed_progress[:, None] - grid[None, :]).argmin(axis=0)
        nearest = observed_values[nearest_indices]
        nearest_sides[depth] = classify(nearest, low, high)
        reused_nearest[depth] = len(nearest_indices) - len(set(nearest_indices.tolist()))
        max_nearest_distance[depth] = float(
            np.max(np.abs(observed_progress[nearest_indices] - grid))
        )

    sides14 = result["d14_side"].to_numpy()
    sides16 = result["d16_side"].to_numpy()
    n = len(result)
    counts = {
        depth: result[f"d{depth}_side"].value_counts().reindex(
            ["below", "inside", "above"], fill_value=0
        )
        for depth in (14, 16)
    }
    joint = {
        side: int(np.sum((sides14 == side) & (sides16 == side)))
        for side in ("below", "inside", "above")
    }
    if joint["below"] > n / 2 or joint["above"] > n / 2:
        verdict = "supported"
    elif counts[14]["inside"] > n / 2 and counts[16]["inside"] > n / 2:
        verdict = "refuted"
    else:
        verdict = "inconclusive"

    noninit = result.iloc[1:]
    matched_offsets = {
        depth: float(
            np.median(
                (noninit[f"d{depth}"] - noninit["d12_median"])
                / noninit["d12_median"]
            )
        )
        for depth in (14, 16)
    }
    d12_noninit = d12.iloc[1:]
    seed_sd_relative = float(
        np.median(d12_noninit.std(axis=1, ddof=1) / d12_noninit.median(axis=1))
    )
    seed_range_relative = float(
        np.median(
            (d12_noninit.max(axis=1) - d12_noninit.min(axis=1))
            / d12_noninit.median(axis=1)
        )
    )

    summary = {
        "verdict": verdict,
        "counts": counts,
        "joint": joint,
        "noninit_joint_below": int(
            np.sum(
                (noninit["d14_side"].to_numpy() == "below")
                & (noninit["d16_side"].to_numpy() == "below")
            )
        ),
        "matched_offsets": matched_offsets,
        "seed_sd_relative": seed_sd_relative,
        "seed_range_relative": seed_range_relative,
        "nearest_sides": nearest_sides,
        "nearest_joint": {
            side: int(
                np.sum((nearest_sides[14] == side) & (nearest_sides[16] == side))
            )
            for side in ("below", "inside", "above")
        },
        "reused_nearest": reused_nearest,
        "max_nearest_distance": max_nearest_distance,
    }
    return result, summary


def structural_analysis(rows: pd.DataFrame) -> dict:
    init = rows.query("update_index == 0")
    init_by_run = init.groupby(["depth", "seed"])["value_scalar"].agg(
        matrices="size", zeros=lambda x: int((x == 0).sum()), median="median", mean="mean"
    )
    init_by_role = init.groupby(["depth", "param_role"])["value_scalar"].agg(
        rows="size", zeros=lambda x: int((x == 0).sum()), median="median"
    )

    post = rows.query("update_index != 0")
    run_role = post.groupby(["depth", "seed", "param_role"])["value_scalar"].median()
    role_table = pd.DataFrame(index=ROLES, columns=(12, 14, 16), dtype=float)
    role_table[12] = run_role.loc[12].groupby("param_role").median().reindex(ROLES)
    for depth in (14, 16):
        role_table[depth] = run_role.loc[depth].droplevel("seed").reindex(ROLES)

    per_matrix = (
        post.groupby(["depth", "seed", "param_role", "layer"], as_index=False)[
            "value_scalar"
        ].median()
    )
    d12_matrix = (
        per_matrix.query("depth == 12")
        .groupby(["depth", "param_role", "layer"], as_index=False)["value_scalar"]
        .median()
    )
    matrix_profile = pd.concat(
        [d12_matrix, per_matrix.query("depth > 12").drop(columns="seed")],
        ignore_index=True,
    )
    matrix_profile["relative_layer"] = matrix_profile["layer"] / matrix_profile["depth"]
    matrix_profile["relative_quartile"] = pd.cut(
        matrix_profile["relative_layer"],
        [-1e-9, 0.25, 0.5, 0.75, 1.0000001],
        labels=["Q1", "Q2", "Q3", "Q4"],
    )

    # Compare effects after removing the overall log-median shift of each depth.
    centered_log = np.log(matrix_profile["value_scalar"].to_numpy())
    centered_log -= (
        pd.Series(centered_log)
        .groupby(matrix_profile["depth"].reset_index(drop=True))
        .transform("mean")
        .to_numpy()
    )

    def grouped_r2(keys: list[str]) -> float:
        work = matrix_profile.copy()
        work["centered_log"] = centered_log
        prediction = (
            work.groupby(keys, observed=True)["centered_log"].transform("mean").to_numpy()
        )
        residual_ss = np.square(centered_log - prediction).sum()
        total_ss = np.square(centered_log - centered_log.mean()).sum()
        return float(1 - residual_ss / total_ss)

    r2 = {
        "role": grouped_r2(["param_role"]),
        "relative_quartile": grouped_r2(["relative_quartile"]),
        "role_plus_quartile": grouped_r2(["param_role", "relative_quartile"]),
    }
    binned = matrix_profile.groupby(
        ["depth", "param_role", "relative_quartile"], observed=True
    )["value_scalar"].median().unstack("depth")
    correlations = {}
    for depth in (14, 16):
        pair = binned[[12, depth]].dropna()
        correlations[depth] = {
            "raw": float(np.corrcoef(np.log(pair[12]), np.log(pair[depth]))[0, 1])
        }
        role_centered = np.log(pair)
        role_centered -= role_centered.groupby(level="param_role").transform("mean")
        correlations[depth]["role_centered"] = float(
            np.corrcoef(role_centered[12], role_centered[depth])[0, 1]
        )

    return {
        "init_by_run": init_by_run,
        "init_by_role": init_by_role,
        "role_table": role_table,
        "matrix_profile": matrix_profile,
        "r2": r2,
        "correlations": correlations,
    }


def plot_decision(decision: pd.DataFrame, summary: dict) -> None:
    fig, (ax, counts_ax) = plt.subplots(
        2, 1, figsize=(9.2, 7.4), gridspec_kw={"height_ratios": [4, 1.35]}, constrained_layout=True
    )
    x = decision["normalized_progress"]
    ax.fill_between(
        x,
        100 * decision["d12_low"],
        100 * decision["d12_high"],
        color="#8da0cb",
        alpha=0.35,
        label="d12 five-seed range",
    )
    ax.plot(x, 100 * decision["d12_median"], color="#4c5c99", lw=1.5, label="d12 seed median")
    ax.plot(x, 100 * decision["d14"], color="#d95f02", marker="o", ms=3.2, lw=1.4, label="d14 s7 (interpolated)")
    ax.plot(x, 100 * decision["d16"], color="#1b9e77", marker="s", ms=3.0, lw=1.4, label="d16 s7 (interpolated)")
    ax.set_xscale("symlog", linthresh=0.01, linscale=1.0)
    ax.set_xlabel("Normalized training progress (symlog below 0.01)")
    ax.set_ylabel("Checkpoint median replay update relative error (%)")
    ax.set_title("Frozen decision comparison on the d12 progress grid")
    ax.grid(alpha=0.22)
    ax.legend(ncol=2, fontsize=9)

    order = ["below", "inside", "above"]
    colors = {"below": "#377eb8", "inside": "#bdbdbd", "above": "#e41a1c"}
    left = np.zeros(2)
    y = np.arange(2)
    for side in order:
        values = np.array([summary["counts"][depth][side] for depth in (14, 16)])
        counts_ax.barh(y, values, left=left, color=colors[side], label=side)
        for yi, x0, value in zip(y, left, values):
            if value:
                counts_ax.text(x0 + value / 2, yi, str(value), ha="center", va="center", fontsize=9)
        left += values
    counts_ax.set_yticks(y, ["d14", "d16"])
    counts_ax.set_xlim(0, 30)
    counts_ax.set_xlabel("Matched checkpoints")
    counts_ax.set_title(
        f"Both below together: {summary['joint']['below']}/30; verdict: {summary['verdict'].upper()}"
    )
    counts_ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.83), frameon=False)
    counts_ax.grid(axis="x", alpha=0.2)
    fig.savefig(HERE / "decision_rule.png", dpi=180)
    plt.close(fig)


def plot_structure(structure: dict) -> None:
    profile = structure["matrix_profile"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharex=True, sharey=True, constrained_layout=True)
    for ax, depth in zip(axes, (12, 14, 16)):
        depth_rows = profile.query("depth == @depth")
        for role in ROLES:
            role_rows = depth_rows.query("param_role == @role").sort_values("relative_layer")
            ax.plot(
                role_rows["relative_layer"],
                100 * role_rows["value_scalar"],
                marker="o",
                ms=3.2,
                lw=1.15,
                color=ROLE_COLORS[role],
                label=role,
            )
        suffix = "five-seed median" if depth == 12 else "seed 7"
        ax.set_title(f"d{depth}, {suffix}")
        ax.set_xlabel("Relative layer = zero-based layer / depth")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Per-matrix median after init (%)")
    axes[-1].legend(fontsize=8, ncol=1, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Within-model decoherence structure over post-initialization checkpoints")
    fig.savefig(HERE / "matrix_structure.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def print_results(rows: pd.DataFrame, decision: pd.DataFrame, summary: dict, structure: dict) -> None:
    print("selection")
    print(f"  defined rows: {len(rows):,}")
    print(f"  unique parameter_name channels: {rows['parameter_name'].nunique()}")
    print(
        "  run-matrix series:",
        rows[["segment", "parameter_name"]].drop_duplicates().shape[0],
    )
    print(f"  non-null shape fields: {rows['shape'].notna().sum()}")
    print("\nfrozen decision")
    print("  verdict:", summary["verdict"])
    for depth in (14, 16):
        print(f"  d{depth} counts:", summary["counts"][depth].to_dict())
    print("  joint same-side counts:", summary["joint"])
    print("  joint below without init:", f"{summary['noninit_joint_below']}/29")
    print("  nearest-neighbor joint counts:", summary["nearest_joint"])
    print("  nearest reused checkpoints:", summary["reused_nearest"])
    print("  nearest maximum progress distance:", summary["max_nearest_distance"])
    print(
        "  median matched offsets after init:",
        {depth: f"{100 * value:.4f}%" for depth, value in summary["matched_offsets"].items()},
    )
    print(f"  aggregate d12 seed sd-relative: {100 * summary['seed_sd_relative']:.4f}%")
    print(f"  aggregate d12 seed range-relative: {100 * summary['seed_range_relative']:.4f}%")

    print("\ninitialization zeros")
    print(structure["init_by_run"].to_string())
    print("\ninitialization by role")
    print(structure["init_by_role"].to_string())
    print("\npost-init role medians")
    print((100 * structure["role_table"]).to_string(float_format=lambda x: f"{x:.4f}%"))
    print("\nstructure diagnostics")
    print("  grouped R2:", structure["r2"])
    print("  log-profile correlations:", structure["correlations"])
    print("\ndecision table")
    print(decision.to_string(index=False, float_format=lambda x: f"{x:.8f}"))


def main() -> None:
    rows, _ = load_selected()
    checkpoints = checkpoint_medians(rows)
    decision, decision_summary = decision_analysis(checkpoints)
    structure = structural_analysis(rows)
    assert decision_summary["verdict"] == "supported"
    plot_decision(decision, decision_summary)
    plot_structure(structure)
    print_results(rows, decision, decision_summary, structure)


if __name__ == "__main__":
    main()
