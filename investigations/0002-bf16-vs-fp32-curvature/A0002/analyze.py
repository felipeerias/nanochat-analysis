#!/usr/bin/env python3
"""Paired native-bf16 versus IEEE-fp32-shadow curvature analysis for I0002.

The frozen protocol defines the pairing key as (segment, step, metric).  This
script reads only the sparse parquet tiers of the seven named schema-v3 sweep
segments; it never discovers or reads the legacy d12-iter segment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-i0002-a0002-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
ANALYSIS_ROOT = HERE.parents[2]
DATA_ROOT = ANALYSIS_ROOT.parent / "telemetry-data" / "sweep" / "telemetry-data"
FIGURE_DIR = HERE / "figures"

EXPECTED = {
    "d12-s7": (12, 7),
    "d12-s8": (12, 8),
    "d12-s9": (12, 9),
    "d12-s10": (12, 10),
    "d12-s11": (12, 11),
    "d14-s7": (14, 7),
    "d16-s7": (16, 7),
}
ARMS = ("native", "shadow_fp32")
PAIR_KEY = ["segment", "step", "metric"]
READ_COLUMNS = [
    "metric",
    "step",
    "normalized_progress",
    "value_scalar",
    "is_defined",
    "acceptance_arm",
    "run_id",
    "schema_version",
]


def qstats(values: pd.Series) -> tuple[float, float, float]:
    """Return pandas' linear-interpolation Q1, median, and Q3."""
    x = values.dropna()
    if x.empty:
        return np.nan, np.nan, np.nan
    q = x.quantile([0.25, 0.50, 0.75])
    return float(q.loc[0.25]), float(q.loc[0.50]), float(q.loc[0.75])


def spearman(x: pd.Series, y: pd.Series) -> float:
    """Spearman rank correlation without scipy; ties receive average ranks."""
    keep = x.notna() & y.notna()
    xk = x[keep]
    yk = y[keep]
    if len(xk) < 3 or xk.nunique() < 2 or yk.nunique() < 2:
        return np.nan
    return float(xk.rank(method="average").corr(yk.rank(method="average")))


def locate_segments() -> dict[str, Path]:
    """Resolve only the seven protocol-specified segment prefixes."""
    resolved: dict[str, Path] = {}
    for short_name in EXPECTED:
        matches = sorted(DATA_ROOT.glob(f"{short_name}-s0-*"))
        if len(matches) != 1:
            raise AssertionError(f"Expected one segment for {short_name}, got {matches}")
        if "d12-iter" in matches[0].name:
            raise AssertionError("Legacy d12-iter must never enter this analysis")
        resolved[short_name] = matches[0]
    return resolved


def load_data(segments: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read sparse parquet and return rows plus a segment inventory."""
    frames: list[pd.DataFrame] = []
    inventory: list[dict[str, object]] = []
    for short_name, segment_path in segments.items():
        depth, seed = EXPECTED[short_name]
        files = sorted((segment_path / "sparse").glob("*.parquet"))
        if not files:
            raise AssertionError(f"No sparse parquet files in {segment_path}")
        frame = pd.concat(
            [pq.read_table(path, columns=READ_COLUMNS).to_pandas() for path in files],
            ignore_index=True,
        )
        if set(frame["schema_version"].dropna().astype(str)) != {"3"}:
            raise AssertionError(
                f"Non-v3 data in {segment_path}: {frame['schema_version'].unique()}"
            )
        if set(frame["run_id"].dropna()) != {segment_path.name}:
            raise AssertionError(f"run_id mismatch in {segment_path}")
        frame["segment"] = segment_path.name
        frame["run"] = short_name
        frame["depth"] = depth
        frame["seed"] = seed
        frames.append(frame)
        inventory.append(
            {
                "run": short_name,
                "segment": segment_path.name,
                "depth": depth,
                "seed": seed,
                "sparse_files": len(files),
                "sparse_rows": len(frame),
            }
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(inventory)


def make_pairs(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply the frozen selection and construct one-to-one arm pairs."""
    family = raw["metric"].fillna("").str.startswith(("curvature/", "update/"))
    eligible = raw[
        family
        & raw["acceptance_arm"].isin(ARMS)
        & raw["is_defined"].eq(True)  # explicit defined-row filtering
        & raw["value_scalar"].notna()  # scalar families only
    ].copy()

    metrics_by_arm = {
        arm: set(eligible.loc[eligible["acceptance_arm"].eq(arm), "metric"])
        for arm in ARMS
    }
    shared_metrics = metrics_by_arm["native"] & metrics_by_arm["shadow_fp32"]
    eligible = eligible[eligible["metric"].isin(shared_metrics)]

    duplicate_counts = eligible.groupby(PAIR_KEY + ["acceptance_arm"]).size()
    if not (duplicate_counts == 1).all():
        raise AssertionError("Pair key is not unique within an arm")

    metadata = [
        "normalized_progress",
        "run_id",
        "run",
        "depth",
        "seed",
        "value_scalar",
    ]
    native = eligible.loc[eligible["acceptance_arm"].eq("native"), PAIR_KEY + metadata]
    shadow = eligible.loc[
        eligible["acceptance_arm"].eq("shadow_fp32"), PAIR_KEY + metadata
    ]
    pairs = native.merge(
        shadow,
        on=PAIR_KEY,
        how="inner",
        suffixes=("_native", "_shadow"),
        validate="one_to_one",
    )
    for column in ["normalized_progress", "run_id", "run", "depth", "seed"]:
        left = pairs[f"{column}_native"]
        right = pairs[f"{column}_shadow"]
        if column == "normalized_progress":
            same = np.array_equal(left.to_numpy(), right.to_numpy())
        else:
            same = left.equals(right)
        if not same:
            raise AssertionError(f"Arm metadata disagree for {column}")
        pairs[column] = left

    if set(pairs["metric"]) != shared_metrics:
        missing = sorted(shared_metrics - set(pairs["metric"]))
        raise AssertionError(f"Shared scalar families without pairs: {missing}")
    if not np.isfinite(pairs[["value_scalar_native", "value_scalar_shadow"]]).all().all():
        raise AssertionError("Defined scalar arm values must be finite")

    pairs = pairs[
        PAIR_KEY
        + [
            "run",
            "run_id",
            "depth",
            "seed",
            "normalized_progress",
            "value_scalar_native",
            "value_scalar_shadow",
        ]
    ].copy()
    pairs.rename(
        columns={
            "value_scalar_native": "native",
            "value_scalar_shadow": "shadow",
        },
        inplace=True,
    )
    pairs["absolute_difference"] = (pairs["native"] - pairs["shadow"]).abs()
    pairs["shadow_is_zero"] = pairs["shadow"].eq(0.0)
    nonzero = ~pairs["shadow_is_zero"]
    pairs["signed_relative_difference"] = np.nan
    pairs.loc[nonzero, "signed_relative_difference"] = (
        pairs.loc[nonzero, "native"] - pairs.loc[nonzero, "shadow"]
    ) / pairs.loc[nonzero, "shadow"].abs()
    pairs["absolute_relative_difference"] = pairs[
        "signed_relative_difference"
    ].abs()
    # np.sign treats zero as its own sign, so zero/nonzero pairs disagree.
    pairs["sign_disagreement"] = np.sign(pairs["native"]) != np.sign(pairs["shadow"])

    native_checkpoint_verdict = raw[
        raw["metric"].eq("curvature/native_verdict_code")
        & raw["acceptance_arm"].eq("native")
        & raw["is_defined"].eq(True)
        & raw["value_scalar"].notna()
    ]
    if native_checkpoint_verdict.empty:
        raise AssertionError("Missing native checkpoint verdict rows")
    if not native_checkpoint_verdict["value_scalar"].eq(2.0).all():
        raise AssertionError("Protocol says every native checkpoint verdict failed")

    audit = {
        "curvature_update_rows": int(family.sum()),
        "eligible_defined_scalar_rows": int(len(eligible)),
        "native_scalar_families": len(metrics_by_arm["native"]),
        "shadow_scalar_families": len(metrics_by_arm["shadow_fp32"]),
        "shared_scalar_families": len(shared_metrics),
        "paired_rows": len(pairs),
        "relative_rows": int(nonzero.sum()),
        "shadow_zero_rows": int(pairs["shadow_is_zero"].sum()),
        "native_checkpoint_failed_verdict_rows": int(len(native_checkpoint_verdict)),
    }
    return pairs.sort_values(PAIR_KEY).reset_index(drop=True), audit


def summarize_metrics(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric, group in pairs.groupby("metric", sort=True):
        sr_q1, sr_med, sr_q3 = qstats(group["signed_relative_difference"])
        ar_q1, ar_med, ar_q3 = qstats(group["absolute_relative_difference"])
        ad_q1, ad_med, ad_q3 = qstats(group["absolute_difference"])
        sh_q1, sh_med, sh_q3 = qstats(group["shadow"].abs())
        rows.append(
            {
                "metric": metric,
                "n_pairs": len(group),
                "n_relative": group["signed_relative_difference"].notna().sum(),
                "n_shadow_zero": group["shadow_is_zero"].sum(),
                "signed_relative_q1": sr_q1,
                "signed_relative_median": sr_med,
                "signed_relative_q3": sr_q3,
                "absolute_relative_q1": ar_q1,
                "absolute_relative_median": ar_med,
                "absolute_relative_q3": ar_q3,
                "absolute_difference_q1": ad_q1,
                "absolute_difference_median": ad_med,
                "absolute_difference_q3": ad_q3,
                "shadow_absolute_q1": sh_q1,
                "shadow_absolute_median": sh_med,
                "shadow_absolute_q3": sh_q3,
                "sign_disagreements": group["sign_disagreement"].sum(),
                "sign_disagreement_fraction": group["sign_disagreement"].mean(),
                "exact_agreements": group["absolute_difference"].eq(0.0).sum(),
            }
        )
    return pd.DataFrame(rows)


def summarize_training(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate training trend within runs, then summarize equally by run."""
    run_rows: list[dict[str, object]] = []
    rel = pairs.dropna(subset=["absolute_relative_difference"])
    for (metric, run), group in rel.groupby(["metric", "run"], sort=True):
        early = group.loc[
            group["normalized_progress"].le(0.25), "absolute_relative_difference"
        ]
        late = group.loc[
            group["normalized_progress"].ge(0.75), "absolute_relative_difference"
        ]
        run_rows.append(
            {
                "metric": metric,
                "run": run,
                "depth": int(group["depth"].iloc[0]),
                "seed": int(group["seed"].iloc[0]),
                "n": len(group),
                "spearman_abs_relative_vs_progress": spearman(
                    group["absolute_relative_difference"], group["normalized_progress"]
                ),
                "early_median_abs_relative": early.median() if not early.empty else np.nan,
                "late_median_abs_relative": late.median() if not late.empty else np.nan,
            }
        )
    by_run = pd.DataFrame(run_rows)

    metric_rows: list[dict[str, object]] = []
    for metric, group in by_run.groupby("metric", sort=True):
        rho_q1, rho_med, rho_q3 = qstats(
            group["spearman_abs_relative_vs_progress"]
        )
        early_q1, early_med, early_q3 = qstats(group["early_median_abs_relative"])
        late_q1, late_med, late_q3 = qstats(group["late_median_abs_relative"])
        metric_rows.append(
            {
                "metric": metric,
                "n_runs": group["run"].nunique(),
                "n_nonconstant_run_correlations": group[
                    "spearman_abs_relative_vs_progress"
                ].notna().sum(),
                "spearman_q1_across_runs": rho_q1,
                "spearman_median_across_runs": rho_med,
                "spearman_q3_across_runs": rho_q3,
                "runs_positive_spearman": group[
                    "spearman_abs_relative_vs_progress"
                ].gt(0).sum(),
                "runs_negative_spearman": group[
                    "spearman_abs_relative_vs_progress"
                ].lt(0).sum(),
                "early_q1_across_runs": early_q1,
                "early_median_across_runs": early_med,
                "early_q3_across_runs": early_q3,
                "late_q1_across_runs": late_q1,
                "late_median_across_runs": late_med,
                "late_q3_across_runs": late_q3,
                "late_minus_early": late_med - early_med,
                "late_to_early_ratio": (
                    late_med / early_med if early_med > 0 else np.nan
                ),
            }
        )
    return by_run, pd.DataFrame(metric_rows)


def summarize_depth(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Balance depths using per-depth medians; retain d12 seed sensitivity."""
    rel = pairs.dropna(subset=["absolute_relative_difference"])
    by_run = (
        rel.groupby(["metric", "run", "depth", "seed"], as_index=False)[
            "absolute_relative_difference"
        ]
        .median()
        .rename(columns={"absolute_relative_difference": "run_median_abs_relative"})
    )
    by_depth = (
        rel.groupby(["metric", "depth"], as_index=False)[
            "absolute_relative_difference"
        ]
        .median()
        .rename(columns={"absolute_relative_difference": "depth_median_abs_relative"})
    )
    wide = by_depth.pivot(
        index="metric", columns="depth", values="depth_median_abs_relative"
    )
    wide = wide.rename(columns={12: "d12_median", 14: "d14_median", 16: "d16_median"})
    for column in ["d12_median", "d14_median", "d16_median"]:
        if column not in wide:
            wide[column] = np.nan

    d12_seeds = by_run[by_run["depth"].eq(12)].groupby("metric")[
        "run_median_abs_relative"
    ]
    seed_range = pd.DataFrame(
        {
            "d12_seed_min": d12_seeds.min(),
            "d12_seed_max": d12_seeds.max(),
            "d12_seed_iqr": d12_seeds.quantile(0.75) - d12_seeds.quantile(0.25),
        }
    )
    summary = wide.join(seed_range).reset_index()
    summary["d16_minus_d12"] = summary["d16_median"] - summary["d12_median"]
    summary["d16_to_d12_ratio"] = np.where(
        summary["d12_median"].gt(0),
        summary["d16_median"] / summary["d12_median"],
        np.nan,
    )
    summary["d16_gt_d12"] = summary["d16_median"].gt(summary["d12_median"])
    summary["d16_lt_d12"] = summary["d16_median"].lt(summary["d12_median"])
    summary["strict_monotonic_increase"] = (
        summary["d12_median"].lt(summary["d14_median"])
        & summary["d14_median"].lt(summary["d16_median"])
    )
    summary["strict_monotonic_decrease"] = (
        summary["d12_median"].gt(summary["d14_median"])
        & summary["d14_median"].gt(summary["d16_median"])
    )
    summary["d16_above_all_d12_seeds"] = summary["d16_median"].gt(
        summary["d12_seed_max"]
    )
    return by_run, summary


def plot_metric_summary(summary: pd.DataFrame) -> None:
    plot = summary.sort_values("absolute_relative_median").reset_index(drop=True)
    positives = plot.loc[
        plot[["absolute_relative_q1", "absolute_relative_median", "absolute_relative_q3"]]
        .gt(0)
        .any(axis=1),
        ["absolute_relative_q1", "absolute_relative_median", "absolute_relative_q3"],
    ].to_numpy()
    positives = positives[np.isfinite(positives) & (positives > 0)]
    floor = float(positives.min() / 2) if len(positives) else 1e-12
    y = np.arange(len(plot))
    med = plot["absolute_relative_median"].fillna(0).to_numpy(float)
    q1 = plot["absolute_relative_q1"].fillna(0).to_numpy(float)
    q3 = plot["absolute_relative_q3"].fillna(0).to_numpy(float)
    med_plot = np.maximum(med, floor)
    q1_plot = np.maximum(q1, floor)
    q3_plot = np.maximum(q3, floor)

    fig, (ax_mag, ax_sign) = plt.subplots(
        1, 2, figsize=(14, 18), gridspec_kw={"width_ratios": [2.2, 1.0]}, sharey=True
    )
    ax_mag.errorbar(
        med_plot,
        y,
        xerr=np.vstack([med_plot - q1_plot, q3_plot - med_plot]),
        fmt="o",
        markersize=3.5,
        linewidth=0.8,
        capsize=1.5,
        color="#1f77b4",
    )
    zero = med == 0
    ax_mag.scatter(
        np.full(zero.sum(), floor), y[zero], marker="x", color="black", s=22, zorder=3
    )
    ax_mag.set_xscale("log")
    ax_mag.set_xlabel("absolute relative difference, median [IQR] (log scale)")
    ax_mag.set_yticks(y)
    ax_mag.set_yticklabels(plot["metric"], fontsize=7)
    ax_mag.grid(axis="x", alpha=0.25)
    ax_mag.set_title("Magnitude (x = exact-zero median, shown at floor)")

    ax_sign.barh(
        y,
        plot["sign_disagreement_fraction"],
        color="#d95f02",
        height=0.65,
    )
    ax_sign.set_xlim(0, 1)
    ax_sign.set_xlabel("sign-disagreement fraction")
    ax_sign.grid(axis="x", alpha=0.25)
    ax_sign.set_title("Sign disagreement")
    fig.suptitle("Native bf16 versus IEEE-fp32 shadow: all paired scalar families")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "metric_distortion_and_sign.png", dpi=180)
    plt.close(fig)


def heatmap_values(matrix: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    values = matrix.to_numpy(float)
    positive = values[np.isfinite(values) & (values > 0)]
    floor = float(positive.min() / 2) if len(positive) else 1e-12
    logged = np.log10(np.where(np.isfinite(values), np.maximum(values, floor), np.nan))
    finite = logged[np.isfinite(logged)]
    return logged, float(finite.min()), float(finite.max())


def plot_progress_heatmap(pairs: pd.DataFrame, summary: pd.DataFrame) -> None:
    rel = pairs.dropna(subset=["absolute_relative_difference"]).copy()
    bins = np.linspace(0.0, 1.0, 11)
    rel["progress_bin"] = pd.cut(
        rel["normalized_progress"], bins=bins, include_lowest=True, labels=False
    )
    matrix = rel.pivot_table(
        index="metric",
        columns="progress_bin",
        values="absolute_relative_difference",
        aggfunc="median",
    )
    order = summary.sort_values("absolute_relative_median")["metric"]
    matrix = matrix.reindex(order)
    logged, vmin, vmax = heatmap_values(matrix)
    fig, ax = plt.subplots(figsize=(12, 17))
    image = ax.imshow(logged, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_yticks(np.arange(len(matrix)))
    ax.set_yticklabels(matrix.index, fontsize=7)
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels([f"{bins[i]:.1f}–{bins[i+1]:.1f}" for i in range(10)], rotation=45, ha="right")
    ax.set_xlabel("normalized progress bin")
    ax.set_title("Median absolute relative distortion over training")
    cbar = fig.colorbar(image, ax=ax, pad=0.01)
    cbar.set_label("log10(median absolute relative difference); zeros at display floor")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "distortion_over_progress.png", dpi=180)
    plt.close(fig)


def plot_depth_heatmap(depth: pd.DataFrame, metric_summary: pd.DataFrame) -> None:
    matrix = depth.set_index("metric")[["d12_median", "d14_median", "d16_median"]]
    order = metric_summary.sort_values("absolute_relative_median")["metric"]
    matrix = matrix.reindex(order)
    logged, vmin, vmax = heatmap_values(matrix)
    fig, ax = plt.subplots(figsize=(7, 17))
    image = ax.imshow(logged, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_yticks(np.arange(len(matrix)))
    ax.set_yticklabels(matrix.index, fontsize=7)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["d12", "d14", "d16"])
    ax.set_title("Median absolute relative distortion by recipe size")
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label("log10(median absolute relative difference); zeros at display floor")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "distortion_by_depth.png", dpi=180)
    plt.close(fig)


def build_headline(
    pairs: pd.DataFrame,
    summary: pd.DataFrame,
    training: pd.DataFrame,
    depth: pd.DataFrame,
    audit: dict[str, object],
) -> dict[str, object]:
    sr_q1, sr_med, sr_q3 = qstats(pairs["signed_relative_difference"])
    ar_q1, ar_med, ar_q3 = qstats(pairs["absolute_relative_difference"])
    metric_medians = summary["absolute_relative_median"].dropna()
    nonzero_metrics = summary[summary["absolute_relative_median"].gt(0)]
    trend_valid = training.dropna(subset=["spearman_median_across_runs"])
    depth_ratio_valid = depth.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["d16_to_d12_ratio"]
    )
    top_mag = summary.loc[summary["absolute_relative_median"].idxmax()]
    top_sign = summary.loc[summary["sign_disagreement_fraction"].idxmax()]
    return {
        **audit,
        "pooled_signed_relative_q1": sr_q1,
        "pooled_signed_relative_median": sr_med,
        "pooled_signed_relative_q3": sr_q3,
        "pooled_absolute_relative_q1": ar_q1,
        "pooled_absolute_relative_median": ar_med,
        "pooled_absolute_relative_q3": ar_q3,
        "median_across_metric_medians_abs_relative": float(metric_medians.median()),
        "metrics_with_exact_zero_median_abs_relative": int(
            summary["absolute_relative_median"].eq(0).sum()
        ),
        "metrics_with_nonzero_median_abs_relative": int(len(nonzero_metrics)),
        "pooled_sign_disagreements": int(pairs["sign_disagreement"].sum()),
        "pooled_sign_disagreement_fraction": float(pairs["sign_disagreement"].mean()),
        "metrics_with_any_sign_disagreement": int(
            summary["sign_disagreements"].gt(0).sum()
        ),
        "top_magnitude_metric": str(top_mag["metric"]),
        "top_magnitude_metric_median_abs_relative": float(
            top_mag["absolute_relative_median"]
        ),
        "top_sign_metric": str(top_sign["metric"]),
        "top_sign_metric_fraction": float(top_sign["sign_disagreement_fraction"]),
        "training_metrics_with_nonconstant_within_run_trend": int(len(trend_valid)),
        "training_metrics_positive_median_spearman": int(
            trend_valid["spearman_median_across_runs"].gt(0).sum()
        ),
        "training_metrics_negative_median_spearman": int(
            trend_valid["spearman_median_across_runs"].lt(0).sum()
        ),
        "training_median_of_metric_median_spearman": float(
            trend_valid["spearman_median_across_runs"].median()
        ),
        "training_metrics_late_gt_early": int(
            training["late_minus_early"].gt(0).sum()
        ),
        "training_metrics_late_lt_early": int(
            training["late_minus_early"].lt(0).sum()
        ),
        "depth_metrics_with_finite_d16_d12_ratio": int(len(depth_ratio_valid)),
        "depth_metrics_d16_gt_d12": int(depth["d16_gt_d12"].sum()),
        "depth_metrics_d16_lt_d12": int(depth["d16_lt_d12"].sum()),
        "depth_metrics_d16_equal_d12": int(
            (~depth["d16_gt_d12"] & ~depth["d16_lt_d12"]).sum()
        ),
        "depth_metrics_strict_monotonic_increase": int(
            depth["strict_monotonic_increase"].sum()
        ),
        "depth_metrics_strict_monotonic_decrease": int(
            depth["strict_monotonic_decrease"].sum()
        ),
        "depth_median_d16_to_d12_ratio": float(
            depth_ratio_valid["d16_to_d12_ratio"].median()
        ),
        "depth_metrics_d16_above_all_d12_seeds": int(
            depth["d16_above_all_d12_seeds"].sum()
        ),
    }


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    segments = locate_segments()
    raw, inventory = load_data(segments)
    pairs, audit = make_pairs(raw)
    summary = summarize_metrics(pairs)
    training_by_run, training = summarize_training(pairs)
    depth_by_run, depth = summarize_depth(pairs)

    if len(summary) != audit["shared_scalar_families"]:
        raise AssertionError("Metric summary does not cover the tested universe")

    inventory.to_csv(HERE / "segment_inventory.csv", index=False)
    pairs.to_csv(HERE / "paired_values.csv", index=False, float_format="%.17g")
    summary.to_csv(HERE / "metric_summary.csv", index=False, float_format="%.17g")
    training_by_run.to_csv(
        HERE / "training_trend_by_run.csv", index=False, float_format="%.17g"
    )
    training.to_csv(HERE / "training_trend_summary.csv", index=False, float_format="%.17g")
    depth_by_run.to_csv(
        HERE / "depth_by_run.csv", index=False, float_format="%.17g"
    )
    depth.to_csv(HERE / "depth_summary.csv", index=False, float_format="%.17g")

    plot_metric_summary(summary)
    plot_progress_heatmap(pairs, summary)
    plot_depth_heatmap(depth, summary)
    headline = build_headline(pairs, summary, training, depth, audit)
    (HERE / "headline.json").write_text(
        json.dumps(headline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(headline, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
