#!/usr/bin/env python3
"""I0005/A0002: certified d12 curvature trajectories.

The frozen selection is implemented as an explicit per-run join to the
shadow-fp32 gradient-direction verdict.  In particular, this script does not
use loader.certified(), whose arm-level verdict is intentionally more
pessimistic than the per-direction verdict required by I0005.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/i0005-a0002-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ANALYSIS_ROOT = HERE.parents[2]
DATA_ROOT = ANALYSIS_ROOT.parent / "telemetry-data/sweep/telemetry-data"
FIGURE_DIR = HERE / "figures"
sys.path.insert(0, str(ANALYSIS_ROOT))

from loader.telemetry_load import load_segment  # noqa: E402


SEGMENTS = {
    7: "d12-s7-s0-6c217854f4174ce884dfb2b2dcc13c45",
    8: "d12-s8-s0-2b2e72e4395440029b92226213d137bb",
    9: "d12-s9-s0-b872e698e81b4a8fa0cd08512f43c2c2",
    10: "d12-s10-s0-0ffeb05742bb4154a3f1afc202dc3955",
    11: "d12-s11-s0-5fd1bf807f764954b30658872fc3e2ad",
}

PRIMARY = [
    "curvature/gHg",
    "curvature/eta_star",
    "curvature/dhd",
    "curvature/vhv_gradient",
    "curvature/e_curv_gradient",
]

# Every additional scalar curvature channel that is non-directional or
# gradient-specific.  Direction verdicts are selection gates rather than
# outcome channels; random/update metrics and vector-valued sweeps are outside
# the frozen universe.
AUXILIARY = [
    "curvature/Hg_norm",
    "curvature/arith_eps",
    "curvature/c_fd_gradient",
    "curvature/curv_eps_gradient",
    "curvature/curv_floor_gradient",
    "curvature/curv_snr_gradient",
    "curvature/e_fd_gradient",
    "curvature/e_lin_gradient",
    "curvature/e_sym_gradient",
    "curvature/eta_star_rho",
    "curvature/eta_star_rho_threshold",
    "curvature/fd_conclusive_gradient",
    "curvature/fd_cos_gradient",
    "curvature/fd_eps_gradient",
    "curvature/fd_floor_gradient",
    "curvature/fd_snr_gradient",
    "curvature/gg",
]

ALL_SCALARS = PRIMARY + AUXILIARY
COLORS = dict(zip(SEGMENTS, plt.get_cmap("tab10").colors[: len(SEGMENTS)]))


def select_rows() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return certified scalar rows, all verdict rows, and all eta* rows."""
    certified_parts: list[pd.DataFrame] = []
    verdict_parts: list[pd.DataFrame] = []
    eta_parts: list[pd.DataFrame] = []

    for seed, segment in SEGMENTS.items():
        loaded = load_segment(str(DATA_ROOT), segment)
        sparse = loaded["tiers"]["sparse"]
        shadow = sparse.loc[sparse["acceptance_arm"].eq("shadow_fp32")].copy()

        verdict = shadow.loc[
            shadow["metric"].isin(
                [
                    "curvature/verdict_code_random",
                    "curvature/verdict_code_gradient",
                    "curvature/verdict_code_update",
                ]
            ),
            ["metric", "step", "normalized_progress", "value_scalar", "is_defined"],
        ].copy()
        verdict["seed"] = seed
        assert len(verdict) == 90
        assert verdict["is_defined"].all()
        verdict_parts.append(verdict)

        gradient_verdict = verdict.loc[
            verdict["metric"].eq("curvature/verdict_code_gradient"),
            ["step", "value_scalar"],
        ].rename(columns={"value_scalar": "gradient_verdict"})
        assert len(gradient_verdict) == 30
        passed = gradient_verdict.loc[gradient_verdict["gradient_verdict"].eq(0), ["step"]]

        eta_all = shadow.loc[
            shadow["metric"].eq("curvature/eta_star"),
            [
                "step",
                "normalized_progress",
                "value_scalar",
                "is_defined",
                "undefined_reason",
            ],
        ].copy()
        eta_all["seed"] = seed
        eta_all = eta_all.merge(gradient_verdict, on="step", validate="one_to_one")
        assert len(eta_all) == 30
        eta_parts.append(eta_all)

        outcomes = shadow.loc[
            shadow["metric"].isin(ALL_SCALARS),
            [
                "metric",
                "step",
                "normalized_progress",
                "value_scalar",
                "value_vector",
                "is_defined",
                "undefined_reason",
                "acceptance_arm",
                "estimator_id",
            ],
        ].copy()
        selected = outcomes.merge(passed, on="step", how="inner", validate="many_to_one")
        selected["seed"] = seed

        counts = selected.groupby("metric", observed=True).size()
        expected_passes = len(passed)
        assert counts.index.tolist() == sorted(ALL_SCALARS)
        assert counts.eq(expected_passes).all()
        assert selected["acceptance_arm"].eq("shadow_fp32").all()
        assert selected["value_vector"].isna().all()
        certified_parts.append(selected)

    certified = pd.concat(certified_parts, ignore_index=True)
    verdicts = pd.concat(verdict_parts, ignore_index=True)
    eta_all = pd.concat(eta_parts, ignore_index=True)

    # Undefined rows are counted before this explicit filter.  Passing the
    # gradient verdict happens to imply definedness for all 22 scalar outcomes.
    assert certified["is_defined"].all()
    certified = certified.loc[certified["is_defined"]].copy()
    return certified, verdicts, eta_all


def band_table(certified: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        certified.loc[certified["metric"].eq(metric)]
        .groupby("normalized_progress", as_index=False)["value_scalar"]
        .agg(median="median", minimum="min", maximum="max", n="count")
        .sort_values("normalized_progress")
    )


def all_checkpoint_schedule(verdicts: pd.DataFrame) -> pd.DataFrame:
    grad = verdicts.loc[verdicts["metric"].eq("curvature/verdict_code_gradient")]
    return (
        grad.groupby("normalized_progress", as_index=False)["value_scalar"]
        .agg(n_certified=lambda x: int(x.eq(0).sum()))
        .sort_values("normalized_progress")
    )


def use_log_axis(values: pd.Series) -> bool:
    if not values.gt(0).all():
        return False
    lo, hi = values.min(), values.max()
    return lo > 0 and hi / lo >= 20


def plot_metric(ax: plt.Axes, certified: pd.DataFrame, metric: str) -> None:
    values = certified.loc[certified["metric"].eq(metric)]
    band = band_table(certified, metric)
    x = band["normalized_progress"].to_numpy()
    lo = band["minimum"].to_numpy()
    hi = band["maximum"].to_numpy()
    med = band["median"].to_numpy()

    ax.fill_between(x, lo, hi, color="0.75", alpha=0.35, linewidth=0)
    ax.vlines(x, lo, hi, color="0.55", alpha=0.35, linewidth=0.5)
    ax.plot(x, med, color="black", linestyle="--", linewidth=1.2, label="median")
    for seed in SEGMENTS:
        run = values.loc[values["seed"].eq(seed)].sort_values("normalized_progress")
        ax.plot(
            run["normalized_progress"],
            run["value_scalar"],
            color=COLORS[seed],
            marker="o",
            markersize=2.2,
            linewidth=0.9,
            alpha=0.85,
            label=f"seed {seed}",
        )

    if use_log_axis(values["value_scalar"]):
        ax.set_yscale("log")
    elif values["value_scalar"].nunique() == 1:
        value = values["value_scalar"].iloc[0]
        pad = max(abs(value) * 0.05, 1e-12)
        ax.set_ylim(value - pad, value + pad)
    ax.set_xlim(0, 1.01)
    ax.set_title(metric.removeprefix("curvature/"), fontsize=9)
    ax.grid(True, which="both", alpha=0.18, linewidth=0.5)


def plot_primary(certified: pd.DataFrame, verdicts: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11, 12), constrained_layout=True)
    for ax, metric in zip(axes.flat, PRIMARY):
        plot_metric(ax, certified, metric)
        ax.set_xlabel("normalized_progress")

    count_ax = axes.flat[-1]
    schedule = all_checkpoint_schedule(verdicts)
    count_ax.step(
        schedule["normalized_progress"],
        schedule["n_certified"],
        where="mid",
        color="black",
        linewidth=1.2,
    )
    count_ax.scatter(
        schedule["normalized_progress"], schedule["n_certified"], color="black", s=14
    )
    count_ax.set(xlim=(0, 1.01), ylim=(-0.2, 5.3), xlabel="normalized_progress")
    count_ax.set_yticks(range(6))
    count_ax.set_title("certified seeds at each checkpoint", fontsize=9)
    count_ax.grid(True, alpha=0.18, linewidth=0.5)

    handles = [
        plt.Line2D([], [], color=COLORS[seed], marker="o", markersize=3, label=f"seed {seed}")
        for seed in SEGMENTS
    ]
    handles.extend(
        [
            plt.Line2D([], [], color="black", linestyle="--", label="across-seed median"),
            plt.Rectangle((0, 0), 1, 1, color="0.75", alpha=0.35, label="min–max band"),
        ]
    )
    fig.legend(handles=handles, loc="outside lower center", ncol=7, frameon=False)
    fig.suptitle(
        "Certified shadow-fp32 gradient curvature — five d12 seeds\n"
        "Only per-run verdict_code_gradient == 0 and defined scalar rows",
        fontsize=12,
    )
    fig.savefig(FIGURE_DIR / "primary_trajectories.png", dpi=180)
    plt.close(fig)


def plot_auxiliary(certified: pd.DataFrame) -> None:
    fig, axes = plt.subplots(5, 4, figsize=(14, 15), constrained_layout=True)
    for ax, metric in zip(axes.flat, AUXILIARY):
        plot_metric(ax, certified, metric)
        ax.set_xlabel("progress", fontsize=7)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(AUXILIARY) :]:
        ax.axis("off")

    fig.suptitle(
        "All 17 auxiliary certified scalar channels\n"
        "Seed lines, across-seed median, and checkpoint-wise min–max band",
        fontsize=12,
    )
    fig.savefig(FIGURE_DIR / "auxiliary_scalar_trajectories.png", dpi=180)
    plt.close(fig)


def fmt(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.3e}"
    return f"{value:.4g}"


def print_inventory(verdicts: pd.DataFrame, eta_all: pd.DataFrame) -> None:
    print("# Selection inventory")
    for direction in ["random", "gradient", "update"]:
        rows = verdicts.loc[
            verdicts["metric"].eq(f"curvature/verdict_code_{direction}")
        ]
        counts = rows["value_scalar"].value_counts().sort_index().to_dict()
        print(f"{direction}: {counts}; passed={int(rows['value_scalar'].eq(0).sum())}/150")

    gradient = verdicts.loc[
        verdicts["metric"].eq("curvature/verdict_code_gradient")
    ]
    per_seed = gradient.groupby("seed")["value_scalar"].apply(lambda x: int(x.eq(0).sum()))
    print("gradient passed per seed:", per_seed.to_dict())
    common = (
        gradient.loc[gradient["value_scalar"].eq(0)]
        .groupby("normalized_progress")["seed"]
        .nunique()
        .eq(5)
        .sum()
    )
    print(f"common certified checkpoints: {int(common)}/30")

    eta_bad = eta_all.loc[~eta_all["is_defined"]]
    eta_bad_certified = eta_bad.loc[eta_bad["gradient_verdict"].eq(0)]
    print(
        "eta reliable-sign exclusions:",
        f"raw={len(eta_bad)}/150; certified={len(eta_bad_certified)}/129;",
        "per_seed=",
        eta_bad.groupby("seed").size().reindex(SEGMENTS, fill_value=0).to_dict(),
        "reasons=",
        eta_bad["undefined_reason"].value_counts().to_dict(),
    )


def print_shape_summaries(certified: pd.DataFrame) -> None:
    print("\n# Within-run shape summaries")
    primary = certified.loc[certified["metric"].isin(PRIMARY)]
    n_by_progress = primary.groupby("normalized_progress")["seed"].nunique()
    p0 = float(n_by_progress.loc[n_by_progress.eq(5)].index.min())
    pend = float(n_by_progress.index.max())
    print(f"first common certified progress={p0:.9f}; end={pend:.1f}")

    for metric in PRIMARY:
        rows = primary.loc[primary["metric"].eq(metric)]
        first = rows.loc[rows["normalized_progress"].eq(p0)].set_index("seed")["value_scalar"]
        final = rows.loc[rows["normalized_progress"].eq(pend)].set_index("seed")["value_scalar"]
        ratio = final / first
        print(
            metric,
            "start=", {int(k): fmt(v) for k, v in first.items()},
            "end=", {int(k): fmt(v) for k, v in final.items()},
            "end/start=", {int(k): fmt(v) for k, v in ratio.items()},
        )

    eta = primary.loc[primary["metric"].eq("curvature/eta_star")]
    eta_common = eta.loc[eta["normalized_progress"].ge(p0)]
    peaks = eta_common.loc[eta_common.groupby("seed")["value_scalar"].idxmax()].set_index("seed")
    eta_end = eta_common.loc[eta_common["normalized_progress"].eq(pend)].set_index("seed")
    print(
        "eta common-support peaks:",
        {
            int(seed): {
                "progress": round(row["normalized_progress"], 6),
                "peak": fmt(row["value_scalar"]),
                "end/peak": fmt(
                    eta_end.loc[seed, "value_scalar"] / row["value_scalar"]
                ),
            }
            for seed, row in peaks.iterrows()
        },
    )


def print_endpoint_bands(certified: pd.DataFrame) -> None:
    print("\n# Between-run endpoint bands")
    for progress in [0.006746031746031746, 1.0]:
        print(f"progress={progress:.9f}")
        for metric in PRIMARY:
            values = certified.loc[
                certified["metric"].eq(metric)
                & certified["normalized_progress"].eq(progress),
                "value_scalar",
            ]
            sd_rel = values.std(ddof=1) / values.median()
            print(
                metric,
                f"median={fmt(values.median())}",
                f"band=[{fmt(values.min())}, {fmt(values.max())}]",
                f"n={len(values)}",
                f"sd/median={sd_rel:.1%}",
            )


def print_checkpoint_table(certified: pd.DataFrame, verdicts: pd.DataFrame) -> None:
    print("\n# Primary checkpoint bands (Markdown)")
    print("| progress | certified n | gHg | eta* | dhd | vhv_gradient | e_curv_gradient |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    schedule = all_checkpoint_schedule(verdicts)
    bands = {metric: band_table(certified, metric).set_index("normalized_progress") for metric in PRIMARY}
    for row in schedule.itertuples(index=False):
        cells = []
        for metric in PRIMARY:
            table = bands[metric]
            if row.normalized_progress not in table.index:
                cells.append("—")
                continue
            stat = table.loc[row.normalized_progress]
            cells.append(
                f"{fmt(stat['median'])} [{fmt(stat['minimum'])}, {fmt(stat['maximum'])}]"
            )
        print(
            f"| {row.normalized_progress:.6f} | {row.n_certified} | "
            + " | ".join(cells)
            + " |"
        )


def print_auxiliary_table(certified: pd.DataFrame) -> None:
    print("\n# Auxiliary scalar audit (Markdown)")
    print("| metric | defined certified | overall min–max | first-common median | final median | end/start range | runs ending higher |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    p0 = 0.006746031746031746
    for metric in AUXILIARY:
        rows = certified.loc[certified["metric"].eq(metric)]
        first = rows.loc[rows["normalized_progress"].eq(p0)].set_index("seed")["value_scalar"]
        final = rows.loc[rows["normalized_progress"].eq(1.0)].set_index("seed")["value_scalar"]
        ratio = final / first
        print(
            f"| `{metric}` | {len(rows)} | {fmt(rows['value_scalar'].min())}–{fmt(rows['value_scalar'].max())} "
            f"| {fmt(first.median())} | {fmt(final.median())} "
            f"| {fmt(ratio.min())}–{fmt(ratio.max())} | {int(final.gt(first).sum())}/5 |"
        )


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    certified, verdicts, eta_all = select_rows()
    assert len(ALL_SCALARS) == 22
    assert len(certified) == 22 * 129
    print_inventory(verdicts, eta_all)
    print_shape_summaries(certified)
    print_endpoint_bands(certified)
    print_checkpoint_table(certified, verdicts)
    print_auxiliary_table(certified)
    plot_primary(certified, verdicts)
    plot_auxiliary(certified)
    print(f"\nWrote figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
