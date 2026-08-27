#!/usr/bin/env python3
"""I0006/A0002: quantify fixed-step warmup confounding across d12 and d16.

The frozen protocol asks for scalar metric families present at all three
depths and for two alignments: absolute recipe step and normalized progress.
This implementation deliberately uses d12 observation times as the anchors.

Family definition
-----------------
The unit is (metric, acceptance_arm), with missing arms represented by
"none".  At each observation time, repeated rows within a family (layers,
parameters, probe variants, etc.) are collapsed by their median so models
with different numbers of layers do not receive different row weights.

Alignment
---------
* Continuous and periodic curves are linearly interpolated within d16's
  observed support, never extrapolated.
* Sparse curves are not interpolated.  Absolute matches require an identical
  phase-adjusted recipe step.  Progress matches must be mutual nearest
  neighbors and differ by no more than half of one d12 plus one d16 update in
  normalized-progress units.  This makes missing early deep-checkpoint
  support visible instead of manufacturing it.

The phase-adjusted recipe step is `step` for pre-update rows and `step - 1`
for post-update rows, following DATASET.md's checkpoint convention.

Classification
--------------
At every d12 anchor, the five d12 seeds define a median, min/max band, and
sample standard deviation.  A window's effect is the mean absolute d16-minus-
d12-median deviation.  It is called detectable at a conservative 3x the mean
d12 seed SD (the upper end of I0001's 2--3x rule).  A detectable warm-window
effect is "warmup-dominated" when its relative magnitude is at least twice
the post-window effect.  At least two matched anchors in both windows are
required; otherwise the family is not estimable.

A family is marked unsafe when either requested alignment classifies its
cross-depth contrast as warmup-dominated.  A second, narrower flag records
whether the absolute-vs-progress disagreement is also warmup-dominated; only
that narrower subset directly supports attribution to the fixed-step schedule.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/i0006-a0002-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
from loader.telemetry_load import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    defined,
    load_segment,
)


DATA_ROOT = DEFAULT_DATA_ROOT
WARMUP_END = 400
D12_STEPS = 2520
D16_STEPS = 5376
D12_WARMDOWN_START = 882
PROGRESS_TOL = 0.5 * (1.0 / D12_STEPS + 1.0 / D16_STEPS)
MIN_WINDOW_POINTS = 2
SEED_MULTIPLE = 3.0
DOMINANCE_RATIO = 2.0


def segment_identity(name: str) -> tuple[str, int, int]:
    match = re.match(r"^(d(\d+))-s(\d+)-s0-", name)
    if not match:
        raise ValueError(f"unexpected segment name: {name}")
    return match.group(1), int(match.group(2)), int(match.group(3))


def family_label(metric: str, arm: str) -> str:
    return metric if arm == "none" else f"{metric} [{arm}]"


def load_curves() -> tuple[pd.DataFrame, dict[str, dict], list[str]]:
    segments = sorted(
        p.name
        for p in DATA_ROOT.iterdir()
        if p.is_dir() and not p.name.startswith("d12-iter-")
    )
    if len(segments) != 7:
        raise AssertionError(f"expected seven schema-v3 segments, got {segments}")

    parts: list[pd.DataFrame] = []
    provenance: dict[str, dict] = {}
    family_sets: list[set[tuple[str, str]]] = []

    for segment in segments:
        depth_label, depth, seed = segment_identity(segment)
        loaded = load_segment(str(DATA_ROOT), segment)
        provenance[segment] = loaded["provenance"]
        if depth == 12 and loaded["provenance"]["deep_schedule_landmarks"][-1] != D12_WARMDOWN_START:
            raise AssertionError("unexpected d12 warmdown landmark")
        run_parts: list[pd.DataFrame] = []

        for tier, raw in loaded["tiers"].items():
            if set(raw["schema_version"].dropna().astype(int).unique()) != {3}:
                raise AssertionError(f"non-v3 rows in {segment}/{tier}")
            q = defined(raw)
            q = q[q["value_scalar"].notna()].copy()
            values = q["value_scalar"].astype(float).to_numpy()
            if not np.isfinite(values).all():
                raise AssertionError(f"defined non-finite scalar in {segment}/{tier}")
            q["arm"] = q["acceptance_arm"].fillna("none").astype(str)
            q["tier_source"] = tier
            q["recipe_step"] = q["step"].astype(int)
            q.loc[q["phase"] == "post_update", "recipe_step"] -= 1
            run_parts.append(
                q[
                    [
                        "metric",
                        "arm",
                        "tier_source",
                        "phase",
                        "recipe_step",
                        "normalized_progress",
                        "value_scalar",
                    ]
                ]
            )

        run = pd.concat(run_parts, ignore_index=True)
        family_sets.append(set(map(tuple, run[["metric", "arm"]].drop_duplicates().to_numpy())))

        # A metric-by-arm family must have one tier and phase.  Keeping these
        # semantics fixed prevents accidental merging of incompatible clocks.
        clocks = run.groupby(["metric", "arm"], dropna=False).agg(
            n_tiers=("tier_source", "nunique"), n_phases=("phase", "nunique")
        )
        if (clocks[["n_tiers", "n_phases"]] > 1).any().any():
            raise AssertionError(f"family spans clocks in {segment}")

        summary = (
            run.groupby(
                [
                    "metric",
                    "arm",
                    "tier_source",
                    "phase",
                    "recipe_step",
                    "normalized_progress",
                ],
                dropna=False,
                as_index=False,
            )["value_scalar"]
            .median()
            .rename(columns={"value_scalar": "value"})
        )
        summary["segment"] = segment
        summary["run"] = f"{depth_label}-s{seed}"
        summary["depth"] = depth
        summary["seed"] = seed
        parts.append(summary)

    universe = set.intersection(*family_sets)
    union = set.union(*family_sets)
    if len(universe) != 263 or len(union) != 265:
        raise AssertionError(f"unexpected family counts: common={len(universe)}, union={len(union)}")

    curves = pd.concat(parts, ignore_index=True)
    keep = pd.MultiIndex.from_frame(curves[["metric", "arm"]]).isin(
        pd.MultiIndex.from_tuples(sorted(universe), names=["metric", "arm"])
    )
    curves = curves[keep].copy()
    curves["family"] = [family_label(m, a) for m, a in curves[["metric", "arm"]].itertuples(index=False)]
    return curves, provenance, segments


def reference_curve(d12: pd.DataFrame) -> pd.DataFrame:
    expected = [f"d12-s{s}" for s in (7, 8, 9, 10, 11)]
    wide = d12.pivot_table(index="recipe_step", columns="run", values="value", aggfunc="median")
    for col in expected:
        if col not in wide:
            wide[col] = np.nan
    wide = wide[expected]

    meta = d12.groupby("recipe_step", as_index=True).agg(
        normalized_progress=("normalized_progress", "median")
    )
    out = meta.join(wide)
    out["n_seed"] = out[expected].notna().sum(axis=1)
    out["center"] = out[expected].median(axis=1, skipna=True)
    out["seed_sd"] = out[expected].std(axis=1, ddof=1, skipna=True)
    out["band_min"] = out[expected].min(axis=1, skipna=True)
    out["band_max"] = out[expected].max(axis=1, skipna=True)
    out = out[out["n_seed"] >= 3].reset_index()
    return out


def collapse_xy(frame: pd.DataFrame, axis: str) -> tuple[np.ndarray, np.ndarray]:
    q = frame[[axis, "value"]].dropna().groupby(axis, as_index=False)["value"].median()
    q = q.sort_values(axis)
    return q[axis].to_numpy(float), q["value"].to_numpy(float)


def interpolate_within(x: np.ndarray, y: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.full(len(target), np.nan)
    mismatch = np.full(len(target), np.nan)
    if len(x) == 0:
        return values, mismatch
    if len(x) == 1:
        exact = np.isclose(target, x[0], rtol=0.0, atol=1e-15)
        values[exact] = y[0]
        mismatch[exact] = 0.0
        return values, mismatch
    inside = (target >= x[0]) & (target <= x[-1])
    values[inside] = np.interp(target[inside], x, y)
    idx = np.searchsorted(x, target[inside])
    idx = np.clip(idx, 1, len(x) - 1)
    left = np.abs(target[inside] - x[idx - 1])
    right = np.abs(x[idx] - target[inside])
    mismatch[inside] = np.minimum(left, right)
    return values, mismatch


def sparse_direct_match(
    x: np.ndarray, y: np.ndarray, target: np.ndarray, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Mutual-nearest direct matches, without interpolation or reuse."""
    values = np.full(len(target), np.nan)
    mismatch = np.full(len(target), np.nan)
    if len(x) == 0 or len(target) == 0:
        return values, mismatch
    for i, t in enumerate(target):
        j = int(np.argmin(np.abs(x - t)))
        distance = float(abs(x[j] - t))
        reverse = int(np.argmin(np.abs(target - x[j])))
        if reverse == i and distance <= tolerance:
            values[i] = y[j]
            mismatch[i] = distance
    return values, mismatch


def aligned_comparison(
    ref: pd.DataFrame, d16: pd.DataFrame, tier: str, alignment: str
) -> pd.DataFrame:
    if alignment == "absolute":
        axis = "recipe_step"
        target = ref[axis].to_numpy(float)
    elif alignment == "progress":
        axis = "normalized_progress"
        target = ref[axis].to_numpy(float)
    else:
        raise ValueError(alignment)

    x, y = collapse_xy(d16, axis)
    if tier == "sparse":
        tolerance = 1e-12 if alignment == "absolute" else PROGRESS_TOL
        values, mismatch = sparse_direct_match(x, y, target, tolerance)
        method = "direct-mutual-nearest"
    else:
        values, mismatch = interpolate_within(x, y, target)
        method = "linear-within-support"

    out = ref.copy()
    out["d16"] = values
    out["axis_mismatch"] = mismatch
    out["alignment"] = alignment
    out["method"] = method
    out = out[out["d16"].notna()].copy()
    out["difference"] = out["d16"] - out["center"]
    out["outside_band"] = (out["d16"] < out["band_min"]) | (out["d16"] > out["band_max"])
    out["window"] = np.where(out["recipe_step"] <= WARMUP_END, "warm", "post")
    return out


def window_stats(frame: pd.DataFrame, window: str, value_col: str = "difference") -> dict[str, float]:
    q = frame[frame["window"] == window].copy()
    n = len(q)
    if n == 0:
        return {
            "n": 0,
            "effect": math.nan,
            "effect_rel": math.nan,
            "seed_sd": math.nan,
            "seed_rel": math.nan,
            "seed_multiple": math.nan,
            "outside_fraction": math.nan,
            "signed_rel": math.nan,
            "baseline": math.nan,
            "detectable": False,
        }
    effect = float(q[value_col].abs().mean())
    baseline = float(q["center"].abs().mean())
    seed_sd = float(q["seed_sd"].fillna(0.0).mean())
    scale = max(1.0, baseline, float(q.get("d16", pd.Series([0.0])).abs().mean()))
    numeric_tol = 1e-12 * scale
    effect_rel = effect / baseline if baseline > numeric_tol else math.nan
    seed_rel = seed_sd / baseline if baseline > numeric_tol else math.nan
    if seed_sd > numeric_tol:
        seed_multiple = effect / seed_sd
    elif effect > numeric_tol:
        seed_multiple = math.inf
    else:
        seed_multiple = 0.0
    detectable = n >= MIN_WINDOW_POINTS and effect > max(SEED_MULTIPLE * seed_sd, numeric_tol)
    outside = float(q["outside_band"].mean()) if "outside_band" in q else math.nan
    signed_rel = float(q[value_col].mean() / baseline) if baseline > numeric_tol else math.nan
    return {
        "n": n,
        "effect": effect,
        "effect_rel": effect_rel,
        "seed_sd": seed_sd,
        "seed_rel": seed_rel,
        "seed_multiple": seed_multiple,
        "outside_fraction": outside,
        "signed_rel": signed_rel,
        "baseline": baseline,
        "detectable": bool(detectable),
    }


def interval_stats(
    frame: pd.DataFrame,
    lower_exclusive: int,
    upper_inclusive: int,
    value_col: str = "difference",
) -> dict[str, float]:
    q = frame[
        (frame["recipe_step"] > lower_exclusive)
        & (frame["recipe_step"] <= upper_inclusive)
    ].copy()
    q["window"] = "interval"
    return window_stats(q, "interval", value_col=value_col)


def magnitude(stats: dict[str, float]) -> float:
    return stats["effect_rel"] if np.isfinite(stats["effect_rel"]) else stats["effect"]


def classify(warm: dict[str, float], post: dict[str, float]) -> tuple[str, float]:
    if warm["n"] < MIN_WINDOW_POINTS or post["n"] < MIN_WINDOW_POINTS:
        return "not estimable", math.nan
    w = magnitude(warm)
    p = magnitude(post)
    dominance = math.inf if p == 0 and w > 0 else (w / p if p > 0 else 1.0)
    if warm["detectable"] and dominance >= DOMINANCE_RATIO:
        return "warmup-dominated", dominance
    if warm["detectable"] or post["detectable"]:
        return "uniformly different", dominance
    return "not different", dominance


def prefixed(stats: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in stats.items()}


def analyze_family(family: str, frame: pd.DataFrame) -> tuple[dict, dict[str, pd.DataFrame]]:
    metric = str(frame["metric"].iloc[0])
    arm = str(frame["arm"].iloc[0])
    tier = str(frame["tier_source"].iloc[0])
    phase = str(frame["phase"].iloc[0])
    d12 = frame[frame["depth"] == 12]
    d16 = frame[(frame["depth"] == 16) & (frame["seed"] == 7)]
    ref = reference_curve(d12)
    absolute = aligned_comparison(ref, d16, tier, "absolute")
    progress = aligned_comparison(ref, d16, tier, "progress")

    aw, ap = window_stats(absolute, "warm"), window_stats(absolute, "post")
    pw, pp = window_stats(progress, "warm"), window_stats(progress, "post")
    abs_class, abs_dom = classify(aw, ap)
    prog_class, prog_dom = classify(pw, pp)

    # Alignment disagreement is evaluated only where both alignments produced
    # a d16 estimate for the same d12 anchor.
    common = absolute[
        [
            "recipe_step",
            "normalized_progress",
            "center",
            "seed_sd",
            "band_min",
            "band_max",
            "window",
            "d16",
        ]
    ].merge(
        progress[["recipe_step", "d16"]],
        on="recipe_step",
        how="inner",
        suffixes=("_absolute", "_progress"),
    )
    common["alignment_difference"] = common["d16_absolute"] - common["d16_progress"]
    # `window_stats` expects these columns even though out-of-band is not a
    # meaningful concept for an alignment-vs-alignment difference.
    common["difference"] = common["alignment_difference"]
    common["d16"] = common["d16_absolute"]
    common["outside_band"] = False
    dw, dp = window_stats(common, "warm", value_col="alignment_difference"), window_stats(
        common, "post", value_col="alignment_difference"
    )
    disagreement_class, disagreement_dom = classify(dw, dp)

    # Sensitivity: the frozen post window includes the recipe's normalized
    # warmdown, which begins at d12 step 882.  Comparing warmup only with the
    # post-ramp/pre-warmdown interval (400, 882] isolates the stable plateau.
    am = interval_stats(absolute, WARMUP_END, D12_WARMDOWN_START)
    pm = interval_stats(progress, WARMUP_END, D12_WARMDOWN_START)
    dm = interval_stats(common, WARMUP_END, D12_WARMDOWN_START, value_col="alignment_difference")
    abs_mid_class, abs_mid_dom = classify(aw, am)
    prog_mid_class, prog_mid_dom = classify(pw, pm)
    disagreement_mid_class, disagreement_mid_dom = classify(dw, dm)

    unsafe = abs_class == "warmup-dominated" or prog_class == "warmup-dominated"
    schedule_supported = unsafe and disagreement_class == "warmup-dominated"
    prewarmdown_unsafe = abs_mid_class == "warmup-dominated" or prog_mid_class == "warmup-dominated"
    prewarmdown_supported = prewarmdown_unsafe and disagreement_mid_class == "warmup-dominated"

    result = {
        "family": family,
        "metric": metric,
        "arm": arm,
        "tier": tier,
        "phase": phase,
        "absolute_class": abs_class,
        "absolute_dominance": abs_dom,
        "progress_class": prog_class,
        "progress_dominance": prog_dom,
        "alignment_class": disagreement_class,
        "alignment_dominance": disagreement_dom,
        "unsafe_for_depth_claims": bool(unsafe),
        "alignment_supports_schedule_attribution": bool(schedule_supported),
        "absolute_prewarmdown_class": abs_mid_class,
        "absolute_prewarmdown_dominance": abs_mid_dom,
        "progress_prewarmdown_class": prog_mid_class,
        "progress_prewarmdown_dominance": prog_mid_dom,
        "alignment_prewarmdown_class": disagreement_mid_class,
        "alignment_prewarmdown_dominance": disagreement_mid_dom,
        "prewarmdown_unsafe": bool(prewarmdown_unsafe),
        "prewarmdown_alignment_support": bool(prewarmdown_supported),
    }
    result.update(prefixed(aw, "absolute_warm"))
    result.update(prefixed(ap, "absolute_post"))
    result.update(prefixed(pw, "progress_warm"))
    result.update(prefixed(pp, "progress_post"))
    result.update(prefixed(dw, "alignment_warm"))
    result.update(prefixed(dp, "alignment_post"))
    result.update(prefixed(am, "absolute_prewarmdown"))
    result.update(prefixed(pm, "progress_prewarmdown"))
    result.update(prefixed(dm, "alignment_prewarmdown"))
    return result, {"absolute": absolute, "progress": progress, "disagreement": common}


def mutual_schedule_matches(phase: str) -> pd.DataFrame:
    d12_raw = np.array([0, 1, 2, 4, 8, 16, 32, 40, 64, 126, 252, 378, 400, 504, 630, 756,
                        882, 1008, 1134, 1260, 1385, 1511, 1637, 1763, 1889, 2015, 2141,
                        2267, 2393, 2519], dtype=float)
    d16_raw = np.array([0, 1, 2, 4, 8, 16, 32, 40, 64, 128, 256, 269, 400, 538, 806, 1075,
                        1344, 1613, 1881, 1882, 2150, 2419, 2688, 2956, 3225, 3494, 3763,
                        4031, 4300, 4569, 4838, 5106, 5375], dtype=float)
    offset = 1.0 if phase == "post_update" else 0.0
    p12 = (d12_raw + offset) / D12_STEPS
    p16 = (d16_raw + offset) / D16_STEPS
    rows = []
    for i, p in enumerate(p12):
        j = int(np.argmin(np.abs(p16 - p)))
        reverse = int(np.argmin(np.abs(p12 - p16[j])))
        distance = float(abs(p16[j] - p))
        if reverse == i and distance <= PROGRESS_TOL:
            rows.append(
                {
                    "phase": phase,
                    "d12_checkpoint": int(d12_raw[i]),
                    "d16_checkpoint": int(d16_raw[j]),
                    "d12_progress": p,
                    "d16_progress": p16[j],
                    "progress_mismatch": distance,
                    "d12_window": "warm" if d12_raw[i] <= WARMUP_END else "post",
                }
            )
    return pd.DataFrame(rows)


def safe_log(values: pd.Series) -> np.ndarray:
    arr = values.astype(float).to_numpy()
    positive = arr[np.isfinite(arr) & (arr > 0)]
    floor = positive.min() / 3 if len(positive) else 1e-12
    return np.log10(np.where(np.isfinite(arr) & (arr > 0), arr, floor))


def make_figures(results: pd.DataFrame, sparse_matches: pd.DataFrame) -> None:
    colors = {
        "warmup-dominated": "#d95f02",
        "uniformly different": "#1b9e77",
        "not different": "#7570b3",
        "not estimable": "#bdbdbd",
    }
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for cls, q in results.groupby("progress_class"):
        ax.scatter(
            safe_log(q["progress_post_effect_rel"]),
            safe_log(q["progress_warm_effect_rel"]),
            s=20,
            alpha=0.72,
            label=f"{cls} (n={len(q)})",
            color=colors[cls],
        )
    lo, hi = ax.get_xlim()[0], ax.get_xlim()[1]
    lo = min(lo, ax.get_ylim()[0])
    hi = max(hi, ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--", label="warm = post")
    ax.plot([lo, hi], [lo + math.log10(2), hi + math.log10(2)], color="black", lw=1, ls=":",
            label="warm = 2× post")
    ax.set_xlabel("log10 mean absolute d12–d16 difference / |d12|, post")
    ax.set_ylabel("log10 mean absolute d12–d16 difference / |d12|, step ≤ 400")
    ax.set_title("Normalized-progress alignment: warm-window concentration")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(HERE / "warm_vs_post_progress.png", dpi=180)
    plt.close(fig)

    q = results[results["alignment_class"] == "warmup-dominated"].copy()
    q = q[np.isfinite(q["alignment_warm_effect_rel"])].nlargest(24, "alignment_warm_effect_rel")
    fig_height = max(4.5, 0.29 * max(len(q), 1) + 1.5)
    fig, ax = plt.subplots(figsize=(9.2, fig_height))
    if len(q):
        y = np.arange(len(q))
        ax.barh(
            y,
            100 * q["alignment_warm_effect_rel"],
            color=np.where(q["alignment_supports_schedule_attribution"], "#d95f02", "#7570b3"),
        )
        ax.set_yticks(y, q["family"], fontsize=7)
        ax.invert_yaxis()
    ax.set_xlabel("mean |d16(abs step) − d16(same progress)| / mean |d12| (%)")
    ax.set_title("Largest warmup-dominated alignment disagreements\n(orange also has a warmup-dominated cross-depth contrast)")
    fig.tight_layout()
    fig.savefig(HERE / "alignment_disagreement.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, phase in zip(axes, ("pre_update", "post_update")):
        q = sparse_matches[sparse_matches["phase"] == phase]
        ax.scatter(q["d12_progress"], q["d16_progress"], c=np.where(q["d12_window"] == "warm", "#d95f02", "#1b9e77"), s=27)
        ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--")
        ax.set_title(f"{phase}: {len(q)} mutual matches")
        ax.set_xlabel("d12 normalized progress")
    axes[0].set_ylabel("matched d16 normalized progress")
    fig.suptitle(f"Direct sparse matches (tolerance {PROGRESS_TOL:.6f})")
    fig.tight_layout()
    fig.savefig(HERE / "sparse_progress_matches.png", dpi=180)
    plt.close(fig)


def sensitivity_counts(table: pd.DataFrame) -> pd.DataFrame:
    """Vary the two disclosed decision thresholds without re-reading data."""

    def alternate_class(row: pd.Series, prefix: str, seed_multiple: float, dominance_ratio: float) -> str:
        if row[f"{prefix}_warm_n"] < MIN_WINDOW_POINTS or row[f"{prefix}_post_n"] < MIN_WINDOW_POINTS:
            return "not estimable"
        warm_detect = row[f"{prefix}_warm_seed_multiple"] > seed_multiple
        post_detect = row[f"{prefix}_post_seed_multiple"] > seed_multiple
        warm = row[f"{prefix}_warm_effect_rel"]
        post = row[f"{prefix}_post_effect_rel"]
        if not np.isfinite(warm):
            warm = row[f"{prefix}_warm_effect"]
        if not np.isfinite(post):
            post = row[f"{prefix}_post_effect"]
        dominance = math.inf if post == 0 and warm > 0 else (warm / post if post > 0 else 1.0)
        if warm_detect and dominance >= dominance_ratio:
            return "warmup-dominated"
        if warm_detect or post_detect:
            return "uniformly different"
        return "not different"

    rows = []
    for seed_multiple in (2.0, 3.0):
        for dominance_ratio in (1.5, 2.0, 3.0):
            classifications = {
                prefix: table.apply(
                    alternate_class,
                    axis=1,
                    args=(prefix, seed_multiple, dominance_ratio),
                )
                for prefix in ("absolute", "progress", "alignment")
            }
            abs_set = set(table.loc[classifications["absolute"] == "warmup-dominated", "family"])
            progress_set = set(table.loc[classifications["progress"] == "warmup-dominated", "family"])
            alignment_set = set(table.loc[classifications["alignment"] == "warmup-dominated", "family"])
            unsafe = abs_set | progress_set
            rows.append(
                {
                    "seed_multiple": seed_multiple,
                    "dominance_ratio": dominance_ratio,
                    "absolute_warmup_dominated": len(abs_set),
                    "progress_warmup_dominated": len(progress_set),
                    "unsafe_union": len(unsafe),
                    "alignment_supported": len(unsafe & alignment_set),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    curves, provenance, segments = load_curves()
    results = []
    for family, frame in curves.groupby("family", sort=True):
        result, _comparisons = analyze_family(family, frame)
        results.append(result)

    table = pd.DataFrame(results).sort_values("family").reset_index(drop=True)
    table.to_csv(HERE / "family_results.csv", index=False, float_format="%.10g")
    sensitivity = sensitivity_counts(table)
    sensitivity.to_csv(HERE / "sensitivity.csv", index=False, float_format="%.10g")

    sparse_matches = pd.concat(
        [mutual_schedule_matches("pre_update"), mutual_schedule_matches("post_update")],
        ignore_index=True,
    )
    sparse_matches.to_csv(HERE / "sparse_progress_matches.csv", index=False, float_format="%.10g")
    make_figures(table, sparse_matches)

    excluded = sorted(
        set(
            family_label(m, a)
            for m, a in curves[["metric", "arm"]].drop_duplicates().itertuples(index=False)
        )
    )
    del excluded  # universe is already fully represented by `table`

    summary = {
        "segments": segments,
        "n_families": len(table),
        "class_counts_absolute": dict(Counter(table["absolute_class"])),
        "class_counts_progress": dict(Counter(table["progress_class"])),
        "class_counts_alignment": dict(Counter(table["alignment_class"])),
        "n_unsafe": int(table["unsafe_for_depth_claims"].sum()),
        "unsafe_families": table.loc[table["unsafe_for_depth_claims"], "family"].tolist(),
        "n_alignment_supported": int(table["alignment_supports_schedule_attribution"].sum()),
        "alignment_supported_families": table.loc[
            table["alignment_supports_schedule_attribution"], "family"
        ].tolist(),
        "prewarmdown_sensitivity": {
            "d12_interval": f"({WARMUP_END}, {D12_WARMDOWN_START}]",
            "n_unsafe": int(table["prewarmdown_unsafe"].sum()),
            "n_alignment_supported": int(table["prewarmdown_alignment_support"].sum()),
            "unsafe_families": table.loc[table["prewarmdown_unsafe"], "family"].tolist(),
        },
        "progress_tolerance": PROGRESS_TOL,
        "sparse_matches": {
            phase: {
                "all": int(len(q)),
                "warm": int((q["d12_window"] == "warm").sum()),
                "post": int((q["d12_window"] == "post").sum()),
            }
            for phase, q in sparse_matches.groupby("phase")
        },
        "parameters": {
            "warmup_end": WARMUP_END,
            "seed_multiple": SEED_MULTIPLE,
            "dominance_ratio": DOMINANCE_RATIO,
            "minimum_points_per_window": MIN_WINDOW_POINTS,
        },
        "sensitivity": sensitivity.to_dict(orient="records"),
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
