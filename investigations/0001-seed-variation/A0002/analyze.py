#!/usr/bin/env python3
"""Blind seed-spread analysis for I0001/A0002.

Reads only the five d12 parquet segments.  A comparison element is one scalar
or one vector component from the same semantic channel at exactly matching
normalized progress in all five runs.  The cross-seed spread is max - min;
the relative spread is that range divided by abs(the five-seed median).

The script writes the machine-readable ranking, a Markdown rendering of the
complete table, and a compact JSON summary into this directory.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from loader.paths import DEFAULT_DATA_ROOT


HERE = Path(__file__).resolve().parent
DATA_ROOT = DEFAULT_DATA_ROOT
SEGMENTS = {
    "d12-s7": "d12-s7-s0-6c217854f4174ce884dfb2b2dcc13c45",
    "d12-s8": "d12-s8-s0-2b2e72e4395440029b92226213d137bb",
    "d12-s9": "d12-s9-s0-b872e698e81b4a8fa0cd08512f43c2c2",
    "d12-s10": "d12-s10-s0-0ffeb05742bb4154a3f1afc202dc3955",
    "d12-s11": "d12-s11-s0-5fd1bf807f764954b30658872fc3e2ad",
}
RUNS = tuple(SEGMENTS)
TIERS = ("continuous", "periodic", "sparse")
N_STEPS = 2520
EARLY_CUTOFF = 400 / N_STEPS

# Columns sufficient to construct stable cross-seed channels. Hash-like
# probe_id values are shared across these runs and relabelled semantically.
READ_COLUMNS = [
    "metric",
    "tier",
    "phase",
    "aggregation",
    "step",
    "normalized_progress",
    "batch_unit",
    "value_scalar",
    "value_vector",
    "is_defined",
    "undefined_reason",
    "param_role",
    "parameter_name",
    "shape",
    "layer",
    "head",
    "segment_id",
    "probe_id",
    "checkpoint_id",
    "optimizer_kind",
    "optimizer_group_id",
    "estimator_id",
    "sample_count",
    "sketch_seed",
    "parameter_schema_hash",
    "units",
    "dtype",
    "backend",
    "acceptance_status",
    "acceptance_arm",
    "tolerance_version",
]

CHANNEL_COLUMNS = [
    "phase",
    "aggregation",
    "batch_unit",
    "param_role",
    "parameter_name",
    "shape",
    "layer",
    "head",
    "segment_id",
    "probe_label",
    "checkpoint_id",
    "optimizer_kind",
    "optimizer_group_id",
    "estimator_id",
    "sample_count",
    "sketch_seed",
    "parameter_schema_hash",
    "units",
    "dtype",
    "backend",
    "tolerance_version",
]


def load_tier(run: str, tier: str) -> pd.DataFrame:
    path = DATA_ROOT / SEGMENTS[run] / tier
    return ds.dataset(path, format="parquet").to_table(columns=READ_COLUMNS).to_pandas()


def norm_key_value(value: Any) -> Any:
    """Convert pandas/NumPy values into deterministic hashable key values."""
    if value is None:
        return "<NA>"
    try:
        if pd.isna(value):
            return "<NA>"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return int(f) if f.is_integer() else f
    return str(value)


def make_probe_maps(periodic: dict[str, pd.DataFrame]) -> dict[str, dict[str, str]]:
    """Map opaque probe hashes to the stable train_stream/val labels.

    The profiles identify the lower final probe loss as train_stream and the
    higher as val in every run.  Every other non-null probe hash is the short
    probe used by attention/deep-checkpoint measurements.
    """
    maps: dict[str, dict[str, str]] = {}
    for run, frame in periodic.items():
        rows = frame[(frame["metric"] == "probe/loss") & frame["is_defined"]]
        finals = (
            rows.sort_values("normalized_progress")
            .groupby("probe_id", dropna=False)
            .tail(1)
            .sort_values("value_scalar")
        )
        if len(finals) != 2:
            raise AssertionError(f"{run}: expected two probe/loss probes, got {len(finals)}")
        maps[run] = {
            str(finals.iloc[0]["probe_id"]): "train_stream",
            str(finals.iloc[1]["probe_id"]): "val",
        }
    return maps


def add_probe_labels(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    frame = frame.copy()

    def label(value: Any) -> str:
        if pd.isna(value):
            return "<none>"
        return mapping.get(str(value), "short")

    frame["probe_label"] = frame["probe_id"].map(label)
    return frame


def direction_for_metric(metric: str) -> str | None:
    """Return the per-direction verdict governing a curvature metric."""
    for direction in ("gradient", "random", "update"):
        if metric.endswith(f"_{direction}"):
            return direction
    if metric in {
        "curvature/Hg_norm",
        "curvature/gHg",
        "curvature/gg",
        "curvature/eta_star",
        "curvature/eta_star_rho",
        "curvature/eta_star_rho_threshold",
    }:
        return "gradient"
    if metric == "curvature/dhd":
        return "update"
    return None


def build_pass_steps(sparse: dict[str, pd.DataFrame]) -> dict[str, dict[str, dict[str, set[int]]]]:
    out: dict[str, dict[str, dict[str, set[int]]]] = {}
    for run, frame in sparse.items():
        out[run] = {}
        for arm in ("native", "shadow_fp32"):
            out[run][arm] = {}
            for direction in ("gradient", "random", "update"):
                metric = f"curvature/verdict_code_{direction}"
                rows = frame[
                    (frame["metric"] == metric)
                    & (frame["acceptance_arm"] == arm)
                    & frame["is_defined"]
                    & (frame["value_scalar"] == 0.0)
                ]
                out[run][arm][direction] = set(rows["step"].astype(int))
    return out


def family_variants(frames: dict[str, dict[str, pd.DataFrame]]) -> list[tuple[str, str, str]]:
    variants: set[tuple[str, str, str]] = set()
    for tier in TIERS:
        for frame in frames[tier].values():
            for metric, rows in frame.groupby("metric", sort=False):
                arms = sorted(str(a) for a in rows["acceptance_arm"].dropna().unique())
                if arms:
                    variants.update((tier, str(metric), arm) for arm in arms)
                else:
                    variants.add((tier, str(metric), "none"))
    return sorted(variants)


def row_value(row: Any) -> np.ndarray:
    vector = row.value_vector
    if vector is not None and not (isinstance(vector, float) and math.isnan(vector)):
        arr = np.asarray(vector, dtype=np.float64)
    else:
        arr = np.asarray([row.value_scalar], dtype=np.float64)
    return arr.reshape(-1)


def row_key(row: Any) -> tuple[Any, ...]:
    progress = round(float(row.normalized_progress), 12)
    return (progress,) + tuple(norm_key_value(getattr(row, c)) for c in CHANNEL_COLUMNS)


def rows_to_map(frame: pd.DataFrame, family: str) -> dict[tuple[Any, ...], np.ndarray]:
    mapped: dict[tuple[Any, ...], np.ndarray] = {}
    for row in frame.itertuples(index=False):
        key = row_key(row)
        if key in mapped:
            raise AssertionError(f"duplicate semantic key in {family}: {key}")
        arr = row_value(row)
        if not np.all(np.isfinite(arr)):
            raise AssertionError(f"defined non-finite value in {family}: {key}")
        mapped[key] = arr
    return mapped


def safe_median(values: np.ndarray) -> float:
    return float(np.median(values)) if len(values) else math.nan


def safe_max(values: np.ndarray) -> float:
    return float(np.max(values)) if len(values) else math.nan


def classify(
    typical: float,
    worst: float,
    early: float,
    late: float,
    inf_fraction: float,
    n: int,
) -> str:
    """Exploratory utility bands declared before inspecting seed spreads."""
    if n == 0 or math.isnan(typical):
        return "unavailable"
    region_values = [v for v in (early, late) if not math.isnan(v)]
    if (
        math.isfinite(typical)
        and typical <= 0.05
        and all(math.isfinite(v) and v <= 0.05 for v in region_values)
        and math.isfinite(worst)
        and worst <= 0.20
    ):
        return "tight"
    if not math.isfinite(typical) or typical >= 0.50 or inf_fraction >= 0.50:
        return "noisy"
    return "intermediate"


def analyze_variant(
    tier: str,
    metric: str,
    arm: str,
    tier_frames: dict[str, pd.DataFrame],
    pass_steps: dict[str, dict[str, dict[str, set[int]]]],
) -> dict[str, Any]:
    maps: dict[str, dict[tuple[Any, ...], np.ndarray]] = {}
    defined_rows = 0
    raw_rows = 0
    probe_sampled = False
    filtered_for_pass = False
    length_mismatch_keys = 0
    direction = direction_for_metric(metric)

    for run, frame in tier_frames.items():
        rows = frame[frame["metric"] == metric]
        if arm != "none":
            rows = rows[rows["acceptance_arm"] == arm]
        else:
            rows = rows[rows["acceptance_arm"].isna()]
        raw_rows += len(rows)
        probe_sampled = probe_sampled or bool(rows["probe_id"].notna().any())
        rows = rows[rows["is_defined"]]

        # Frozen protocol: curvature uses passing per-direction verdicts.
        # Explicit native exception: the protocol simultaneously requires its
        # spread despite universal failure, so native is computed unfiltered
        # and flagged uncertified.  Generic/status curvature rows without a
        # governing direction are not conditioned.
        if metric.startswith("curvature/") and direction and arm == "shadow_fp32":
            rows = rows[rows["step"].isin(pass_steps[run][arm][direction])]
            filtered_for_pass = True

        defined_rows += len(rows)
        maps[run] = rows_to_map(rows, f"{tier}:{metric}[{arm}]")

    key_sets = [set(maps[run]) for run in RUNS]
    union_keys = set().union(*key_sets)
    common_keys = set.intersection(*key_sets)

    spreads: list[np.ndarray] = []
    relatives: list[np.ndarray] = []
    progresses: list[np.ndarray] = []
    zero_denoms = 0
    inf_values = 0

    for key in sorted(common_keys, key=repr):
        arrays = [maps[run][key] for run in RUNS]
        lengths = [len(a) for a in arrays]
        length = min(lengths)
        if len(set(lengths)) != 1:
            length_mismatch_keys += 1
        if length == 0:
            continue
        values = np.stack([a[:length] for a in arrays], axis=0)
        spread = np.max(values, axis=0) - np.min(values, axis=0)
        median = np.median(values, axis=0)
        denom = np.abs(median)
        relative = np.empty_like(spread)
        nonzero = denom > 0
        relative[nonzero] = spread[nonzero] / denom[nonzero]
        both_zero = (~nonzero) & (spread == 0)
        relative[both_zero] = 0.0
        relative[(~nonzero) & (spread > 0)] = np.inf
        zero_denoms += int(np.sum(~nonzero))
        inf_values += int(np.sum(np.isinf(relative)))
        spreads.append(spread)
        relatives.append(relative)
        progresses.append(np.full(length, float(key[0])))

    if spreads:
        abs_values = np.concatenate(spreads)
        rel_values = np.concatenate(relatives)
        progress_values = np.concatenate(progresses)
    else:
        abs_values = np.asarray([], dtype=np.float64)
        rel_values = np.asarray([], dtype=np.float64)
        progress_values = np.asarray([], dtype=np.float64)

    early_mask = progress_values <= EARLY_CUTOFF
    late_mask = progress_values > EARLY_CUTOFF
    typical_rel = safe_median(rel_values)
    worst_rel = safe_max(rel_values)
    early_rel = safe_median(rel_values[early_mask])
    late_rel = safe_median(rel_values[late_mask])
    n = len(rel_values)
    inf_fraction = inf_values / n if n else math.nan
    zero_fraction = zero_denoms / n if n else math.nan
    variant = metric if arm == "none" else f"{metric} [{arm}]"
    certification = (
        "uncertified_native_raw"
        if arm == "native"
        else "passing_direction_only"
        if filtered_for_pass
        else "not_applicable"
    )

    return {
        "tier": tier,
        "metric": metric,
        "acceptance_arm": arm,
        "family": f"{tier}:{variant}",
        "direction": direction or "none",
        "probe_sampled": probe_sampled,
        "certification": certification,
        "raw_rows_five_runs": raw_rows,
        "selected_defined_rows_five_runs": defined_rows,
        "union_row_keys": len(union_keys),
        "aligned_row_keys": len(common_keys),
        "row_key_coverage": len(common_keys) / len(union_keys) if union_keys else math.nan,
        "aligned_elements": n,
        "early_elements": int(np.sum(early_mask)),
        "late_elements": int(np.sum(late_mask)),
        "vector_length_mismatch_keys": length_mismatch_keys,
        "typical_absolute_spread": safe_median(abs_values),
        "worst_absolute_spread": safe_max(abs_values),
        "typical_relative_spread": typical_rel,
        "worst_relative_spread": worst_rel,
        "early_typical_relative_spread": early_rel,
        "late_typical_relative_spread": late_rel,
        "zero_median_fraction": zero_fraction,
        "infinite_relative_fraction": inf_fraction,
        "utility_band": classify(
            typical_rel, worst_rel, early_rel, late_rel, inf_fraction, n
        ),
    }


def rank_key(value: float) -> tuple[int, float]:
    if pd.isna(value):
        return (2, math.inf)
    if math.isinf(float(value)):
        return (1, math.inf)
    return (0, float(value))


def format_number(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    value = float(value)
    if math.isinf(value):
        return "inf"
    if value == 0:
        return "0"
    if abs(value) >= 1000 or abs(value) < 0.001:
        return f"{value:.3e}"
    return f"{value:.4f}"


def write_markdown_table(ranking: pd.DataFrame) -> None:
    cols = [
        "rank",
        "family",
        "aligned_elements",
        "row_key_coverage",
        "typical_absolute_spread",
        "worst_absolute_spread",
        "typical_relative_spread",
        "worst_relative_spread",
        "early_typical_relative_spread",
        "late_typical_relative_spread",
        "utility_band",
        "flags",
    ]
    labels = [
        "rank",
        "family",
        "n",
        "key cov.",
        "typ. abs",
        "worst abs",
        "typ. rel",
        "worst rel",
        "early rel",
        "late rel",
        "band",
        "flags",
    ]
    lines = ["| " + " | ".join(labels) + " |", "|" + "|".join(["---"] * len(labels)) + "|"]
    numeric = {
        "row_key_coverage",
        "typical_absolute_spread",
        "worst_absolute_spread",
        "typical_relative_spread",
        "worst_relative_spread",
        "early_typical_relative_spread",
        "late_typical_relative_spread",
    }
    for row in ranking[cols].itertuples(index=False, name=None):
        rendered = []
        for col, value in zip(cols, row):
            if col in numeric:
                rendered.append(format_number(value))
            else:
                rendered.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(rendered) + " |")
    (HERE / "family_ranking.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    frames: dict[str, dict[str, pd.DataFrame]] = {
        tier: {run: load_tier(run, tier) for run in RUNS} for tier in TIERS
    }
    probe_maps = make_probe_maps(frames["periodic"])
    for tier in TIERS:
        for run in RUNS:
            frames[tier][run] = add_probe_labels(frames[tier][run], probe_maps[run])

    base_by_tier = {
        tier: sorted(set().union(*(set(f["metric"]) for f in frames[tier].values())))
        for tier in TIERS
    }
    base_families = {(tier, metric) for tier, metrics in base_by_tier.items() for metric in metrics}
    variants = family_variants(frames)
    pass_steps = build_pass_steps(frames["sparse"])

    results = [
        analyze_variant(tier, metric, arm, frames[tier], pass_steps)
        for tier, metric, arm in variants
    ]
    ranking = pd.DataFrame(results)
    order = sorted(
        range(len(ranking)),
        key=lambda i: (
            rank_key(ranking.iloc[i]["typical_relative_spread"]),
            ranking.iloc[i]["family"],
        ),
    )
    ranking = ranking.iloc[order].reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

    def flags(row: pd.Series) -> str:
        values: list[str] = []
        if row["probe_sampled"]:
            values.append("probe")
        if row["certification"] == "uncertified_native_raw":
            values.append("uncertified")
        elif row["certification"] == "passing_direction_only":
            values.append("pass-only")
        if row["infinite_relative_fraction"] > 0:
            values.append("zero-median")
        return ",".join(values) or "-"

    ranking["flags"] = ranking.apply(flags, axis=1)
    ranking.to_csv(HERE / "family_ranking.csv", index=False)
    write_markdown_table(ranking)

    pass_counts = {
        run: {
            arm: {direction: len(steps) for direction, steps in directions.items()}
            for arm, directions in arms.items()
        }
        for run, arms in pass_steps.items()
    }
    summary = {
        "runs": list(RUNS),
        "base_family_count": len(base_families),
        "base_family_count_by_tier": {k: len(v) for k, v in base_by_tier.items()},
        "reported_arm_specific_family_count": len(variants),
        "utility_band_counts": ranking["utility_band"].value_counts().to_dict(),
        "probe_sampled_family_count": int(ranking["probe_sampled"].sum()),
        "uncertified_native_family_count": int(
            (ranking["certification"] == "uncertified_native_raw").sum()
        ),
        "passing_direction_family_count": int(
            (ranking["certification"] == "passing_direction_only").sum()
        ),
        "families_with_vector_length_mismatch": int(
            (ranking["vector_length_mismatch_keys"] > 0).sum()
        ),
        "pass_counts": pass_counts,
        "early_cutoff_normalized_progress": EARLY_CUTOFF,
        "relative_zero_convention": "0/0 -> 0; positive/0 -> inf",
        "tight_rule": "typical, early, late <= 0.05 and worst <= 0.20",
        "noisy_rule": "typical >= 0.50 or nonfinite, or >=50% infinite relative spreads",
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nFirst 20 ranked families:")
    print(
        ranking[
            [
                "rank",
                "family",
                "typical_relative_spread",
                "worst_relative_spread",
                "early_typical_relative_spread",
                "late_typical_relative_spread",
                "utility_band",
                "aligned_elements",
                "flags",
            ]
        ].head(20).to_string(index=False)
    )
    print("\nLast 20 ranked families:")
    print(
        ranking[
            [
                "rank",
                "family",
                "typical_relative_spread",
                "worst_relative_spread",
                "utility_band",
                "aligned_elements",
                "flags",
            ]
        ].tail(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()
