#!/usr/bin/env python3
"""Feasibility analysis for I0008/A0002.

This script deliberately treats the eight noise-estimator slices as random
sub-batches, never as semantic data groups.  It reads only the seven schema-v3
segments and writes all derived artifacts beside this file.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib-i0008-a0002")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds


HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parents[2]
DATA_ROOT = ANALYSIS.parent / "telemetry-data" / "sweep" / "telemetry-data"

PERIODIC_METRICS = {
    "noise/pairwise_cosines",
    "noise/per_sub_sq_norm",
    "noise/s2",
    "noise/signal_raw",
    "noise/b_noise",
    "noise/mean_grad_norm",
    "grad/norm",
    "sketch/grad",
    "sketch/grad_sq_norm",
    "sketch/grad_cosine_prev",
    "overhead/noise",
    "overhead/grads_ready/periodic_scan",
}

SPARSE_METRICS = {
    "sketch/probe_grad",
    "sketch/probe_grad_sq_norm",
    "sketch/probe_grad_cosine_prev",
    "calib/grad_cosine_prev",
    "update/p1",
    "update/p2",
    "update/actual",
    "update/residual_p1",
    "update/residual_p2",
    "update/normalized_residual",
    "update/loss_before",
    "update/loss_after",
    "update/direction_norm",
    "curvature/dhd",
    "curvature/gHg",
    "curvature/gg",
    "curvature/Hg_norm",
    "curvature/eta_star",
    "curvature/eta_star_rho",
    "curvature/verdict_code_random",
    "curvature/verdict_code_gradient",
    "curvature/verdict_code_update",
    "curvature/native_verdict_code",
    "curvature/shadow_verdict_code",
    "overhead/probe_grad_sketch",
    "overhead/shadow_acceptance",
    "overhead/update_effectiveness",
}

COLUMNS = [
    "metric",
    "step",
    "normalized_progress",
    "value_scalar",
    "value_vector",
    "is_defined",
    "undefined_reason",
    "param_role",
    "layer",
    "acceptance_arm",
    "acceptance_status",
    "estimator_id",
    "sample_count",
    "parameter_schema_hash",
    "sketch_seed",
    "run_id",
]


def segment_metadata(path: Path) -> dict:
    prov = json.loads((path / "provenance.json").read_text())
    label_match = re.match(r"(d(\d+)-s(\d+))-s0-", path.name)
    if label_match is None:
        raise ValueError(path.name)
    return {
        "segment": path.name,
        "run": label_match.group(1),
        "depth": int(label_match.group(2)),
        "seed": int(label_match.group(3)),
        "iterations": int(prov["num_iterations"]),
        "periodic_every": int(prov["telemetry_every"]),
        "device_batch_rows": int(prov["device_batch_size"]),
        "logical_batch_tokens": int(prov["total_batch_size"]),
        "grad_accum_steps": int(prov["grad_accum_steps"]),
        "sequence_len": int(prov["model_config"]["sequence_len"]),
        "noise_K": int(prov["telemetry_config"]["noise_K"]),
        "sketch_k": int(prov["telemetry_config"]["sketch_k"]),
        "deep_steps": prov["telemetry_deep_steps"],
    }


def discover_segments() -> list[dict]:
    paths = sorted(p for p in DATA_ROOT.iterdir() if p.is_dir() and "-iter-" not in p.name)
    meta = [segment_metadata(p) | {"path": p} for p in paths]
    assert len(meta) == 7, [m["segment"] for m in meta]
    assert all(m["noise_K"] == 8 for m in meta)
    assert all(m["device_batch_rows"] == 32 for m in meta)
    return meta


def load_tier(path: Path, tier: str, metrics: set[str]) -> pd.DataFrame:
    dataset = ds.dataset(str(path / tier), format="parquet")
    table = dataset.to_table(
        filter=ds.field("metric").isin(sorted(metrics)),
        columns=COLUMNS,
    )
    return table.to_pandas()


def phase(progress: float) -> str:
    if progress < 0.25:
        return "early"
    if progress >= 0.75:
        return "late"
    return "middle"


def spearman(x, y) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return math.nan
    return float(pair["x"].rank(method="average").corr(pair["y"].rank(method="average")))


def pearson(x, y) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return math.nan
    return float(pair["x"].corr(pair["y"]))


def scalar_rows(df: pd.DataFrame, name: str, defined: bool = True) -> pd.DataFrame:
    out = df[df.metric.eq(name)].copy()
    if defined:
        out = out[out.is_defined]
    return out


def vector_value(row) -> np.ndarray:
    return np.asarray(row.value_vector, dtype=np.float64)


def build_subbatch_events(periodic: pd.DataFrame, meta: dict):
    cos = scalar_rows(periodic, "noise/pairwise_cosines")
    norms = scalar_rows(periodic, "noise/per_sub_sq_norm")
    mean_role = scalar_rows(periodic, "noise/mean_grad_norm")
    bnoise = scalar_rows(periodic, "noise/b_noise")
    bnoise_map = dict(zip(bnoise.step.astype(int), bnoise.value_scalar))
    records = []
    alignments = []
    assert set(cos.step) == set(norms.step)
    for _, crow in cos.iterrows():
        nrow = norms[norms.step.eq(crow.step)].iloc[0]
        cs = vector_value(crow)
        ns = np.sqrt(np.maximum(vector_value(nrow), 0.0))
        assert len(cs) == 28
        assert len(ns) == 8
        gram = np.diag(ns**2)
        cursor = 0
        for i in range(8):
            for j in range(i + 1, 8):
                gram[i, j] = gram[j, i] = cs[cursor] * ns[i] * ns[j]
                cursor += 1
        mean_norm_rows = mean_role[mean_role.step.eq(crow.step)]
        mean_norm_exact = math.sqrt(float(np.square(mean_norm_rows.value_scalar).sum()))
        dot_mean = gram.sum(axis=1) / 8
        cos_mean = dot_mean / (ns * mean_norm_exact)
        for i in range(8):
            alignments.append({
                "run": meta["run"],
                "depth": meta["depth"],
                "seed": meta["seed"],
                "step": int(crow.step),
                "normalized_progress": float(crow.normalized_progress),
                "phase": phase(float(crow.normalized_progress)),
                "slice_index": i,
                "slice_grad_norm": float(ns[i]),
                "slice_dot_device_mean": float(dot_mean[i]),
                "slice_cosine_device_mean": float(cos_mean[i]),
                "device_mean_grad_norm_exact": mean_norm_exact,
            })
        records.append({
            "run": meta["run"],
            "depth": meta["depth"],
            "seed": meta["seed"],
            "step": int(crow.step),
            "normalized_progress": float(crow.normalized_progress),
            "phase": phase(float(crow.normalized_progress)),
            "pair_cos_mean": float(np.mean(cs)),
            "pair_cos_median": float(np.median(cs)),
            "pair_cos_q10": float(np.quantile(cs, 0.1)),
            "pair_cos_q90": float(np.quantile(cs, 0.9)),
            "pair_cos_negative_fraction": float(np.mean(cs < 0)),
            "sub_grad_norm_mean": float(np.mean(ns)),
            "sub_grad_norm_cv": float(np.std(ns, ddof=1) / np.mean(ns)),
            "sub_grad_norm_max_min_ratio": float(np.max(ns) / np.min(ns)),
            "slice_to_device_mean_cos_median": float(np.median(cos_mean)),
            "slice_to_device_mean_cos_min": float(np.min(cos_mean)),
            "b_noise": float(bnoise_map.get(int(crow.step), math.nan)),
        })
    return pd.DataFrame(records), pd.DataFrame(alignments)


def build_gradient_cosines(periodic: pd.DataFrame, sparse: pd.DataFrame, meta: dict) -> pd.DataFrame:
    specs = [
        (periodic, "sketch/grad_cosine_prev", "logical_sketch_prev"),
        (sparse, "sketch/probe_grad_cosine_prev", "fixed_probe_sketch_prev"),
        (sparse, "calib/grad_cosine_prev", "logical_exact_prev_deep"),
    ]
    records = []
    for frame, metric, source in specs:
        rows = scalar_rows(frame, metric)
        for _, row in rows.iterrows():
            records.append({
                "run": meta["run"],
                "depth": meta["depth"],
                "seed": meta["seed"],
                "source": source,
                "step": int(row.step),
                "normalized_progress": float(row.normalized_progress),
                "phase": phase(float(row.normalized_progress)),
                "cosine": float(row.value_scalar),
                "sample_count": row.sample_count,
            })
    return pd.DataFrame(records)


def sketch_events(df: pd.DataFrame, vector_metric: str, norm_metric: str) -> dict[int, dict]:
    vectors = scalar_rows(df, vector_metric)
    norms = scalar_rows(df, norm_metric)
    events = {}
    for step, vstep in vectors.groupby("step"):
        nstep = norms[norms.step.eq(step)]
        v_by_key = {}
        n_by_key = {}
        hashes = set(vstep.parameter_schema_hash.dropna())
        seeds = set(vstep.sketch_seed.dropna())
        assert len(hashes) == 1 and len(seeds) == 1
        for _, row in vstep.iterrows():
            key = (row.param_role, None if pd.isna(row.layer) else int(row.layer))
            v_by_key[key] = vector_value(row)
        for _, row in nstep.iterrows():
            key = (row.param_role, None if pd.isna(row.layer) else int(row.layer))
            n_by_key[key] = float(row.value_scalar)
        assert set(v_by_key) == set(n_by_key)
        events[int(step)] = {
            "vectors": v_by_key,
            "sq_norms": n_by_key,
            "progress": float(vstep.normalized_progress.iloc[0]),
            "schema_hash": next(iter(hashes)),
            "sketch_seed": int(next(iter(seeds))),
        }
    return events


def sketch_cosine(a: dict, b: dict) -> float:
    assert a["schema_hash"] == b["schema_hash"]
    assert a["sketch_seed"] == b["sketch_seed"]
    assert set(a["vectors"]) == set(b["vectors"])
    keys = sorted(a["vectors"], key=str)
    dot = sum(float(np.dot(a["vectors"][key], b["vectors"][key])) for key in keys)
    na = sum(a["sq_norms"][key] for key in keys)
    nb = sum(b["sq_norms"][key] for key in keys)
    return dot / math.sqrt(na * nb)


def build_sketch_long_range(periodic: pd.DataFrame, sparse: pd.DataFrame, meta: dict):
    logical = sketch_events(periodic, "sketch/grad", "sketch/grad_sq_norm")
    probe = sketch_events(sparse, "sketch/probe_grad", "sketch/probe_grad_sq_norm")
    records = []
    for source, events in (("logical_batch", logical), ("fixed_probe", probe)):
        first_step = min(events)
        first = events[first_step]
        for step, event in sorted(events.items()):
            records.append({
                "run": meta["run"],
                "depth": meta["depth"],
                "seed": meta["seed"],
                "source": source,
                "step": step,
                "normalized_progress": event["progress"],
                "phase": phase(event["progress"]),
                "cosine_to_initial": sketch_cosine(first, event),
            })
    common = sorted(set(logical) & set(probe))
    synchronous = [{
        "run": meta["run"],
        "depth": meta["depth"],
        "seed": meta["seed"],
        "step": step,
        "normalized_progress": logical[step]["progress"],
        "fixed_probe_vs_logical_cosine": sketch_cosine(probe[step], logical[step]),
    } for step in common]
    return pd.DataFrame(records), pd.DataFrame(synchronous)


def build_role_layer_shares(periodic: pd.DataFrame, meta: dict):
    rows = scalar_rows(periodic, "grad/norm")
    rows = rows.assign(
        run=meta["run"],
        depth=meta["depth"],
        seed=meta["seed"],
        sq=rows.value_scalar.astype(float).pow(2),
        phase=rows.normalized_progress.map(phase),
    )
    totals = rows.groupby("step").sq.sum().rename("total_sq")
    role = rows.groupby(["run", "depth", "seed", "step", "normalized_progress", "phase", "param_role"], dropna=False).sq.sum().reset_index()
    role = role.join(totals, on="step")
    role["share"] = role.sq / role.total_sq

    layered = rows[rows.layer.notna()].copy()
    layered["layer"] = layered.layer.astype(int)
    layered["layer_fraction"] = (layered.layer + 0.5) / meta["depth"]
    layered["layer_region"] = pd.cut(
        layered.layer_fraction,
        bins=[0.0, 1 / 3, 2 / 3, 1.0],
        labels=["lower_third", "middle_third", "upper_third"],
        include_lowest=True,
    ).astype(str)
    layer = layered.groupby(["run", "depth", "seed", "step", "normalized_progress", "phase", "layer_region"], observed=True).sq.sum().reset_index()
    ltot = layered.groupby("step").sq.sum().rename("layered_total_sq")
    layer = layer.join(ltot, on="step")
    layer["share_within_layered"] = layer.sq / layer.layered_total_sq
    return role, layer, len(rows)


def sparse_wide(sparse: pd.DataFrame, arm: str) -> pd.DataFrame:
    rows = sparse[sparse.acceptance_arm.eq(arm) & sparse.is_defined].copy()
    keep = rows[rows.metric.str.startswith(("update/", "curvature/"))]
    idx = ["run_id", "step", "normalized_progress"]
    wide = keep.pivot_table(index=idx, columns="metric", values="value_scalar", aggfunc="first").reset_index()
    return wide


def build_update_and_hvp(sparse: pd.DataFrame, meta: dict):
    wide = sparse_wide(sparse, "shadow_fp32")
    wide.insert(0, "run", meta["run"])
    wide.insert(1, "depth", meta["depth"])
    wide.insert(2, "seed", meta["seed"])
    wide["phase"] = wide.normalized_progress.map(phase)
    wide["benefit_relative"] = -wide["update/actual"] / wide["update/loss_before"]
    wide["p1_benefit_relative"] = -wide["update/p1"] / wide["update/loss_before"]
    wide["p2_benefit_relative"] = -wide["update/p2"] / wide["update/loss_before"]
    wide["grad_norm"] = np.sqrt(wide["curvature/gg"])
    wide["p1_normalized_abs_error"] = (
        (wide["update/actual"] - wide["update/p1"]).abs()
        / np.maximum.reduce([
            wide["update/actual"].abs().to_numpy(),
            wide["update/p1"].abs().to_numpy(),
            np.full(len(wide), 1e-12),
        ])
    )
    wide["p2_normalized_abs_error"] = wide["update/normalized_residual"].abs()
    return wide


def build_hvp_verdicts(sparse: pd.DataFrame, meta: dict) -> pd.DataFrame:
    names = [f"curvature/verdict_code_{d}" for d in ("random", "gradient", "update")]
    rows = sparse[
        sparse.metric.isin(names)
        & sparse.acceptance_arm.eq("shadow_fp32")
        & sparse.is_defined
    ].copy()
    rows["direction"] = rows.metric.str.rsplit("_", n=1).str[-1]
    rows["verdict"] = rows.value_scalar.map({0.0: "passed", 1.0: "inconclusive", 2.0: "failed"})
    rows["run"] = meta["run"]
    rows["depth"] = meta["depth"]
    rows["seed"] = meta["seed"]
    return rows[["run", "depth", "seed", "step", "normalized_progress", "direction", "verdict", "value_scalar"]]


def build_overhead(periodic: pd.DataFrame, sparse: pd.DataFrame, continuous: pd.DataFrame, meta: dict) -> pd.DataFrame:
    step_dt = scalar_rows(continuous, "step/observed_dt").value_scalar.median()
    wanted = [
        (periodic, "overhead/noise"),
        (periodic, "overhead/grads_ready/periodic_scan"),
        (sparse, "overhead/probe_grad_sketch"),
        (sparse, "overhead/shadow_acceptance"),
        (sparse, "overhead/update_effectiveness"),
    ]
    records = []
    for frame, metric in wanted:
        rows = scalar_rows(frame, metric)
        records.append({
            "run": meta["run"],
            "depth": meta["depth"],
            "metric": metric,
            "events": len(rows),
            "median_seconds": float(rows.value_scalar.median()),
            "mean_seconds": float(rows.value_scalar.mean()),
            "median_training_step_seconds": float(step_dt),
            "median_event_in_step_equivalents": float(rows.value_scalar.median() / step_dt),
        })
    return pd.DataFrame(records)


def correlation_summary(updates: pd.DataFrame) -> pd.DataFrame:
    predictors = {
        "p1 predicted relative benefit": "p1_benefit_relative",
        "p2 predicted relative benefit (uncertified HVP)": "p2_benefit_relative",
        "probe gradient norm": "grad_norm",
        "probe g-Hg cosine": "curvature/eta_star_rho",
        "probe gHg": "curvature/gHg",
        "actual update norm": "update/direction_norm",
    }
    records = []
    for run, group in updates.groupby("run"):
        for label, column in predictors.items():
            records.append({
                "run": run,
                "predictor": label,
                "n": int(group[[column, "benefit_relative"]].dropna().shape[0]),
                "pearson": pearson(group[column], group.benefit_relative),
                "spearman": spearman(group[column], group.benefit_relative),
            })
    return pd.DataFrame(records)


def summarize(
    meta_all,
    subbatch,
    gradient_cosines,
    sketch_long,
    synchronous,
    subbatch_alignments,
    role_shares,
    layer_shares,
    updates,
    verdicts,
    overhead,
    correlation,
    grad_norm_row_count,
):
    d12 = subbatch[subbatch.depth.eq(12)]
    per_seed = d12.groupby(["run", "phase"]).pair_cos_median.median().unstack()
    per_seed["late_minus_early"] = per_seed.late - per_seed.early
    bnoise_per_seed = d12.groupby(["run", "phase"]).b_noise.median().unstack()
    bnoise_per_seed["late_over_early"] = bnoise_per_seed.late / bnoise_per_seed.early
    sub_phase = subbatch.groupby("phase").agg(
        events=("step", "size"),
        pair_cos_median=("pair_cos_median", "median"),
        pair_cos_negative_fraction=("pair_cos_negative_fraction", "median"),
        sub_grad_norm_cv=("sub_grad_norm_cv", "median"),
        slice_to_device_mean_cos_median=("slice_to_device_mean_cos_median", "median"),
        b_noise=("b_noise", "median"),
    )
    sub_trends = subbatch.groupby("run").apply(
        lambda g: pd.Series({
            "spearman_pair_cos_vs_progress": spearman(g.normalized_progress, g.pair_cos_median),
            "spearman_norm_cv_vs_progress": spearman(g.normalized_progress, g.sub_grad_norm_cv),
        }),
        include_groups=False,
    )

    cosine_phase = gradient_cosines.groupby(["source", "phase"]).cosine.agg(["count", "median", "min", "max"])
    long_phase = sketch_long.groupby(["source", "phase"]).cosine_to_initial.agg(["count", "median", "min", "max"])

    role_summary = role_shares.groupby(["param_role", "phase"]).share.median().unstack()
    role_summary["late_minus_early"] = role_summary.get("late", np.nan) - role_summary.get("early", np.nan)
    role_summary = role_summary.sort_values("late", ascending=False)
    layer_summary = layer_shares.groupby(["layer_region", "phase"]).share_within_layered.median().unstack()
    layer_summary["late_minus_early"] = layer_summary.late - layer_summary.early

    verdict_summary = verdicts.groupby(["direction", "verdict"]).size().unstack(fill_value=0)
    for col in ["passed", "inconclusive", "failed"]:
        if col not in verdict_summary:
            verdict_summary[col] = 0
    verdict_summary = verdict_summary[["passed", "inconclusive", "failed"]]
    certified_gradient = updates[updates["curvature/verdict_code_gradient"].eq(0.0)].copy()
    certified_curvature_phase = certified_gradient.groupby("phase").agg(
        checkpoints=("step", "size"),
        median_gHg=("curvature/gHg", "median"),
        median_grad_norm=("grad_norm", "median"),
        median_g_Hg_cosine=("curvature/eta_star_rho", "median"),
        median_eta_star=("curvature/eta_star", "median"),
    )

    update_per_run = updates.groupby("run").apply(
        lambda g: pd.Series({
            "n": len(g),
            "fraction_actual_probe_improvement": float((g["update/actual"] < 0).mean()),
            "median_relative_benefit": float(g.benefit_relative.median()),
            "p1_sign_agreement": float((np.sign(g["update/p1"]) == np.sign(g["update/actual"])).mean()),
            "p2_sign_agreement": float((np.sign(g["update/p2"]) == np.sign(g["update/actual"])).mean()),
            "median_p1_normalized_abs_error": float(g.p1_normalized_abs_error.median()),
            "median_p2_normalized_abs_error": float(g.p2_normalized_abs_error.median()),
            "pearson_p1_actual": pearson(g["update/p1"], g["update/actual"]),
            "pearson_p2_actual_uncertified": pearson(g["update/p2"], g["update/actual"]),
        }),
        include_groups=False,
    )

    corr_summary = correlation.groupby("predictor").agg(
        runs=("run", "size"),
        median_spearman=("spearman", "median"),
        min_spearman=("spearman", "min"),
        max_spearman=("spearman", "max"),
        median_pearson=("pearson", "median"),
    ).sort_values("median_spearman", ascending=False)

    overhead_summary = overhead.groupby(["metric", "depth"]).agg(
        runs=("run", "size"),
        median_seconds=("median_seconds", "median"),
        median_step_equivalents=("median_event_in_step_equivalents", "median"),
    )

    def table_dict(df):
        return json.loads(df.reset_index().to_json(orient="records"))

    summary = {
        "selection": {
            "segments": [m["segment"] for m in meta_all],
            "schema_v3_segments": len(meta_all),
            "periodic_events": int(len(subbatch)),
            "defined_b_noise_events": int(subbatch.b_noise.notna().sum()),
            "random_slices_per_event": 8,
            "random_slice_pair_cosines_per_event": 28,
            "random_slice_pair_cosines_total": int(len(subbatch) * 28),
            "subbatch_gradient_norms_total": int(len(subbatch) * 8),
            "random_slice_to_device_mean_alignments": int(len(subbatch_alignments)),
            "gradient_norm_role_layer_rows": int(grad_norm_row_count),
            "update_effectiveness_checkpoints_shadow_fp32": int(len(updates)),
            "hvp_direction_verdicts_shadow_fp32": int(len(verdicts)),
            "synchronous_fixed_probe_logical_sketch_points": int(len(synchronous)),
            "synchronous_steps_by_run": synchronous.groupby("run").step.apply(lambda s: [int(v) for v in s]).to_dict(),
        },
        "subbatch_phase": table_dict(sub_phase),
        "d12_pair_cosine_by_seed": table_dict(per_seed),
        "d12_b_noise_by_seed": table_dict(bnoise_per_seed),
        "d12_late_minus_early_mean": float(per_seed.late_minus_early.mean()),
        "d12_late_minus_early_sd": float(per_seed.late_minus_early.std(ddof=1)),
        "d12_late_minus_early_sign_count": int((per_seed.late_minus_early > 0).sum()),
        "subbatch_per_run_trends": table_dict(sub_trends),
        "gradient_cosine_phase": table_dict(cosine_phase),
        "sketch_cosine_to_initial_phase": table_dict(long_phase),
        "synchronous_probe_vs_logical": json.loads(synchronous.to_json(orient="records")),
        "role_share_summary": table_dict(role_summary),
        "layer_region_summary": table_dict(layer_summary),
        "hvp_verdict_summary": table_dict(verdict_summary),
        "certified_gradient_hvp_phase": table_dict(certified_curvature_phase),
        "certified_gradient_hvp_checkpoints": int(len(certified_gradient)),
        "certified_update_hvp_checkpoints": int(updates["curvature/verdict_code_update"].eq(0.0).sum()),
        "update_effectiveness_by_run": table_dict(update_per_run),
        "update_effectiveness_all": {
            "n": int(len(updates)),
            "fraction_actual_probe_improvement": float((updates["update/actual"] < 0).mean()),
            "median_relative_benefit": float(updates.benefit_relative.median()),
            "median_p1_normalized_abs_error": float(updates.p1_normalized_abs_error.median()),
            "median_p2_normalized_abs_error_uncertified": float(updates.p2_normalized_abs_error.median()),
            "p1_sign_agreement": float((np.sign(updates["update/p1"]) == np.sign(updates["update/actual"])).mean()),
            "p2_sign_agreement_uncertified": float((np.sign(updates["update/p2"]) == np.sign(updates["update/actual"])).mean()),
        },
        "predictor_correlations_by_run_summary": table_dict(corr_summary),
        "overhead_summary": table_dict(overhead_summary),
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def make_figures(subbatch, sketch_long, role_shares, updates):
    colors = {12: "#2166ac", 14: "#f28e2b", 16: "#b2182b"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), constrained_layout=True)
    for run, group in subbatch.groupby("run"):
        depth = int(group.depth.iloc[0])
        alpha = 0.45 if depth == 12 else 0.9
        axes[0, 0].plot(group.normalized_progress, group.pair_cos_median, marker="o", ms=2.5,
                        lw=1, alpha=alpha, color=colors[depth])
        axes[0, 1].plot(group.normalized_progress, group.sub_grad_norm_cv, marker="o", ms=2.5,
                        lw=1, alpha=alpha, color=colors[depth])
    axes[0, 0].axhline(0, color="black", lw=0.6)
    axes[0, 0].set(title="Random 4-row slice gradient alignment", ylabel="median of 28 sketched cosines")
    axes[0, 1].set(title="Random-slice gradient norm dispersion", ylabel="CV across 8 slice norms")

    probe = sketch_long[sketch_long.source.eq("fixed_probe")]
    for run, group in probe.groupby("run"):
        depth = int(group.depth.iloc[0])
        alpha = 0.45 if depth == 12 else 0.9
        axes[1, 0].plot(group.normalized_progress, group.cosine_to_initial, marker=".", lw=1,
                        alpha=alpha, color=colors[depth])
    axes[1, 0].axhline(0, color="black", lw=0.6)
    axes[1, 0].set(title="Fixed-probe gradient rotation", ylabel="CountSketch cosine to step 0")

    role_top = role_shares.groupby("param_role").share.median().nlargest(5).index
    d12_roles = role_shares[role_shares.depth.eq(12)]
    for role in role_top:
        curve = d12_roles[d12_roles.param_role.eq(role)].groupby("normalized_progress").share.median()
        axes[1, 1].plot(curve.index, curve.values, marker=".", lw=1, label=role)
    axes[1, 1].set(title="Largest parameter-role shares (d12 seed median)", ylabel="share of logical-gradient squared norm")
    axes[1, 1].legend(fontsize=7, ncol=2)
    for ax in axes.flat:
        ax.set_xlabel("normalized progress")
        ax.grid(alpha=0.2)
    fig.savefig(HERE / "proxy_trajectories.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 6), constrained_layout=True)
    for depth, group in updates.groupby("depth"):
        ax.scatter(group["update/actual"], group["update/p1"], s=17, alpha=0.65,
                   color=colors[int(depth)], label=f"d{depth}")
    vals = np.r_[updates["update/actual"].to_numpy(), updates["update/p1"].to_numpy()]
    lo, hi = np.quantile(vals[np.isfinite(vals)], [0.01, 0.99])
    ax.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--", label="identity")
    ax.set(xlabel="actual one-update fixed-probe loss change",
           ylabel="first-order prediction g_probe^T Delta",
           title="Nominal applied update: immediate effect, not future group value")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(HERE / "update_effectiveness.png", dpi=170)
    plt.close(fig)


def main():
    meta_all = discover_segments()
    all_subbatch = []
    all_cosines = []
    all_subbatch_alignments = []
    all_sketch_long = []
    all_sync = []
    all_role = []
    all_layer = []
    all_updates = []
    all_verdicts = []
    all_overhead = []
    grad_norm_row_count = 0

    for meta in meta_all:
        periodic = load_tier(meta["path"], "periodic", PERIODIC_METRICS)
        sparse = load_tier(meta["path"], "sparse", SPARSE_METRICS)
        continuous = load_tier(meta["path"], "continuous", {"step/observed_dt"})
        assert periodic.run_id.nunique() == sparse.run_id.nunique() == continuous.run_id.nunique() == 1
        subbatch, subbatch_alignments = build_subbatch_events(periodic, meta)
        all_subbatch.append(subbatch)
        all_subbatch_alignments.append(subbatch_alignments)
        all_cosines.append(build_gradient_cosines(periodic, sparse, meta))
        sketch_long, sync = build_sketch_long_range(periodic, sparse, meta)
        all_sketch_long.append(sketch_long)
        all_sync.append(sync)
        role, layer, count = build_role_layer_shares(periodic, meta)
        all_role.append(role)
        all_layer.append(layer)
        grad_norm_row_count += count
        all_updates.append(build_update_and_hvp(sparse, meta))
        all_verdicts.append(build_hvp_verdicts(sparse, meta))
        all_overhead.append(build_overhead(periodic, sparse, continuous, meta))

    subbatch = pd.concat(all_subbatch, ignore_index=True)
    gradient_cosines = pd.concat(all_cosines, ignore_index=True)
    subbatch_alignments = pd.concat(all_subbatch_alignments, ignore_index=True)
    sketch_long = pd.concat(all_sketch_long, ignore_index=True)
    synchronous = pd.concat(all_sync, ignore_index=True)
    role_shares = pd.concat(all_role, ignore_index=True)
    layer_shares = pd.concat(all_layer, ignore_index=True)
    updates = pd.concat(all_updates, ignore_index=True)
    verdicts = pd.concat(all_verdicts, ignore_index=True)
    overhead = pd.concat(all_overhead, ignore_index=True)
    correlations = correlation_summary(updates)
    certified_gradient_hvp = updates[
        updates["curvature/verdict_code_gradient"].eq(0.0)
    ].copy()

    frames = {
        "subbatch_events.csv": subbatch,
        "subbatch_alignment_to_device_mean.csv": subbatch_alignments,
        "gradient_cosines.csv": gradient_cosines,
        "sketch_cosines_to_initial.csv": sketch_long,
        "synchronous_probe_logical_alignment.csv": synchronous,
        "role_gradient_shares.csv": role_shares,
        "layer_region_gradient_shares.csv": layer_shares,
        "update_effectiveness.csv": updates,
        "hvp_direction_verdicts.csv": verdicts,
        "certified_gradient_hvp.csv": certified_gradient_hvp,
        "overhead_costs.csv": overhead,
        "one_step_predictor_correlations.csv": correlations,
    }
    for name, frame in frames.items():
        frame.to_csv(HERE / name, index=False)

    summary = summarize(
        meta_all,
        subbatch,
        gradient_cosines,
        sketch_long,
        synchronous,
        subbatch_alignments,
        role_shares,
        layer_shares,
        updates,
        verdicts,
        overhead,
        correlations,
        grad_norm_row_count,
    )
    make_figures(subbatch, sketch_long, role_shares, updates)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
