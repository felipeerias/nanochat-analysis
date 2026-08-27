#!/usr/bin/env python3
"""I0007/A0002: exact-zero Muon wake-up ordering.

The analysis deliberately uses literal IEEE floating-point equality to zero.
No tolerance, rounding, or ``isclose`` operation participates in classifying a
matrix as asleep or awake.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
from loader.telemetry_load import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    defined,
    metric,
    read_telemetry,
)


DATA_ROOT = DEFAULT_DATA_ROOT
EARLY_DEEP_UPDATES = (0, 1, 2, 4, 8, 16, 32, 40, 64)
OUTPUT_ROLES = {"attn_out", "mlp_out"}


def exact_zero(values: pd.Series) -> pd.Series:
    """Classify exact numeric zero, including signed zero, without tolerance."""

    return values.eq(0.0)


def load_v3_inventory() -> list[dict]:
    runs = []
    for segment_dir in sorted(DATA_ROOT.iterdir()):
        if not segment_dir.is_dir():
            continue
        with (segment_dir / "provenance.json").open() as handle:
            provenance = json.load(handle)
        # The legacy v1 segment has no explicit deep-step schedule. Confirm the
        # selected segments from their row schema below as a second guard.
        if "telemetry_deep_steps" not in provenance:
            continue
        runs.append(
            {
                "segment": segment_dir.name,
                "run_id": provenance["manifest_run_id"],
                "seed": int(provenance["seed"]),
                "depth": int(provenance["model_config"]["n_layer"]),
                "horizon": int(provenance["num_iterations"]),
                "deep_steps": tuple(int(x) for x in provenance["telemetry_deep_steps"]),
            }
        )
    assert len(runs) == 7, [x["segment"] for x in runs]
    assert sorted(x["run_id"] for x in runs) == [
        "d12-s10",
        "d12-s11",
        "d12-s7",
        "d12-s8",
        "d12-s9",
        "d14-s7",
        "d16-s7",
    ]
    return runs


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, int]]:
    """Small, scipy-free Kendall tau-b implementation with tie counts."""

    concordant = discordant = tied_x_only = tied_y_only = tied_both = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        sx = int(np.sign(x[i] - x[j]))
        sy = int(np.sign(y[i] - y[j]))
        if sx == 0 and sy == 0:
            tied_both += 1
        elif sx == 0:
            tied_x_only += 1
        elif sy == 0:
            tied_y_only += 1
        elif sx == sy:
            concordant += 1
        else:
            discordant += 1
    denom = math.sqrt(
        (concordant + discordant + tied_x_only)
        * (concordant + discordant + tied_y_only)
    )
    tau = (concordant - discordant) / denom if denom else float("nan")
    counts = {
        "concordant": concordant,
        "discordant": discordant,
        "tied_x_only": tied_x_only,
        "tied_y_only": tied_y_only,
        "tied_both": tied_both,
    }
    return tau, counts


def extract_run(run: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    sparse = read_telemetry(str(DATA_ROOT), run["segment"], "sparse").to_pandas()
    periodic = read_telemetry(str(DATA_ROOT), run["segment"], "periodic").to_pandas()
    assert set(sparse["schema_version"].astype(str)) == {"3"}
    assert set(periodic["schema_version"].astype(str)) == {"3"}

    raw_decoh = metric(sparse, "muon/replay_update_relerr", phase="post_update").copy()
    raw_grad = metric(sparse, "sketch/probe_grad_sq_norm", phase="pre_update").copy()
    assert len(raw_decoh) and len(raw_grad)
    undefined_counts = {
        "decoherence": int((~raw_decoh["is_defined"]).sum()),
        "gradient_sq_norm_all_roles": int((~raw_grad["is_defined"]).sum()),
    }
    decoh = defined(raw_decoh).copy()
    grad = defined(raw_grad).copy()
    assert decoh["value_scalar"].notna().all()
    assert grad["value_scalar"].notna().all()
    assert np.isfinite(decoh["value_scalar"].to_numpy()).all()
    assert np.isfinite(grad["value_scalar"].to_numpy()).all()
    assert decoh["value_scalar"].ge(0.0).all()
    assert grad["value_scalar"].ge(0.0).all()

    # Post-update sparse rows label checkpoint s as step s+1. Bring the replay
    # metric back to update index s; pre-update probe gradients already use s.
    decoh["checkpoint_update"] = decoh["step"].astype(int) - 1
    grad["checkpoint_update"] = grad["step"].astype(int)
    assert tuple(sorted(decoh["checkpoint_update"].unique())) == run["deep_steps"]

    key_info = decoh[["param_role", "layer", "parameter_name"]].drop_duplicates()
    assert key_info["parameter_name"].notna().all()
    assert not key_info.duplicated(["param_role", "layer"]).any()
    assert not key_info.duplicated(["parameter_name"]).any()
    # Sparse probe-gradient rows intentionally omit parameter_name. For Muon
    # matrices, (param_role, layer) is one-to-one with the replay row name.
    grad = grad.drop(columns="parameter_name").merge(
        key_info, on=["param_role", "layer"], how="inner", validate="many_to_one"
    )
    assert tuple(sorted(grad["checkpoint_update"].unique())) == run["deep_steps"]
    assert grad["parameter_name"].nunique() == decoh["parameter_name"].nunique()

    expected_rows = len(key_info) * len(run["deep_steps"])
    assert len(decoh) == expected_rows
    assert len(grad) == expected_rows
    assert not decoh.duplicated(["parameter_name", "checkpoint_update"]).any()
    assert not grad.duplicated(["parameter_name", "checkpoint_update"]).any()

    for frame in (decoh, grad):
        frame["exact_zero"] = exact_zero(frame["value_scalar"])
        frame["run_id"] = run["run_id"]
        frame["seed"] = run["seed"]
        frame["depth"] = run["depth"]
        frame["horizon"] = run["horizon"]

    def first_nonzero(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        out = (
            frame.loc[~frame["exact_zero"]]
            .groupby("parameter_name", as_index=False)["checkpoint_update"]
            .min()
            .rename(columns={"checkpoint_update": f"{label}_wake_update"})
        )
        assert len(out) == len(key_info), (run["run_id"], label, len(out), len(key_info))
        return out

    wake = key_info.merge(first_nonzero(decoh, "decoherence"), on="parameter_name")
    wake = wake.merge(first_nonzero(grad, "gradient"), on="parameter_name")
    wake["layer"] = wake["layer"].astype(int)
    wake["layer_normalized"] = wake["layer"] / (run["depth"] - 1)
    wake["run_id"] = run["run_id"]
    wake["seed"] = run["seed"]
    wake["depth"] = run["depth"]
    wake["horizon"] = run["horizon"]

    d0 = decoh.loc[decoh["checkpoint_update"] == 0, ["parameter_name", "value_scalar", "exact_zero"]]
    d0 = d0.rename(
        columns={
            "value_scalar": "decoherence_at_update0",
            "exact_zero": "decoherence_exact_zero_at_update0",
        }
    )
    g0 = grad.loc[grad["checkpoint_update"] == 0, ["parameter_name", "value_scalar", "exact_zero"]]
    g0 = g0.rename(
        columns={
            "value_scalar": "gradient_sq_norm_at_update0",
            "exact_zero": "gradient_exact_zero_at_update0",
        }
    )
    wake = wake.merge(d0, on="parameter_name", validate="one_to_one")
    wake = wake.merge(g0, on="parameter_name", validate="one_to_one")
    wake["wake_order_matches"] = wake["decoherence_wake_update"].eq(wake["gradient_wake_update"])
    wake["role_rule_wake_update"] = np.where(wake["param_role"].isin(OUTPUT_ROLES), 0, 1)
    wake["role_rule_matches_both"] = (
        wake["decoherence_wake_update"].eq(wake["role_rule_wake_update"])
        & wake["gradient_wake_update"].eq(wake["role_rule_wake_update"])
    )

    long = pd.concat(
        [
            decoh.assign(channel="decoherence"),
            grad.assign(channel="gradient_sq_norm"),
        ],
        ignore_index=True,
    )

    periodic_muon = periodic[periodic["metric"].str.startswith("muon/")].copy()
    stage_audit = []
    for family, rows in periodic_muon.groupby("metric", sort=True):
        assert rows["parameter_name"].notna().all()
        channels = rows["parameter_name"].nunique()
        assert channels == len(key_info)
        defined_rows = defined(rows)
        zero_rows = exact_zero(defined_rows["value_scalar"]).sum()
        measured_steps = sorted(int(x) for x in rows["step"].unique())
        stage_audit.append(
            {
                "run_id": run["run_id"],
                "depth": run["depth"],
                "seed": run["seed"],
                "metric": family,
                "matrix_channels": channels,
                "rows": len(rows),
                "defined_rows": len(defined_rows),
                "undefined_rows": int((~rows["is_defined"]).sum()),
                "exact_zero_defined_rows": int(zero_rows),
                "first_periodic_steps": ",".join(str(x) for x in measured_steps[:3]),
                "early_deep_overlap": ",".join(
                    str(x) for x in sorted(set(measured_steps) & set(EARLY_DEEP_UPDATES))
                ),
            }
        )
    assert len(stage_audit) == 14

    run["undefined_counts"] = undefined_counts
    run["matrix_count"] = len(key_info)
    run["stage_families"] = sorted(periodic_muon["metric"].unique())
    run["first_periodic_steps"] = sorted(int(x) for x in periodic_muon["step"].unique())[:3]
    return wake, long, stage_audit


def seed_agreement(wake: pd.DataFrame) -> pd.DataFrame:
    d12 = wake[wake["depth"] == 12]
    records = []
    for channel, column in [
        ("decoherence", "decoherence_wake_update"),
        ("gradient_sq_norm", "gradient_wake_update"),
    ]:
        pivot = d12.pivot(index="parameter_name", columns="seed", values=column).sort_index()
        assert pivot.shape == (78, 5)
        for seed_a, seed_b in itertools.combinations(sorted(pivot.columns), 2):
            x = pivot[seed_a].to_numpy(dtype=int)
            y = pivot[seed_b].to_numpy(dtype=int)
            tau, counts = kendall_tau_b(x, y)
            relations = []
            for i, j in itertools.combinations(range(len(x)), 2):
                relations.append(int(np.sign(x[i] - x[j])) == int(np.sign(y[i] - y[j])))
            asleep_a = set(pivot.index[x == 1])
            asleep_b = set(pivot.index[y == 1])
            jaccard = len(asleep_a & asleep_b) / len(asleep_a | asleep_b)
            records.append(
                {
                    "channel": channel,
                    "seed_a": int(seed_a),
                    "seed_b": int(seed_b),
                    "matrix_exact_agreement": float(np.mean(x == y)),
                    "pair_relation_agreement": float(np.mean(relations)),
                    "kendall_tau_b": tau,
                    "initial_zero_set_jaccard": jaccard,
                    **counts,
                }
            )
    return pd.DataFrame(records)


def make_seed_figure(wake: pd.DataFrame, path: Path) -> None:
    d12 = wake[wake["depth"] == 12].copy()
    order = (
        d12[d12["seed"] == 7]
        .sort_values(["role_rule_wake_update", "param_role", "layer"])
        ["parameter_name"]
        .tolist()
    )
    role_sizes = (
        d12[d12["seed"] == 7]
        .set_index("parameter_name")
        .loc[order]
        .groupby("param_role", sort=False)
        .size()
    )
    role_bounds = role_sizes.cumsum().to_numpy()
    role_starts = np.r_[0, role_bounds[:-1]]
    role_centers = role_starts + (role_sizes.to_numpy() - 1) / 2
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 10), sharey=True, constrained_layout=True)
    for ax, (title, column) in zip(
        axes,
        [
            ("Replay decoherence", "decoherence_wake_update"),
            ("Probe gradient norm", "gradient_wake_update"),
        ],
    ):
        pivot = d12.pivot(index="parameter_name", columns="seed", values=column).loc[order]
        image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("seed")
        ax.set_xticks(range(5), pivot.columns)
        for boundary in role_bounds[:-1]:
            ax.axhline(boundary - 0.5, color="white", lw=0.6, alpha=0.8)
    axes[0].set_ylabel("parameter role (layers ordered within role)")
    axes[0].set_yticks(role_centers, role_sizes.index)
    colorbar = fig.colorbar(image, ax=axes, shrink=0.55, ticks=[0, 1])
    colorbar.set_label("first nonzero update checkpoint")
    fig.suptitle("Identical d12 wake-up ordering across all five seeds", fontsize=13)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_depth_figure(wake: pd.DataFrame, path: Path) -> None:
    seed7 = wake[wake["seed"] == 7].copy()
    roles = ["attn_out", "mlp_out", "attn_q", "attn_k", "attn_v", "mlp_in", "ve_gate"]
    markers = {12: "o", 14: "s", 16: "^"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True, constrained_layout=True)
    for ax, (title, column) in zip(
        axes,
        [
            ("Replay decoherence", "decoherence_wake_update"),
            ("Probe gradient norm", "gradient_wake_update"),
        ],
    ):
        for depth in (12, 14, 16):
            for role in roles:
                rows = seed7[(seed7["depth"] == depth) & (seed7["param_role"] == role)]
                image = ax.scatter(
                    rows["layer_normalized"],
                    np.full(len(rows), roles.index(role)) + (depth - 14) * 0.025,
                    s=25,
                    alpha=0.78,
                    c=rows[column],
                    cmap="viridis",
                    vmin=0,
                    vmax=1,
                    marker=markers[depth],
                    linewidths=0.25,
                    edgecolors="black",
                )
        ax.set_title(title)
        ax.set_xlabel("normalized layer position  l / (L - 1)")
        ax.set_yticks(range(len(roles)), roles)
        ax.grid(alpha=0.2)
        ax.invert_yaxis()
    axes[0].set_ylabel("parameter role")
    depth_handles = [
        plt.Line2D([], [], marker=markers[d], linestyle="", color="black", label=f"d{d}", markersize=6)
        for d in (12, 14, 16)
    ]
    axes[1].legend(handles=depth_handles, loc="center left", bbox_to_anchor=(1.02, 0.62))
    colorbar = fig.colorbar(image, ax=axes, shrink=0.68, ticks=[0, 1], pad=0.09)
    colorbar.set_label("first nonzero update checkpoint")
    fig.suptitle("Wake-up bin is constant across normalized position within each role", fontsize=13)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    inventory = load_v3_inventory()
    wakes = []
    long_frames = []
    stage_records = []
    for run in inventory:
        wake, long, stage_audit = extract_run(run)
        wakes.append(wake)
        long_frames.append(long)
        stage_records.extend(stage_audit)
    wake = pd.concat(wakes, ignore_index=True).sort_values(
        ["depth", "seed", "role_rule_wake_update", "param_role", "layer"]
    )
    long = pd.concat(long_frames, ignore_index=True)
    stages = pd.DataFrame(stage_records).sort_values(["run_id", "metric"])

    assert len(wake) == 5 * 78 + 91 + 104 == 585
    assert wake["wake_order_matches"].all()
    assert wake["role_rule_matches_both"].all()
    assert int(wake["decoherence_exact_zero_at_update0"].sum()) == 5 * 54 + 63 + 72
    assert wake["decoherence_exact_zero_at_update0"].equals(
        wake["gradient_exact_zero_at_update0"]
    )
    assert int(exact_zero(long["value_scalar"]).sum()) == int(long["exact_zero"].sum())
    # A wake-up is persistent at observed checkpoints: no channel returns to
    # exact zero after its first nonzero checkpoint.
    check = long.merge(
        wake[
            [
                "run_id",
                "parameter_name",
                "decoherence_wake_update",
                "gradient_wake_update",
            ]
        ],
        on=["run_id", "parameter_name"],
        validate="many_to_one",
    )
    check["wake"] = np.where(
        check["channel"].eq("decoherence"),
        check["decoherence_wake_update"],
        check["gradient_wake_update"],
    )
    post_wake_zeros = check[check["checkpoint_update"].ge(check["wake"]) & check["exact_zero"]]
    assert post_wake_zeros.empty

    agreement = seed_agreement(wake)
    assert len(agreement) == 20
    for column in [
        "matrix_exact_agreement",
        "pair_relation_agreement",
        "kendall_tau_b",
        "initial_zero_set_jaccard",
    ]:
        assert agreement[column].eq(1.0).all(), column

    wake.to_csv(HERE / "matrix_wakeup.csv", index=False)
    agreement.to_csv(HERE / "seed_order_agreement.csv", index=False)
    stages.to_csv(HERE / "muon_stage_cadence_audit.csv", index=False)

    role_summary = (
        wake.groupby(["depth", "seed", "param_role"], as_index=False)
        .agg(
            matrices=("parameter_name", "size"),
            decoherence_wake_update=("decoherence_wake_update", "first"),
            gradient_wake_update=("gradient_wake_update", "first"),
            min_normalized_layer=("layer_normalized", "min"),
            max_normalized_layer=("layer_normalized", "max"),
            distinct_decoherence_wake_bins=("decoherence_wake_update", "nunique"),
            distinct_gradient_wake_bins=("gradient_wake_update", "nunique"),
        )
        .sort_values(["depth", "seed", "decoherence_wake_update", "param_role"])
    )
    role_summary.to_csv(HERE / "role_wakeup_summary.csv", index=False)

    make_seed_figure(wake, HERE / "figure-seed-order.png")
    make_depth_figure(wake, HERE / "figure-depth-position.png")

    seed7 = wake[wake["seed"] == 7]
    run_summary = []
    for (run_id, depth, seed, horizon), rows in wake.groupby(
        ["run_id", "depth", "seed", "horizon"], sort=True
    ):
        asleep = int(rows["decoherence_exact_zero_at_update0"].sum())
        run_summary.append(
            {
                "run_id": run_id,
                "depth": int(depth),
                "seed": int(seed),
                "horizon": int(horizon),
                "matrices": len(rows),
                "exact_zero_at_update0": asleep,
                "exact_zero_fraction_at_update0": asleep / len(rows),
                "awake_at_update0": len(rows) - asleep,
                "awake_by_update1": int((rows["decoherence_wake_update"] <= 1).sum()),
            }
        )

    d1_grad = long[
        (long["channel"] == "gradient_sq_norm") & (long["checkpoint_update"] == 1)
    ]["value_scalar"]
    d1_decoh = long[
        (long["channel"] == "decoherence") & (long["checkpoint_update"] == 1)
    ]["value_scalar"]
    initial_grad_nonzero = long[
        (long["channel"] == "gradient_sq_norm")
        & (long["checkpoint_update"] == 0)
        & (~long["exact_zero"])
    ]["value_scalar"]
    initial_decoh_nonzero = long[
        (long["channel"] == "decoherence")
        & (long["checkpoint_update"] == 0)
        & (~long["exact_zero"])
    ]["value_scalar"]

    summary = {
        "segments": [x["segment"] for x in inventory],
        "run_summary": run_summary,
        "run_matrix_instances": len(wake),
        "checkpoint_resolved_wake_channels": 2 * len(wake),
        "sparse_scalar_rows_tested": len(long),
        "muon_metric_families": 15,
        "periodic_muon_stage_families": len(stages["metric"].unique()),
        "periodic_stage_run_matrix_channels_audited": int(stages["matrix_channels"].sum()),
        "all_run_matrix_channels_inspected_including_gradient": int(
            stages["matrix_channels"].sum() + 2 * len(wake)
        ),
        "undefined_relevant_rows": {
            x["run_id"]: x["undefined_counts"] for x in inventory
        },
        "exact_zero_test": "defined finite value_scalar == 0.0; no tolerance/isclose/rounding",
        "negative_signed_zero_count": int(
            np.signbit(long.loc[long["exact_zero"], "value_scalar"].to_numpy()).sum()
        ),
        "post_wake_exact_zero_rows": len(post_wake_zeros),
        "d12_seed_pair_comparisons_per_channel": 10,
        "d12_seed_matrix_exact_agreement_min": float(agreement["matrix_exact_agreement"].min()),
        "d12_seed_pair_relation_agreement_min": float(agreement["pair_relation_agreement"].min()),
        "d12_seed_kendall_tau_b_min": float(agreement["kendall_tau_b"].min()),
        "d12_seed_pair_relations_compared_per_channel": int(
            10 * math.comb(78, 2)
        ),
        "d12_seed_ordered_untied_pairs_compared_per_channel": int(10 * 24 * 54),
        "seed7_cross_depth_matrices": len(seed7),
        "seed7_role_rule_accuracy_both_channels": float(seed7["role_rule_matches_both"].mean()),
        "smallest_nonzero_gradient_sq_norm_at_update1": float(d1_grad.min()),
        "smallest_nonzero_decoherence_at_update1": float(d1_decoh.min()),
        "smallest_nonzero_gradient_sq_norm_at_update0": float(initial_grad_nonzero.min()),
        "smallest_nonzero_decoherence_at_update0": float(initial_decoh_nonzero.min()),
        "normalized_update1_progress": {
            f"d{depth}": 1 / int(rows["horizon"].iloc[0])
            for depth, rows in seed7.groupby("depth")
        },
        "layer_normalization": "layer / (depth - 1)",
        "all_depth_role_groups_have_constant_wake_across_layers": bool(
            role_summary[
                ["distinct_decoherence_wake_bins", "distinct_gradient_wake_bins"]
            ].eq(1).all().all()
        ),
        "stage_family_names": sorted(stages["metric"].unique()),
        "stage_early_deep_overlap": sorted(stages["early_deep_overlap"].unique()),
        "first_periodic_steps_by_run": {
            x["run_id"]: x["first_periodic_steps"] for x in inventory
        },
    }
    with (HERE / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
