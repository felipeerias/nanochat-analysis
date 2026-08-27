"""I0001 / A0001 — d12 seed variation across five runs.

Follows the frozen protocol in ../README.md. Produces:
  channels.csv    per-channel statistics (the detail)
  families.csv    per-family ranking (the product)
  result.md       written by hand from these outputs
  figures/
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
from loader.telemetry_load import DEFAULT_DATA_ROOT, load_segment  # noqa: E402

ROOT = str(DEFAULT_DATA_ROOT)
RUNS = ["d12-s7", "d12-s8", "d12-s9", "d12-s10", "d12-s11"]
TIERS = ("continuous", "periodic", "sparse")
# columns that identify a time series within a run; probe_id is deliberately
# EXCLUDED and replaced by the probe NAME, because probes are drawn per seed
KEY = ["tier", "metric", "acceptance_arm", "param_role", "parameter_name",
       "layer", "head", "optimizer_group_id", "probe_name", "aggregation"]
WARMUP_END = 400          # Muon momentum ramp end; protocol's early/late cut
EPS = 1e-30


def load_all():
    frames, meta = [], {}
    segs = {s.split("-s0-")[0]: s for s in os.listdir(ROOT)}
    for run in RUNS:
        seg = segs[run]
        d = load_segment(ROOT, seg)
        prov = d["provenance"]
        meta[run] = {"segment": seg, "seed": prov["seed"],
                     "iters": prov["num_iterations"]}
        probes = {v: k for k, v in prov["probe_ids"].items()}
        for tier in TIERS:
            df = d["tiers"][tier].copy()
            df["tier"] = tier
            df["run"] = run
            df["probe_name"] = df["probe_id"].map(probes).fillna("-")
            if "acceptance_arm" not in df.columns:
                df["acceptance_arm"] = None
            frames.append(df)
        print(f"  loaded {run}")
    return pd.concat(frames, ignore_index=True), meta


def main():
    print("loading five d12 runs...")
    df, meta = load_all()
    grids = {r: sorted(df[df.run == r].step.unique()) for r in RUNS}
    same_grid = all(grids[r] == grids[RUNS[0]] for r in RUNS)
    print(f"identical step grids across runs: {same_grid}")

    # vector-valued families are not analysed; count them explicitly
    vec = df[df["value_vector"].notna()]
    vec_families = sorted(vec["metric"].unique())
    df = df[df["value_vector"].isna()].copy()

    # undefined rows keep their place but carry no value
    df.loc[~df["is_defined"], "value_scalar"] = np.nan
    for c in KEY:
        df[c] = df[c].fillna("-") if df[c].dtype == object else df[c].fillna(-1)

    print("pivoting to channel x step x run ...")
    wide = df.pivot_table(index=KEY + ["step"], columns="run",
                          values="value_scalar", aggfunc="first")
    wide = wide.reindex(columns=RUNS)
    v = wide.to_numpy(dtype=float)
    n_def = np.sum(~np.isnan(v), axis=1)
    with np.errstate(all="ignore"):
        med = np.nanmedian(v, axis=1)
        sd = np.nanstd(v, axis=1, ddof=1)
        rng = np.nanmax(v, axis=1) - np.nanmin(v, axis=1)
        rel = sd / np.maximum(np.abs(med), EPS)
    pts = wide.index.to_frame(index=False)
    pts["n_def"] = n_def
    pts["median"] = med
    pts["sd"] = sd
    pts["range"] = rng
    pts["rel"] = np.where(np.abs(med) > EPS, rel, np.nan)
    pts["phase"] = np.where(pts["step"] <= WARMUP_END, "early", "late")
    pts = pts[pts["n_def"] >= 2]          # need >= 2 runs to have a spread

    print("aggregating per channel ...")
    def agg(g):
        out = {"n_points": len(g),
               "complete_frac": float((g.n_def == 5).mean()),
               "rel_median": float(np.nanmedian(g["rel"])),
               "rel_worst": float(np.nanmax(g["rel"])) if g["rel"].notna().any() else np.nan,
               "sd_median": float(np.nanmedian(g["sd"])),
               "level_median": float(np.nanmedian(np.abs(g["median"])))}
        # supplementary (beyond the protocol, disclosed in result.md):
        # seed noise relative to how much the channel MOVES during training
        traj = g.sort_values("step")["median"].to_numpy(dtype=float)
        swing = np.nanpercentile(traj, 90) - np.nanpercentile(traj, 10) if len(traj) > 3 else np.nan
        out["swing"] = float(swing) if np.isfinite(swing) else np.nan
        out["noise_vs_swing"] = (float(out["sd_median"] / swing)
                                 if np.isfinite(swing) and abs(swing) > EPS else np.nan)
        return pd.Series(out)

    ch = pts.groupby(KEY, dropna=False).apply(agg, include_groups=False).reset_index()
    ch_phase = (pts.groupby(KEY + ["phase"], dropna=False)["rel"]
                .median().unstack("phase").reset_index()
                .rename(columns={"early": "rel_early", "late": "rel_late"}))
    ch = ch.merge(ch_phase, on=KEY, how="left")
    ch.to_csv(os.path.join(HERE, "channels.csv"), index=False)

    print("aggregating per family ...")
    fam = (ch.groupby(["tier", "metric", "acceptance_arm"], dropna=False)
             .agg(channels=("rel_median", "size"),
                  rel_median=("rel_median", "median"),
                  rel_worst_channel=("rel_median", "max"),
                  rel_worst_point=("rel_worst", "max"),
                  complete_frac=("complete_frac", "median"),
                  noise_vs_swing=("noise_vs_swing", "median"),
                  rel_early=("rel_early", "median"),
                  rel_late=("rel_late", "median"))
             .reset_index()
             .sort_values("rel_median", na_position="last"))
    fam.to_csv(os.path.join(HERE, "families.csv"), index=False)

    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    ok = fam[fam["rel_median"].notna()]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.log10(np.maximum(ok["rel_median"], 1e-12)), bins=40)
    ax.set(xlabel="log10 relative seed spread (median over training)",
           ylabel="metric families",
           title="d12 seed variation: distribution over metric families")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "spread_distribution.png"), dpi=110)
    plt.close(fig)

    summary = {
        "runs": meta, "identical_step_grids": bool(same_grid),
        "vector_families_not_analysed": vec_families,
        "channels": int(len(ch)), "families": int(len(fam)),
        "families_with_defined_spread": int(ok.shape[0]),
        "tightest": ok.head(12)[["tier", "metric", "acceptance_arm",
                                 "rel_median", "channels"]].to_dict("records"),
        "noisiest": ok.tail(12)[["tier", "metric", "acceptance_arm",
                                 "rel_median", "channels"]].to_dict("records"),
        "median_family_spread": float(ok["rel_median"].median()),
        "families_under_1pct": int((ok["rel_median"] < 0.01).sum()),
        "families_over_50pct": int((ok["rel_median"] > 0.5).sum()),
        "early_vs_late": {
            "median_rel_early": float(np.nanmedian(fam["rel_early"])),
            "median_rel_late": float(np.nanmedian(fam["rel_late"]))},
    }
    json.dump(summary, open(os.path.join(HERE, "summary.json"), "w"), indent=2,
              default=str)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("tightest", "noisiest", "runs",
                                   "vector_families_not_analysed")}, indent=2))
    print(f"\nvector families not analysed: {len(vec_families)}")


if __name__ == "__main__":
    main()
