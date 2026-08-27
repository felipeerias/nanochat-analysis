#!/usr/bin/env python
"""I0002 / A0001 — bf16 native vs IEEE-fp32 shadow curvature distortion.

Protocol: investigations/0002-bf16-vs-fp32-curvature/README.md @ e76859c

Selection (literal reading of the protocol):
  - sparse tier, seven schema-v3 segments (d12-iter excluded: schema v1, no
    shadow arm).
  - every curvature/* and update/* SCALAR family (aggregation == "scalar")
    present in BOTH acceptance arms.
  - a pair is (segment, step, metric) where BOTH arms have is_defined == True
    and a finite value_scalar. No verdict conditioning: the protocol forbids
    it (the native arm's checkpoint verdict fails everywhere).

Outputs (written next to this file):
  pairs.csv.gz              every paired observation
  availability.csv          per-family defined counts per arm (pairing loss)
  per_metric.csv            the table the protocol asks for
  per_metric_depth.csv      / per_metric_depth_wide.csv
  seed_spread.csv           d12 five-seed spread of per-run median distortion
  trend.csv                 within-run distortion-vs-progress trends
  verdict_sensitivity.csv   secondary, verdict-conditioned check
  summary.json
  figures/*.png
"""

import glob
import json
import os
import re

import numpy as np
import pandas as pd
import pyarrow.dataset as pds

from loader.paths import DEFAULT_DATA_ROOT

ROOT = str(DEFAULT_DATA_ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

SEGMENTS = [s for s in sorted(os.listdir(ROOT)) if not s.startswith("d12-iter")]
RNG = np.random.default_rng(20260825)

COLS = ["metric", "aggregation", "step", "normalized_progress", "value_scalar",
        "is_defined", "undefined_reason", "acceptance_arm", "dtype", "run_id"]


# ------------------------------------------------------------- metric class
def classify(m):
    """Group families by what they physically are.

    quadratic  — the curvature observables themselves (energies / step sizes)
    update     — measured update effectiveness (loss decrease, model quality)
    error      — acceptance ERROR measures (symmetry / linearity residuals)
    snr        — signal-to-noise ratios of the same probes
    floor      — arithmetic floors, epsilons and thresholds: these are set BY
                 the arm's arithmetic, so a difference is definitional
    flag       — booleans / ordinal verdict codes: ratios are meaningless
    """
    if m.startswith("update/"):
        return "update"
    n = m.split("/", 1)[1]
    if n.startswith("verdict_code") or n.startswith("fd_conclusive"):
        return "flag"
    if ("floor" in n or n == "arith_eps" or n.endswith("_threshold")
            or re.match(r"^(fd_eps|curv_eps|sweep_eps)", n)):
        return "floor"
    if "snr" in n:
        return "snr"
    if re.match(r"^(e_lin|e_sym|e_curv|e_fd|c_fd|fd_cos)", n):
        return "error"
    return "quadratic"


def run_key(seg):
    return seg.split("-s0-")[0]


def depth_of(seg):
    return int(run_key(seg).split("-")[0][1:])


def load_sparse(seg):
    files = sorted(glob.glob(os.path.join(ROOT, seg, "sparse", "*.parquet")))
    df = pds.dataset(files, format="parquet").to_table(columns=COLS).to_pandas()
    df["run"] = run_key(seg)
    df["depth"] = depth_of(seg)
    df["segment"] = seg
    return df


# ------------------------------------------------------------------- load
raw = pd.concat([load_sparse(s) for s in SEGMENTS], ignore_index=True)
raw = raw[raw["metric"].str.startswith(("curvature/", "update/"))].copy()

n_all_families = raw["metric"].nunique()
vector_families = sorted(raw.loc[raw["aggregation"] != "scalar", "metric"].unique())
raw = raw[raw["aggregation"] == "scalar"].copy()
scalar_families = sorted(raw["metric"].unique())

arms_per_metric = raw.groupby("metric")["acceptance_arm"].agg(lambda s: frozenset(s))
both_arm_families = sorted(m for m, a in arms_per_metric.items()
                           if {"native", "shadow_fp32"} <= set(a))
one_arm_only = sorted(set(scalar_families) - set(both_arm_families))
raw = raw[raw["metric"].isin(both_arm_families)].copy()
raw["class"] = raw["metric"].map(classify)

# --------------------------------------------------------------- pair up
key = ["run", "depth", "segment", "step", "normalized_progress", "metric"]
nat = raw[raw["acceptance_arm"] == "native"]
sha = raw[raw["acceptance_arm"] == "shadow_fp32"]
nat = nat[key + ["value_scalar", "is_defined", "undefined_reason"]].rename(
    columns={"value_scalar": "native", "is_defined": "nat_defined",
             "undefined_reason": "nat_reason"})
sha = sha[key + ["value_scalar", "is_defined", "undefined_reason"]].rename(
    columns={"value_scalar": "shadow", "is_defined": "sha_defined",
             "undefined_reason": "sha_reason"})
assert not nat.duplicated(key).any() and not sha.duplicated(key).any()
slots = nat.merge(sha, on=key, how="inner", validate="one_to_one")
n_slots = len(slots)

# availability: what each arm can even define, before any distortion question
avail = (slots.groupby("metric")
         .agg(n_checkpoints=("step", "size"),
              n_native_defined=("nat_defined", "sum"),
              n_shadow_defined=("sha_defined", "sum"),
              n_both=("nat_defined", lambda s: 0))  # filled below
         .reset_index())
bothmask = slots["nat_defined"] & slots["sha_defined"]
avail["n_both"] = (slots.assign(b=bothmask).groupby("metric")["b"].sum()
                   .reindex(avail["metric"]).to_numpy())
avail["class"] = avail["metric"].map(classify)
avail["native_avail"] = avail["n_native_defined"] / avail["n_checkpoints"]
avail["shadow_avail"] = avail["n_shadow_defined"] / avail["n_checkpoints"]
avail["pair_avail"] = avail["n_both"] / avail["n_checkpoints"]
nat_reasons = (slots.loc[~slots["nat_defined"]]
               .groupby(["metric", "nat_reason"]).size()
               .rename("n").reset_index())

# EXPLICIT defined filtering (never implicit)
ok = bothmask & np.isfinite(slots["native"]) & np.isfinite(slots["shadow"])
pairs = slots[ok].copy()
pairs["class"] = pairs["metric"].map(classify)
pairs["abs_diff"] = pairs["native"] - pairs["shadow"]
with np.errstate(divide="ignore", invalid="ignore"):
    pairs["rel_diff"] = pairs["abs_diff"] / np.abs(pairs["shadow"])
pairs.loc[pairs["shadow"] == 0.0, "rel_diff"] = np.nan
pairs["abs_rel"] = np.abs(pairs["rel_diff"])
pairs["sign_disagree"] = np.sign(pairs["native"]) != np.sign(pairs["shadow"])
pairs["either_zero"] = (pairs["native"] == 0.0) | (pairs["shadow"] == 0.0)
pairs["exact_equal"] = pairs["native"] == pairs["shadow"]


# --------------------------------------------------------------- helpers
def spearman(x, y, n_perm=10000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 5:
        return np.nan, np.nan, len(x)
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return np.nan, np.nan, len(x)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    if n_perm:
        perm = np.array([np.corrcoef(rx, RNG.permutation(ry))[0, 1]
                         for _ in range(n_perm)])
        p = float((np.abs(perm) >= abs(rho) - 1e-15).mean())
    else:
        p = np.nan
    return rho, p, len(x)


def q(s, p):
    s = np.asarray(pd.Series(s).dropna(), float)
    return float(np.quantile(s, p)) if len(s) else np.nan


def cluster_boot_median_ci(df, col, by="run", B=2000):
    """Bootstrap CI for the median, resampling RUNS (clusters), not rows."""
    runs = df[by].unique()
    if len(runs) < 2:
        return (np.nan, np.nan)
    groups = {r: np.asarray(df.loc[df[by] == r, col].dropna(), float)
              for r in runs}
    out = []
    for _ in range(B):
        pick = RNG.choice(runs, size=len(runs), replace=True)
        v = np.concatenate([groups[r] for r in pick])
        if len(v):
            out.append(np.median(v))
    if not out:
        return (np.nan, np.nan)
    return (float(np.quantile(out, .025)), float(np.quantile(out, .975)))


# ------------------------------------------------------------ per-metric
rows = []
for m, g in pairs.groupby("metric"):
    r, a = g["rel_diff"], g["abs_diff"]
    nz = g[~g["either_zero"]]
    lo, hi = cluster_boot_median_ci(g, "abs_rel")
    rho, p, nrho = spearman(g["normalized_progress"], g["abs_rel"])
    rows.append(dict(
        metric=m, mclass=classify(m), n_pairs=len(g),
        n_rel=int(r.notna().sum()),
        frac_exact_equal=float(g["exact_equal"].mean()),
        median_rel=q(r, .5), q25_rel=q(r, .25), q75_rel=q(r, .75),
        iqr_rel=q(r, .75) - q(r, .25),
        median_abs_rel=q(g["abs_rel"], .5),
        ci_lo=lo, ci_hi=hi,
        p90_abs_rel=q(g["abs_rel"], .90),
        max_abs_rel=float(g["abs_rel"].max()) if g["abs_rel"].notna().any() else np.nan,
        median_abs_diff=q(a, .5), iqr_abs_diff=q(a, .75) - q(a, .25),
        median_shadow=q(g["shadow"], .5), median_native=q(g["native"], .5),
        sign_disagree_frac=float(g["sign_disagree"].mean()),
        sign_disagree_frac_nonzero=(float(nz["sign_disagree"].mean())
                                    if len(nz) else np.nan),
        n_nonzero=len(nz), frac_either_zero=float(g["either_zero"].mean()),
        rho_progress=rho, p_progress=p, n_progress=nrho))
per_metric = pd.DataFrame(rows)

# families with no usable pair at all
lost = sorted(set(both_arm_families) - set(per_metric["metric"]))
for m in lost:
    per_metric = pd.concat([per_metric, pd.DataFrame([dict(
        metric=m, mclass=classify(m), n_pairs=0, n_rel=0)])], ignore_index=True)
per_metric = per_metric.sort_values(["mclass", "median_abs_rel"],
                                    ascending=[True, False])

# ---------------------------------------------------- trend over training
trend_rows = []
for m, g in pairs.groupby("metric"):
    per_run = []
    for r_, gg in g.groupby("run"):
        rho, _, n = spearman(gg["normalized_progress"], gg["abs_rel"],
                             n_perm=0)
        if np.isfinite(rho):
            per_run.append(rho)
    early = g[g["normalized_progress"] < .5]["abs_rel"]
    late = g[g["normalized_progress"] >= .5]["abs_rel"]
    rho_p, p_p, n_p = spearman(g["normalized_progress"], g["abs_rel"])
    trend_rows.append(dict(
        metric=m, mclass=classify(m), n_runs=len(per_run),
        rho_pooled=rho_p, p_pooled=p_p,
        rho_run_median=float(np.median(per_run)) if per_run else np.nan,
        rho_run_min=float(np.min(per_run)) if per_run else np.nan,
        rho_run_max=float(np.max(per_run)) if per_run else np.nan,
        n_runs_positive=int(np.sum(np.asarray(per_run) > 0)) if per_run else 0,
        median_abs_rel_early=q(early, .5), median_abs_rel_late=q(late, .5),
        late_over_early=(q(late, .5) / q(early, .5)
                         if q(early, .5) not in (0.0, np.nan) else np.nan)))
trend = pd.DataFrame(trend_rows).sort_values("mclass")

# ------------------------------------------------------------- depth
depth_rows = []
for (m, d), g in pairs.groupby(["metric", "depth"]):
    depth_rows.append(dict(metric=m, mclass=classify(m), depth=d, n=len(g),
                           median_rel=q(g["rel_diff"], .5),
                           median_abs_rel=q(g["abs_rel"], .5)))
per_depth = pd.DataFrame(depth_rows)

seed_rows = []
for (m, r_), g in pairs[pairs["depth"] == 12].groupby(["metric", "run"]):
    seed_rows.append(dict(metric=m, run=r_, median_abs_rel=q(g["abs_rel"], .5)))
seed_tab = pd.DataFrame(seed_rows)
seed_spread = (seed_tab.groupby("metric")["median_abs_rel"]
               .agg(n_seeds="count", mean="mean", sd="std", lo="min", hi="max")
               .reset_index())
seed_spread["sd_rel"] = seed_spread["sd"] / seed_spread["mean"].abs()
seed_spread["range_rel"] = (seed_spread["hi"] - seed_spread["lo"]) / seed_spread["mean"].abs()

dw = per_depth.pivot(index="metric", columns="depth", values="median_abs_rel")
dw.columns = [f"d{c}" for c in dw.columns]
dw = dw.join(seed_spread.set_index("metric")[["sd_rel", "range_rel", "lo", "hi"]])
dw["d16_over_d12"] = dw["d16"] / dw["d12"]
dw["d14_over_d12"] = dw["d14"] / dw["d12"]
dw["d16_outside_d12_seed_range"] = (dw["d16"] < dw["lo"]) | (dw["d16"] > dw["hi"])
dw["d14_outside_d12_seed_range"] = (dw["d14"] < dw["lo"]) | (dw["d14"] > dw["hi"])
dw["mclass"] = [classify(m) for m in dw.index]

# ------------------------------------ pooled-by-class depth vs seed floor
depth_class_rows = []
for c, g in pairs.groupby("class"):
    per_run = g.groupby("run")["abs_rel"].median()
    d12_runs = per_run[[r for r in per_run.index if r.startswith("d12")]]
    m12, s12 = float(d12_runs.mean()), float(d12_runs.std())
    row = dict(mclass=c,
               d12_pooled=q(g[g["depth"] == 12]["abs_rel"], .5),
               d14=q(g[g["depth"] == 14]["abs_rel"], .5),
               d16=q(g[g["depth"] == 16]["abs_rel"], .5),
               d12_seed_mean=m12, d12_seed_sd=s12,
               d12_seed_sd_rel=s12 / m12 if m12 else np.nan,
               d12_seed_lo=float(d12_runs.min()), d12_seed_hi=float(d12_runs.max()))
    row["d16_over_d12"] = row["d16"] / row["d12_pooled"]
    row["d16_z_vs_d12_seeds"] = (row["d16"] - m12) / s12 if s12 else np.nan
    row["d14_z_vs_d12_seeds"] = (row["d14"] - m12) / s12 if s12 else np.nan
    depth_class_rows.append(row)
depth_class = pd.DataFrame(depth_class_rows)

# per-family monotonicity in depth
mono = dw.dropna(subset=["d12", "d14", "d16"]).copy()
mono["monotone_up"] = (mono["d12"] < mono["d14"]) & (mono["d14"] < mono["d16"])
mono["monotone_down"] = (mono["d12"] > mono["d14"]) & (mono["d14"] > mono["d16"])
mono_counts = (mono.groupby("mclass")[["monotone_up", "monotone_down"]].sum()
               .join(mono.groupby("mclass").size().rename("n")))

# ------------------------------------------------- pooled class statements
class_rows = []
for c, g in pairs.groupby("class"):
    lo, hi = cluster_boot_median_ci(g, "abs_rel")
    rho, p, n = spearman(g["normalized_progress"], g["abs_rel"])
    class_rows.append(dict(mclass=c, n_families=g["metric"].nunique(),
                           n_pairs=len(g),
                           median_rel=q(g["rel_diff"], .5),
                           median_abs_rel=q(g["abs_rel"], .5),
                           ci_lo=lo, ci_hi=hi,
                           q25=q(g["rel_diff"], .25), q75=q(g["rel_diff"], .75),
                           sign_disagree=float(g["sign_disagree"].mean()),
                           rho_progress=rho, p_progress=p))
per_class = pd.DataFrame(class_rows)

PHYS = ["curvature/gHg", "curvature/dhd", "curvature/vhv_gradient",
        "curvature/vhv_random", "curvature/vhv_update"]
phys = pairs[pairs["metric"].isin(PHYS)]
rho_phys, p_phys, n_phys = spearman(phys["normalized_progress"], phys["abs_rel"])
phys_lo, phys_hi = cluster_boot_median_ci(phys, "abs_rel")

# ------------------------------- secondary: verdict-conditioned sensitivity
vd = raw[raw["metric"].str.startswith("curvature/verdict_code_")]
vd = vd[vd["is_defined"] & (vd["acceptance_arm"] == "shadow_fp32")]
vd["direction"] = vd["metric"].str.replace("curvature/verdict_code_", "", regex=False)
passing = vd[vd["value_scalar"] == 0.0][["run", "step", "direction"]]
vs_rows = []
for direction in ["gradient", "random", "update"]:
    ps = passing[passing["direction"] == direction][["run", "step"]]
    fams = [m for m in pairs["metric"].unique() if m.endswith("_" + direction)]
    sel = pairs[pairs["metric"].isin(fams)].merge(ps, on=["run", "step"])
    for m, g in sel.groupby("metric"):
        vs_rows.append(dict(metric=m, direction=direction, n_pairs=len(g),
                            median_abs_rel=q(g["abs_rel"], .5),
                            median_rel=q(g["rel_diff"], .5),
                            sign_disagree=float(g["sign_disagree"].mean())))
verdict_sens = pd.DataFrame(vs_rows)
shadow_pass_counts = (passing.groupby("direction").size().to_dict())

# ------------------------------------------------------------------ write
pairs.to_csv(os.path.join(HERE, "pairs.csv.gz"), index=False, compression="gzip")
avail.to_csv(os.path.join(HERE, "availability.csv"), index=False)
nat_reasons.to_csv(os.path.join(HERE, "native_undefined_reasons.csv"), index=False)
per_metric.to_csv(os.path.join(HERE, "per_metric.csv"), index=False)
per_depth.to_csv(os.path.join(HERE, "per_metric_depth.csv"), index=False)
dw.reset_index().to_csv(os.path.join(HERE, "per_metric_depth_wide.csv"), index=False)
seed_spread.to_csv(os.path.join(HERE, "seed_spread.csv"), index=False)
trend.to_csv(os.path.join(HERE, "trend.csv"), index=False)
per_class.to_csv(os.path.join(HERE, "per_class.csv"), index=False)
depth_class.to_csv(os.path.join(HERE, "depth_vs_seed_floor.csv"), index=False)
verdict_sens.to_csv(os.path.join(HERE, "verdict_sensitivity.csv"), index=False)

summary = dict(
    protocol_commit="e76859c",
    segments=SEGMENTS,
    n_families_curv_update=int(n_all_families),
    n_vector_families_excluded=len(vector_families),
    n_scalar_families=len(scalar_families),
    n_scalar_families_both_arms=len(both_arm_families),
    scalar_families_one_arm_only=one_arm_only,
    n_families_with_zero_pairs=len(lost),
    families_with_zero_pairs=lost,
    n_pair_slots=int(n_slots),
    n_pairs_both_defined=int(len(pairs)),
    n_pairs_rel_defined=int(pairs["rel_diff"].notna().sum()),
    shadow_direction_pass_counts=shadow_pass_counts,
    pooled_physical_curvature=dict(
        metrics=PHYS, n=int(len(phys)),
        median_abs_rel=q(phys["abs_rel"], .5),
        ci95=[phys_lo, phys_hi],
        median_rel=q(phys["rel_diff"], .5),
        q25=q(phys["rel_diff"], .25), q75=q(phys["rel_diff"], .75),
        p90_abs_rel=q(phys["abs_rel"], .90),
        max_abs_rel=float(phys["abs_rel"].max()),
        sign_disagree=float(phys["sign_disagree"].mean()),
        rho_progress=rho_phys, p_progress=p_phys, n_progress=n_phys),
    per_class=per_class.to_dict("records"),
    depth_vs_seed_floor=depth_class.to_dict("records"),
    depth_monotonicity=mono_counts.reset_index().to_dict("records"),
)
with open(os.path.join(HERE, "summary.json"), "w") as f:
    json.dump(summary, f, indent=1, default=float)

# ----------------------------------------------------------------- figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_COLOR = {"quadratic": "#1f77b4", "update": "#2ca02c", "error": "#d62728",
               "snr": "#9467bd", "floor": "#8c564b", "flag": "#7f7f7f"}

# fig 1 — every paired family, |rel| distribution, ordered and coloured by class
order = (per_metric[per_metric["n_pairs"] > 0]
         .sort_values("median_abs_rel"))
fig, ax = plt.subplots(figsize=(10, 13))
data, labels, colors = [], [], []
for _, r in order.iterrows():
    v = pairs.loc[pairs["metric"] == r["metric"], "abs_rel"].dropna()
    if len(v) == 0:
        continue
    data.append(np.maximum(v, 1e-6))  # clip exact zeros so the log axis works
    labels.append(r["metric"])
    colors.append(CLASS_COLOR[r["mclass"]])
bp = ax.boxplot(data, orientation="horizontal", tick_labels=labels,
                whis=(5, 95), showfliers=False, patch_artist=True)
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(.55)
ax.set_xscale("log")
ax.axvline(1.0, color="crimson", ls="--", lw=1)
ax.axvline(0.01, color="gray", ls=":", lw=1)
ax.set_xlabel("|native - shadow| / |shadow|   (log scale; dotted = 1%, dashed = 100%;\n"
              "exact zeros clipped to 1e-6)")
ax.tick_params(labelsize=7)
handles = [plt.Line2D([], [], color=c, lw=6, alpha=.55, label=k)
           for k, c in CLASS_COLOR.items()]
ax.legend(handles=handles, fontsize=8, loc="lower right")
ax.set_title("bf16 native vs IEEE-fp32 shadow: paired relative distortion\n"
             "7 schema-v3 runs, 215 deep checkpoints, all paired scalar families")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "rel_distortion_all_families.png"), dpi=140)
plt.close(fig)

# fig 2 — distortion vs normalized progress
panel = PHYS + ["curvature/eta_star"]
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
for axi, m in zip(axes.ravel(), panel):
    g = pairs[pairs["metric"] == m]
    for d, mk in ((12, "o"), (14, "s"), (16, "^")):
        gg = g[g["depth"] == d]
        axi.scatter(gg["normalized_progress"], gg["abs_rel"], s=12, marker=mk,
                    alpha=.6, label=f"d{d}")
    axi.set_yscale("log")
    axi.set_title(m, fontsize=9)
    axi.grid(alpha=.25)
axes[0, 0].legend(fontsize=8)
for axi in axes[1]:
    axi.set_xlabel("normalized_progress")
axes[0, 0].set_ylabel("|relative distortion|")
axes[1, 0].set_ylabel("|relative distortion|")
fig.suptitle("|relative distortion| over training, by depth", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "distortion_vs_progress.png"), dpi=140)
plt.close(fig)

# fig 3 — paired scatter
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for axi, m in zip(axes, ["curvature/gHg", "curvature/dhd", "curvature/eta_star"]):
    g = pairs[pairs["metric"] == m]
    sc = axi.scatter(g["shadow"], g["native"], s=14, alpha=.7,
                     c=g["normalized_progress"], cmap="viridis")
    lim = np.array([min(g["shadow"].min(), g["native"].min()),
                    max(g["shadow"].max(), g["native"].max())])
    axi.plot(lim, lim, "k--", lw=.8)
    axi.set_xlabel("shadow_fp32")
    axi.set_ylabel("native bf16")
    axi.set_title(m, fontsize=9)
fig.colorbar(sc, ax=axes, label="normalized_progress", fraction=.02)
fig.suptitle("paired native vs shadow — the quadratic forms agree to <1%", fontsize=11)
fig.savefig(os.path.join(FIG, "native_vs_shadow.png"), dpi=140)
plt.close(fig)

# fig 4 — availability: what bf16 cannot define at all
av = avail.sort_values("native_avail")
av = av[av["native_avail"] < 1.0]
fig, ax = plt.subplots(figsize=(8, max(3, .28 * len(av) + 1.5)))
y = np.arange(len(av))
ax.barh(y - .2, av["native_avail"], height=.38, label="native (bf16)", color="#d62728")
ax.barh(y + .2, av["shadow_avail"], height=.38, label="shadow (fp32)", color="#1f77b4")
ax.set_yticks(y)
ax.set_yticklabels(av["metric"], fontsize=8)
ax.set_xlabel("fraction of the 215 deep checkpoints where the value is DEFINED")
ax.legend(fontsize=8)
ax.set_title("Availability loss: families bf16 cannot define")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "availability.png"), dpi=140)
plt.close(fig)

# fig 5 — depth vs the five-seed floor, pooled by class
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for axi, c in zip(axes, ["quadratic", "update"]):
    g = pairs[pairs["class"] == c]
    per_run = g.groupby(["run", "depth"])["abs_rel"].median().reset_index()
    d12r = per_run[per_run["depth"] == 12]
    axi.scatter(np.full(len(d12r), 12) + RNG.normal(0, .06, len(d12r)),
                d12r["abs_rel"], s=45, color="#1f77b4",
                label="d12 seeds (5 runs)")
    axi.axhspan(d12r["abs_rel"].min(), d12r["abs_rel"].max(),
                color="#1f77b4", alpha=.12, label="d12 five-seed range")
    for d in (14, 16):
        v = per_run[per_run["depth"] == d]
        axi.scatter(v["depth"], v["abs_rel"], s=70, marker="^",
                    color="#d62728")
    axi.set_xticks([12, 14, 16])
    axi.set_xlabel("depth (co-varies with width, batch, LR, horizon)")
    axi.set_ylabel("median |relative distortion| per run")
    axi.set_title(f"{c} families", fontsize=10)
    axi.grid(alpha=.25)
axes[0].legend(fontsize=8)
fig.suptitle("Distortion vs depth against the d12 five-seed floor", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "depth_vs_seed_floor.png"), dpi=140)
plt.close(fig)

# ------------------------------------------------------------------ report
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
print(f"families(curv+update)={n_all_families}  scalar={len(scalar_families)}  "
      f"vector-excluded={len(vector_families)}  both-arms={len(both_arm_families)}  "
      f"one-arm-only={one_arm_only}")
print(f"pair slots={n_slots}  both-defined={len(pairs)}  "
      f"rel-defined={int(pairs['rel_diff'].notna().sum())}  "
      f"zero-pair families={len(lost)} {lost}")
print()
print("=== per class ===")
print(per_class.to_string(index=False))
print()
print("=== per metric ===")
cols = ["metric", "mclass", "n_pairs", "median_rel", "q25_rel", "q75_rel",
        "median_abs_rel", "ci_lo", "ci_hi", "p90_abs_rel",
        "sign_disagree_frac", "rho_progress", "p_progress"]
print(per_metric[cols].to_string(index=False))
print()
print("=== pooled physical curvature ===")
print(json.dumps(summary["pooled_physical_curvature"], indent=1, default=float))
print()
print("=== depth vs the d12 five-seed floor (pooled per class) ===")
print(depth_class.to_string(index=False))
print()
print("=== depth monotonicity of median |rel| (d12<d14<d16) ===")
print(mono_counts.to_string())
print()
print("=== trend over training (within-run Spearman, |rel| vs progress) ===")
tcols = ["metric", "mclass", "rho_pooled", "p_pooled", "rho_run_median",
         "n_runs_positive", "n_runs", "median_abs_rel_early",
         "median_abs_rel_late", "late_over_early"]
print(trend[trend["mclass"].isin(["quadratic", "update"])][tcols]
      .to_string(index=False))
print()
print("=== shadow per-direction passing checkpoints (of 215) ===",
      shadow_pass_counts)
print("=== verdict-conditioned sensitivity (shadow gradient passing only) ===")
print(verdict_sens.to_string(index=False))
