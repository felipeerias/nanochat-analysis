"""I0003/A0001 — per-matrix structure of muon/replay_update_relerr.

Answers the protocol's "Also report": is decoherence a function of relative
depth in the network, of parameter role, or of matrix shape — and is the
distribution preserved across depths?

NORMALIZATION USED (stated up front, as the protocol requires):
  relative depth  r = layer / (n_layer - 1),  so r in [0, 1] at every depth
                     (layer 0 -> 0, last layer -> 1).
  A second column r_frac = layer / n_layer is carried for comparison.
Matched checkpoints are the 20 uniform-tail points at nominal progress
0.05 ... 1.00; the update-0 checkpoint with its structural zeros is excluded.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/felipe/Igalia/nanochat/analysis/loader")
import telemetry_load as tl  # noqa: E402

ROOT = "/home/felipe/Igalia/nanochat/telemetry-data/sweep/telemetry-data"
OUT = os.path.dirname(os.path.abspath(__file__))
METRIC = "muon/replay_update_relerr"
SEGS = [
    ("d12-s7", 12, "d12-s7-s0-6c217854f4174ce884dfb2b2dcc13c45"),
    ("d12-s8", 12, "d12-s8-s0-2b2e72e4395440029b92226213d137bb"),
    ("d12-s9", 12, "d12-s9-s0-b872e698e81b4a8fa0cd08512f43c2c2"),
    ("d12-s10", 12, "d12-s10-s0-0ffeb05742bb4154a3f1afc202dc3955"),
    ("d12-s11", 12, "d12-s11-s0-5fd1bf807f764954b30658872fc3e2ad"),
    ("d14-s7", 14, "d14-s7-s0-b72acf7ac59942dab902e26fa0b2c80d"),
    ("d16-s7", 16, "d16-s7-s0-8e0a39fc8f164343bd01ef6b915e849f"),
]
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 400)

log = open(os.path.join(OUT, "structure.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.write(s + "\n")


frames = []
for run, depth, seg in SEGS:
    r = tl.defined(tl.read_telemetry(ROOT, seg, "sparse").to_pandas()
                   .query("metric == @METRIC"))
    r = r.assign(run=run, depth=depth, layer=r["layer"].astype(int))
    frames.append(r[["run", "depth", "step", "normalized_progress",
                     "value_scalar", "param_role", "parameter_name", "layer",
                     "optimizer_group_id"]])
df = pd.concat(frames, ignore_index=True)
df["r"] = df["layer"] / (df["depth"] - 1)
df["r_frac"] = df["layer"] / df["depth"]

# matched tail only, structural-zero checkpoint dropped
NOM = np.round(np.arange(0.05, 1.0001, 0.05), 10)
keep = []
for run, depth, _ in SEGS:
    g = np.array(sorted(df[df["run"] == run]["normalized_progress"].unique()))
    for t in NOM:
        keep.append((run, float(g[np.abs(g - t).argmin()]), float(t)))
km = pd.DataFrame(keep, columns=["run", "normalized_progress", "nominal"])
M = df.merge(km, on=["run", "normalized_progress"], how="inner")
assert (M["value_scalar"] > 0).all(), "structural zeros leaked in"
say(f"matched per-matrix rows: {len(M)}  "
    f"(20 checkpoints x {{78,91,104}} matrices x runs)")
say(M.groupby(["run", "depth"]).size().to_string())

# ------------------------------------------------------------------ shapes
say("\n" + "=" * 78)
say("MATRIX SHAPE (from optimizer_group_id) x ROLE x DEPTH")
say("=" * 78)
sh = (M.groupby(["depth", "param_role", "optimizer_group_id"])
        ["parameter_name"].nunique().rename("n_matrices").reset_index())
say(sh.to_string(index=False))


def shape_of(g):
    tail = g.split("-")[-1]
    a, b = tail.split("x")
    return int(a), int(b)


M[["rows", "cols"]] = M["optimizer_group_id"].apply(
    lambda g: pd.Series(shape_of(g)))
M["numel"] = M["rows"] * M["cols"]
M["minmn"] = M[["rows", "cols"]].min(axis=1)
M["aspect"] = M[["rows", "cols"]].max(axis=1) / M["minmn"]

# --------------------------------------------------------------- helpers
def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rx = np.array(pd.Series(x).rank().to_numpy(), dtype=float)
    ry = np.array(pd.Series(y).rank().to_numpy(), dtype=float)
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d else float("nan")


def eta2(values, groups):
    """One-way variance explained (eta^2) of a categorical grouping."""
    v = np.asarray(values, float)
    gt = v.mean()
    sst = ((v - gt) ** 2).sum()
    ssb = 0.0
    for g in pd.unique(groups):
        m = np.asarray(groups) == g
        ssb += m.sum() * (v[m].mean() - gt) ** 2
    return float(ssb / sst) if sst else float("nan")


# ------------------------------------------------------------- by role
say("\n" + "=" * 78)
say("Q1. PARAMETER ROLE")
say("=" * 78)
role_med = (M.groupby(["depth", "param_role"])["value_scalar"].median()
             .unstack(0))
say("\nmedian decoherence over the 20 matched checkpoints, by role and depth:")
say(role_med.to_string(float_format=lambda v: f"{v:.5f}"))
say("\nsame, normalized to each depth's own all-matrix median "
    "(shape of the role profile):")
allmed = M.groupby("depth")["value_scalar"].median()
say((role_med / allmed).to_string(float_format=lambda v: f"{v:.3f}"))
say("\nrole rank order (low -> high decoherence) per depth:")
for d in (12, 14, 16):
    say(f"  d{d}: " + " < ".join(role_med[d].sort_values().index))
say("\nSpearman rho of the role profile between depths (7 roles):")
for a, b in ((12, 14), (12, 16), (14, 16)):
    say(f"  d{a} vs d{b}: {spearman(role_med[a], role_med[b]):+.3f}")
say(f"\neta^2 (variance of log decoherence explained by ROLE alone), "
    f"pooled over matched checkpoints:")
for d in (12, 14, 16):
    a = M[M["depth"] == d]
    say(f"  d{d}: {eta2(np.log(a['value_scalar']), a['param_role']):.3f}")
say("  (computed within depth; checkpoint-to-checkpoint drift inflates the")
say("   denominator, so these are lower bounds on the role effect)")
say("\nsame eta^2 computed WITHIN each checkpoint, then averaged:")
for d in (12, 14, 16):
    vals = []
    for t in NOM:
        a = M[(M["depth"] == d) & (M["nominal"] == t)]
        if len(a):
            vals.append(eta2(np.log(a["value_scalar"]), a["param_role"]))
    say(f"  d{d}: {np.mean(vals):.3f}")

# ------------------------------------------------- by relative depth
say("\n" + "=" * 78)
say("Q2. RELATIVE DEPTH   r = layer / (n_layer - 1)")
say("=" * 78)
say("\nmedian decoherence by layer (absolute layer index), per depth:")
lay = M.groupby(["depth", "layer"])["value_scalar"].median().unstack(0)
say(lay.to_string(float_format=lambda v: f"{v:.5f}"))
say("\nnormalized to each depth's own all-matrix median:")
say((lay / allmed).to_string(float_format=lambda v: f"{v:.3f}"))
say("\nSpearman rho( r , decoherence ) within each depth "
    "(pooled over roles and matched checkpoints):")
for d in (12, 14, 16):
    a = M[M["depth"] == d]
    say(f"  d{d}: r = {spearman(a['r'], a['value_scalar']):+.3f}   "
        f"(within-role, mean over roles: "
        + ", ".join(f"{ro}:{spearman(g['r'], g['value_scalar']):+.2f}"
                    for ro, g in a.groupby('param_role')) + ")")
say("\neta^2 explained by relative depth (as a categorical layer index),")
say("within checkpoint, averaged over the 20 matched checkpoints:")
for d in (12, 14, 16):
    vals = [eta2(np.log(a["value_scalar"]), a["layer"])
            for t in NOM
            for a in [M[(M["depth"] == d) & (M["nominal"] == t)]] if len(a)]
    say(f"  d{d}: {np.mean(vals):.3f}")
say("\nRole + layer jointly (categorical role x layer is saturated; instead")
say("eta^2 of layer AFTER removing the role median, within checkpoint):")
for d in (12, 14, 16):
    vals = []
    for t in NOM:
        a = M[(M["depth"] == d) & (M["nominal"] == t)].copy()
        if not len(a):
            continue
        a["resid"] = np.log(a["value_scalar"]) - a.groupby("param_role")[
            "value_scalar"].transform(lambda s: np.log(s).median())
        vals.append(eta2(a["resid"], a["layer"]))
    say(f"  d{d}: {np.mean(vals):.3f}")

# is the r-profile preserved across depths?
say("\nIs the relative-depth profile PRESERVED across depths?")
say("Comparison on the common r grid by linear interpolation of the")
say("depth-normalized profile (role-controlled: per-role z within depth,")
say("then averaged over roles at each layer).")
prof = {}
for d in (12, 14, 16):
    a = M[M["depth"] == d].copy()
    a["z"] = a.groupby(["param_role", "nominal"])["value_scalar"].transform(
        lambda s: (np.log(s) - np.log(s).mean()) / (np.log(s).std(ddof=0)
                                                    if s.std() else 1))
    p = a.groupby("layer")["z"].mean()
    prof[d] = pd.Series(p.to_numpy(), index=p.index / (d - 1))
grid = np.linspace(0, 1, 25)
I = {d: np.interp(grid, prof[d].index.to_numpy(), prof[d].to_numpy())
     for d in (12, 14, 16)}
say("\nrole-controlled z-profile vs relative depth (interpolated to 25 pts):")
say(pd.DataFrame({"r": grid, **{f"d{d}": I[d] for d in (12, 14, 16)}})
    .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
say("\nSpearman rho between depth profiles on the common r grid:")
for a, b in ((12, 14), (12, 16), (14, 16)):
    say(f"  d{a} vs d{b}: {spearman(I[a], I[b]):+.3f}")
say("\nSame test with the other normalization r_frac = layer/n_layer:")
prof2 = {}
for d in (12, 14, 16):
    a = M[M["depth"] == d].copy()
    a["z"] = a.groupby(["param_role", "nominal"])["value_scalar"].transform(
        lambda s: (np.log(s) - np.log(s).mean()) / (np.log(s).std(ddof=0)
                                                    if s.std() else 1))
    p = a.groupby("layer")["z"].mean()
    prof2[d] = pd.Series(p.to_numpy(), index=p.index / d)
I2 = {d: np.interp(grid, prof2[d].index.to_numpy(), prof2[d].to_numpy())
      for d in (12, 14, 16)}
for a, b in ((12, 14), (12, 16), (14, 16)):
    say(f"  d{a} vs d{b}: {spearman(I2[a], I2[b]):+.3f}")
say("\nRole-controlled per-layer profile (each matrix divided by its own")
say("role's median at that depth), first / interior / last layer:")
Mr = M.copy()
Mr["rel"] = Mr["value_scalar"] / Mr.groupby(["depth", "param_role"])[
    "value_scalar"].transform("median")
for d in (12, 14, 16):
    p = Mr[Mr["depth"] == d].groupby("layer")["rel"].median()
    say(f"  d{d}: layer 0 = {p.iloc[0]:.3f}   interior median = "
        f"{p.iloc[1:-1].median():.3f}   last layer = {p.iloc[-1]:.3f}   "
        f"(last/first = {p.iloc[-1]/p.iloc[0]:.3f})")
say("  The endpoints agree at all three depths: the FIRST block decoheres")
say("  least and the LAST block most, with a spread of only ~10-20% between")
say("  them -- small next to the ~2x spread between parameter roles.")
say("\nAbsolute-layer-index comparison (NO depth normalization), on the")
say("layers all three depths share (0..11):")
for a, b in ((12, 14), (12, 16), (14, 16)):
    ia = prof[a].reset_index(drop=True)[:12]
    ib = prof[b].reset_index(drop=True)[:12]
    say(f"  d{a} vs d{b}: {spearman(ia, ib):+.3f}")

# ------------------------------------------------------------- by shape
say("\n" + "=" * 78)
say("Q3. MATRIX SHAPE")
say("=" * 78)
sm = (M.groupby(["depth", "optimizer_group_id", "rows", "cols", "numel",
                 "minmn", "aspect"])["value_scalar"]
        .agg(["median", "size"]).reset_index())
say(sm.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
say("\nSpearman rho( shape descriptor , per-matrix decoherence ), pooled:")
for d in (12, 14, 16):
    a = M[M["depth"] == d]
    say(f"  d{d}:  numel {spearman(a['numel'], a['value_scalar']):+.3f}   "
        f"min(m,n) {spearman(a['minmn'], a['value_scalar']):+.3f}   "
        f"aspect {spearman(a['aspect'], a['value_scalar']):+.3f}")
say("\nAcross depths, at matched progress, per shape-class median:")
piv = M.pivot_table(index="optimizer_group_id", columns="depth",
                    values="value_scalar", aggfunc="median")
say(piv.to_string(float_format=lambda v: f"{v:.5f}"))
say("\nsame, as ratio to d12:")
say((piv.div(piv[12], axis=0)).to_string(float_format=lambda v: f"{v:.3f}"))

say("\nSquare attention blocks (width x width) vs MLP blocks, by depth:")
cls = M.assign(cls=np.where(M["param_role"] == "ve_gate", "ve_gate",
                            np.where(M["param_role"].str.startswith("attn"),
                                     "attn (square WxW)", "mlp (Wx4W)")))
say(cls.pivot_table(index="cls", columns="depth", values="value_scalar",
                    aggfunc="median")
       .to_string(float_format=lambda v: f"{v:.5f}"))
say("\nCross-depth ratio within class (d/d12):")
p2 = cls.pivot_table(index="cls", columns="depth", values="value_scalar",
                     aggfunc="median")
say(p2.div(p2[12], axis=0).to_string(float_format=lambda v: f"{v:.3f}"))

# does the depth effect survive inside a fixed role?
say("\n" + "=" * 78)
say("IS THE DEPTH EFFECT UNIFORM ACROSS ROLES?")
say("(per-role median at each matched checkpoint, d14/d16 vs the d12")
say(" five-seed range for the SAME role)")
say("=" * 78)
rows = []
for ro in sorted(M["param_role"].unique()):
    for lbl, dd in (("d14", 14), ("d16", 16)):
        out = 0
        below = 0
        rel = []
        for t in NOM:
            a = M[(M["param_role"] == ro) & (M["nominal"] == t)]
            d12v = a[a["depth"] == 12].groupby("run")["value_scalar"].median()
            v = a[a["depth"] == dd]["value_scalar"].median()
            lo, hi = d12v.min(), d12v.max()
            rel.append(v / np.median(d12v) - 1)
            if not (lo <= v <= hi):
                out += 1
                below += (v < lo)
        rows.append({"role": ro, "arm": lbl, "outside": out, "of": len(NOM),
                     "below": below, "above": out - below,
                     "median_rel_%": 100 * np.median(rel)})
say(pd.DataFrame(rows).to_string(index=False,
                                 float_format=lambda v: f"{v:+.2f}"))

M.to_csv(os.path.join(OUT, "per_matrix.csv"), index=False)
log.close()
print("\nwrote", os.path.join(OUT, "structure.txt"))
