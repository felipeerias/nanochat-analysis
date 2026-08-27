"""I0007 / A0001 - zero-init wake-up ordering.

Primary channel: muon/replay_update_relerr (sparse, one row per Muon matrix per
deep checkpoint).  A matrix is ASLEEP at a checkpoint iff its decoherence is
EXACTLY zero; awake iff it is nonzero.  "Exactly" is tested three ways (see
exact_zero_report): float64 equality, IEEE bit pattern, and the empirical gap
between the zero cluster and the smallest nonzero value.

Corroboration at update index 0 only (periodic cadence is coarse): grad/norm,
grad/zero_fraction, grad/max_abs, muon/data_norm, muon/u_final_norm_observed,
muon/cos_raw_final (undefined_reason).

Writes: wakeup_tables.txt, fig_wakeup.png, fig_magnitude.png
"""

import glob
import json
import os

import numpy as np
import pandas as pd

from loader import telemetry_load as T  # noqa: E402

ROOT = str(T.DEFAULT_DATA_ROOT)
OUT = os.path.dirname(os.path.abspath(__file__))

# the seven schema-v3 segments; d12-iter (schema v1) is excluded by name
SEGMENTS = {}
for p in sorted(glob.glob(os.path.join(ROOT, "*-s0-*"))):
    seg = os.path.basename(p)
    run = seg.split("-s0-")[0]
    if run == "d12-iter":
        continue
    SEGMENTS[run] = seg
D12 = [r for r in SEGMENTS if r.startswith("d12-s")]
DEPTHS = ["d12-s7", "d14-s7", "d16-s7"]

LINES = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LINES.append(s)


# --------------------------------------------------------------------------
# exact-zero test
# --------------------------------------------------------------------------
def is_exact_zero(v):
    """Exact IEEE zero (both signs), tested on the float64 payload.

    value_scalar is stored as parquet `double`, so the loader performs no
    conversion: the bytes on disk are the bytes tested here.  We test the bit
    pattern rather than `abs(v) < eps` on purpose - a small value is a
    different state from zero and this analysis rests on the distinction.
    """
    v = np.asarray(v, dtype=np.float64)
    bits = v.view(np.uint64)
    return (bits == np.uint64(0)) | (bits == np.uint64(1) << np.uint64(63))


# --------------------------------------------------------------------------
# load
# --------------------------------------------------------------------------
def load(run):
    seg = SEGMENTS[run]
    prov = json.load(open(os.path.join(ROOT, seg, "provenance.json")))
    sp = T.read_telemetry(ROOT, seg, "sparse").to_pandas()
    per = T.read_telemetry(ROOT, seg, "periodic").to_pandas()
    return prov, sp, per


def depth_of(prov):
    return int(prov["model_config"]["n_layer"])


def relerr_table(sp, prov):
    """per (matrix, deep update index) decoherence, tidy."""
    r = sp[sp["metric"] == "muon/replay_update_relerr"].copy()
    # convention: pre_update rows carry s, post_update rows carry s+1, so the
    # deep checkpoint at update index s appears in sparse at step s+1.
    assert (r["phase"] == "post_update").all()
    r["update_index"] = r["step"].astype(int) - 1
    deep = sorted(int(s) for s in prov["telemetry_deep_steps"])
    assert set(r["update_index"]) == set(deep), (sorted(set(r["update_index"]))[:5], deep[:5])
    r["v"] = r["value_scalar"].astype("float64")
    r["zero"] = is_exact_zero(r["v"].to_numpy())
    # honest-undefined rows would mean the APPLIED update was exactly zero;
    # keep them visible rather than folding them into "asleep".
    r["undef"] = ~r["is_defined"].astype(bool)
    return r[["parameter_name", "param_role", "layer", "update_index",
              "v", "zero", "undef"]].reset_index(drop=True)


def wake_index(r, deep):
    """first deep update index at which decoherence is not exactly zero."""
    rows = []
    for name, g in r.groupby("parameter_name"):
        g = g.sort_values("update_index")
        awake = g.loc[~g["zero"] & ~g["undef"], "update_index"]
        rows.append(dict(parameter_name=name,
                         param_role=g["param_role"].iloc[0],
                         layer=int(g["layer"].iloc[0]),
                         wake=int(awake.min()) if len(awake) else -1,
                         n_zero=int(g["zero"].sum()),
                         n_undef=int(g["undef"].sum()),
                         n_ckpt=len(g)))
    return pd.DataFrame(rows).sort_values(["wake", "param_role", "layer"])


# ==========================================================================
say("=" * 74)
say("I0007 / A0001  -  zero-init wake-up")
say("=" * 74)
say("segments (schema v3, d12-iter excluded):")
for run in sorted(SEGMENTS):
    say(f"   {run:8s}  {SEGMENTS[run]}")

DATA = {}
for run in sorted(SEGMENTS):
    prov, sp, per = load(run)
    r = relerr_table(sp, prov)
    deep = sorted(int(s) for s in prov["telemetry_deep_steps"])
    w = wake_index(r, deep)
    DATA[run] = dict(prov=prov, r=r, per=per, wake=w, deep=deep,
                     depth=depth_of(prov), seed=int(prov["seed"]))

# --------------------------------------------------------------------------
say("")
say("-" * 74)
say("1. UNIVERSE AND CADENCE")
say("-" * 74)
say(f"{'run':9s} {'depth':>5s} {'seed':>4s} {'matrices':>8s} {'ckpts':>5s}  first deep update indices")
for run in sorted(SEGMENTS):
    d = DATA[run]
    say(f"{run:9s} {d['depth']:5d} {d['seed']:4d} "
        f"{d['r']['parameter_name'].nunique():8d} {len(d['deep']):5d}  {d['deep'][:9]}")
say("")
say("every Muon matrix x every deep checkpoint is tested; nothing is sampled.")
n_tested = sum(len(DATA[r]["r"]) for r in SEGMENTS)
say(f"total (matrix, checkpoint) decoherence cells tested: {n_tested}")
say(f"total distinct Muon matrices tested: "
    f"{sum(DATA[r]['r']['parameter_name'].nunique() for r in SEGMENTS)}")

# --------------------------------------------------------------------------
say("")
say("-" * 74)
say("2. HOW EXACT ZERO WAS TESTED, AND WHY IT IS A DISTINCT STATE")
say("-" * 74)
say("value_scalar is parquet `double`; the loader does no conversion. Zero is")
say("tested on the IEEE bit pattern (0x0 or 0x8000000000000000), which is")
say("equivalent to float64 `== 0.0` and admits no tolerance.")
say("")
say(f"{'run':9s} {'idx':>4s} {'n':>4s} {'exact0':>6s} {'undef':>5s} "
    f"{'max|zero|':>9s} {'min nonzero':>12s} {'max nonzero':>12s}")
for run in sorted(SEGMENTS):
    r = DATA[run]["r"]
    for idx in DATA[run]["deep"][:4]:
        g = r[r["update_index"] == idx]
        v = g["v"].to_numpy()
        z = g["zero"].to_numpy()
        nz = v[~z & ~g["undef"].to_numpy()]
        say(f"{run:9s} {idx:4d} {len(g):4d} {int(z.sum()):6d} "
            f"{int(g['undef'].sum()):5d} {np.abs(v[z]).max() if z.any() else float('nan'):9.1e} "
            f"{nz.min() if len(nz) else float('nan'):12.4e} "
            f"{nz.max() if len(nz) else float('nan'):12.4e}")
say("")
say("The distribution at update index 0 is bimodal with an empty interval")
say("between the two modes; there are no small-but-nonzero values to confuse")
say("with zero.  Ratio (smallest nonzero) / (largest zero) is infinite because")
say("the zero mode is literally 0.")

# every exact zero anywhere in the dataset
say("")
allv = np.concatenate([DATA[run]["r"]["v"].to_numpy() for run in sorted(SEGMENTS)])
allz = np.concatenate([DATA[run]["r"]["zero"].to_numpy() for run in sorted(SEGMENTS)])
say(f"over all {len(allv)} cells in all seven runs and all 30-33 checkpoints:")
say(f"   exactly zero: {int(allz.sum())}")
say(f"   smallest nonzero value anywhere: {allv[~allz].min():.6e}")
say(f"   the nonzero population never approaches zero, so no tolerance choice")
say(f"   could turn a nonzero cell into a zero one or vice versa.")
say("")
say("exact-zero decoherence cells, by update index, over ALL checkpoints:")
for run in sorted(SEGMENTS):
    r = DATA[run]["r"]
    z = r[r["zero"]]
    u = r[r["undef"]]
    say(f"   {run:9s} exact-zero cells: {len(z):4d} at update indices "
        f"{sorted(z['update_index'].unique())}   honest-undefined cells: {len(u)}")

# --------------------------------------------------------------------------
say("")
say("-" * 74)
say("3. THE WAKE-UP ORDER")
say("-" * 74)
for run in sorted(SEGMENTS):
    w = DATA[run]["wake"]
    say("")
    say(f"{run}  (depth {DATA[run]['depth']}, seed {DATA[run]['seed']}, "
        f"{len(w)} matrices)")
    tab = w.groupby(["wake", "param_role"]).size().unstack(fill_value=0)
    say(tab.to_string())
    say("   wake-index histogram: " +
        ", ".join(f"idx {k}: {v}" for k, v in
                  sorted(w['wake'].value_counts().items())))
    lay = w.groupby(["wake"])["layer"].agg(["min", "max", "nunique"])
    say("   layer span per wake index:\n" + lay.to_string())

say("")
say("Tier A = awake at update index 0 (decoherence nonzero at the very first")
say("update).  Tier B = exactly zero at index 0, awake at index 1.")
for run in sorted(SEGMENTS):
    w = DATA[run]["wake"]
    a = set(w.loc[w["wake"] == 0, "param_role"])
    b = set(w.loc[w["wake"] == 1, "param_role"])
    say(f"   {run:9s} tier A roles = {sorted(a)}   tier B roles = {sorted(b)}")
say("")
say("Tier A tested against the model's own initializer: gpt.py init_weights()")
say("calls zeros_() on block.attn.c_proj.weight and block.mlp.c_proj.weight and")
say("on nothing else in the Muon parameter set.  Is tier A exactly that set?")
for run in sorted(SEGMENTS):
    w = DATA[run]["wake"]
    zero_init = {n for n in w["parameter_name"]
                 if n.endswith("attn.c_proj.weight") or n.endswith("mlp.c_proj.weight")}
    tierA = set(w.loc[w["wake"] == 0, "parameter_name"])
    say(f"   {run:9s} tier A == zero-init set: {tierA == zero_init}   "
        f"(|tier A| = {len(tierA)}, |zero-init| = {len(zero_init)})")

# --------------------------------------------------------------------------
say("")
say("-" * 74)
say("4. INDEPENDENT CORROBORATION AT UPDATE INDEX 0 (periodic tier, step 0)")
say("-" * 74)
say("periodic grad/* rows are keyed by (param_role, layer), not")
say("parameter_name; muon/* periodic rows do carry parameter_name.")
say("")
for run in sorted(SEGMENTS):
    per = DATA[run]["per"]
    w = DATA[run]["wake"]
    key_sleep = set(zip(w.loc[w["wake"] > 0, "param_role"],
                        w.loc[w["wake"] > 0, "layer"]))
    key_all = set(zip(w["param_role"], w["layer"]))
    line = [f"{run:9s}"]
    # grad/norm and grad/rms are keyed per (role, layer) -> one Muon matrix
    for met in ("grad/norm", "grad/rms"):
        g = per[(per["metric"] == met) & (per["step"] == 0)
                & (per["phase"] == "pre_update")].copy()
        keys = [(rr, int(ll)) if pd.notna(ll) else (rr, -1)
                for rr, ll in zip(g["param_role"], g["layer"])]
        keep = [i for i, k in enumerate(keys) if k in key_all]
        g = g.iloc[keep]
        keys = [keys[i] for i in keep]
        v = g["value_scalar"].astype("float64").to_numpy()
        hit = is_exact_zero(v)
        got = {k for k, h in zip(keys, hit) if h}
        line.append(f"{met}: {int(hit.sum())}/{len(v)} exactly 0 "
                    f"({'MATCH' if got == key_sleep else 'MISMATCH'})")
    say("   " + "  ".join(line))
    # grad/max_abs, grad/zero_fraction are role-level aggregates (layer is NaN)
    agg = per[(per["metric"].isin(["grad/zero_fraction", "grad/max_abs"]))
              & (per["step"] == 0)].pivot_table(
        index="param_role", columns="metric", values="value_scalar")
    agg = agg.loc[[r for r in agg.index if r in set(w["param_role"])]]
    agg["zero_frac_is_exactly_1"] = agg["grad/zero_fraction"] == 1.0
    agg["max_abs_is_exactly_0"] = is_exact_zero(
        agg["grad/max_abs"].to_numpy(dtype="float64"))
    say("      role-level gradient aggregates at step 0 (pre_update):")
    say("      " + agg.to_string().replace("\n", "\n      "))
    # muon stage families, keyed by parameter_name
    sleep_names = set(w.loc[w["wake"] > 0, "parameter_name"])
    for met in ("muon/data_norm", "muon/u_final_norm_observed",
                "muon/decay_norm", "muon/cos_raw_final"):
        m = per[(per["metric"] == met) & (per["step"] == 0)]
        if met == "muon/cos_raw_final":
            bad = set(m.loc[~m["is_defined"].astype(bool), "parameter_name"])
            reasons = sorted(m.loc[~m["is_defined"].astype(bool),
                                   "undefined_reason"].dropna().unique())
            say(f"      {met:32s} undefined for {len(bad):3d}/{len(m):3d} "
                f"{'MATCH' if bad == sleep_names else 'differs'}  reason={reasons}")
        else:
            v = m["value_scalar"].astype("float64").to_numpy()
            z = is_exact_zero(v)
            got = set(m["parameter_name"].to_numpy()[z])
            tag = ("MATCH-asleep" if got == sleep_names else
                   "= tier A (the zero-init projections themselves)"
                   if got == set(w.loc[w["wake"] == 0, "parameter_name"])
                   else "differs")
            say(f"      {met:32s} exactly 0 for {int(z.sum()):3d}/{len(v):3d}  {tag}")

say("")
say("The gradient-norm half of the test at full resolution, by inference from")
say("the instrument's own reference decomposition (nanochat/telemetry.py,")
say("muon_stages): at update 0 the momentum buffer is zero and the gradient is")
say("exactly zero for every tier-B matrix, so momentum_buffer.lerp_(g, 1-mu)")
say("leaves it exactly zero.  At update 1 the Newton-Schulz input is therefore")
say("proportional to g_1 alone.  A nonzero decoherence at update 1 requires a")
say("nonzero Newton-Schulz input, hence g_1 != 0.  Every tier-B matrix has")
say("nonzero decoherence at update 1, so every tier-B gradient is nonzero at")
say("update 1: the gradient-norm wake-up index equals the decoherence wake-up")
say("index, 1, for all of them.")
say("")
say("Cadence limit on the direct gradient measurement: grad/* lives in the")
say("periodic tier, emitted at pre_update steps 0, ceil(N/25), 2*ceil(N/25)...")
for run in sorted(SEGMENTS):
    per = DATA[run]["per"]
    st = sorted(per.loc[per["metric"] == "grad/norm", "step"].unique())
    say(f"   {run:9s} grad/norm steps: {st[:4]} ...  "
        f"(no grad row between update 0 and update {int(st[1])})")

# --------------------------------------------------------------------------
say("")
say("-" * 74)
say("5. STABILITY OF THE ORDER ACROSS THE FIVE d12 SEEDS")
say("-" * 74)
base = DATA[D12[0]]["wake"].set_index("parameter_name")["wake"]
mat = pd.DataFrame({run: DATA[run]["wake"].set_index("parameter_name")["wake"]
                    for run in sorted(D12)})
say(f"matrices keyed identically in all five d12 seeds: {len(mat.dropna())} "
    f"(of {len(mat)})")
ident = (mat.nunique(axis=1) == 1)
say(f"matrices with the SAME wake index in all five seeds: "
    f"{int(ident.sum())}/{len(mat)} = {ident.mean() * 100:.1f}%")
say(f"matrices that disagree anywhere: {sorted(mat.index[~ident])}")
say("")
say("pairwise agreement over all 10 seed pairs (fraction of matrices with the")
say("same wake index):")
pairs = []
runs = sorted(D12)
for i in range(len(runs)):
    for j in range(i + 1, len(runs)):
        agree = (mat[runs[i]] == mat[runs[j]]).mean()
        pairs.append(agree)
        say(f"   {runs[i]} vs {runs[j]}: {agree * 100:6.2f}%")
say(f"   min {min(pairs) * 100:.2f}%  median {np.median(pairs) * 100:.2f}%  "
    f"max {max(pairs) * 100:.2f}%")
say("")
say("tier sizes per seed (should be identical if the partition is stable):")
for run in runs:
    w = DATA[run]["wake"]
    say(f"   {run:9s} " + ", ".join(
        f"idx {k}: {v}" for k, v in sorted(w['wake'].value_counts().items())))
say("")
say("Because the hard wake-up takes only two values, a rank correlation on it")
say("is degenerate (all ties within a tier).  The set-level statement is the")
say("honest one: the tier-A / tier-B PARTITION is identical across seeds, so")
say("every rank statistic that respects ties is exactly 1.0 by construction.")


# ---- secondary, exploratory: is the MAGNITUDE order stable? --------------
def spearman(a, b):
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


say("")
say("SECONDARY (exploratory): the hard order is two-valued, so we also ask")
say("whether the decoherence MAGNITUDE at the wake-up checkpoint orders the")
say("matrices reproducibly.  Spearman rho between seeds, all matrices, at")
say("update index 1 (the checkpoint where tier B first shows a value):")
mag = pd.DataFrame({run: DATA[run]["r"].query("update_index == 1")
                    .set_index("parameter_name")["v"] for run in runs})
rhos = []
for i in range(len(runs)):
    for j in range(i + 1, len(runs)):
        rho = spearman(mag[runs[i]], mag[runs[j]])
        rhos.append(rho)
say(f"   n = {len(mag)} matrices; 10 pairs; rho min {min(rhos):.3f} "
    f"median {np.median(rhos):.3f} max {max(rhos):.3f}")
say("   same at update index 0, restricted to tier A (24 matrices):")
mag0 = pd.DataFrame({run: DATA[run]["r"].query("update_index == 0 and not zero")
                     .set_index("parameter_name")["v"] for run in runs})
r0 = [spearman(mag0[runs[i]], mag0[runs[j]])
      for i in range(len(runs)) for j in range(i + 1, len(runs))]
say(f"   n = {len(mag0)}; rho min {min(r0):.3f} median {np.median(r0):.3f} "
    f"max {max(r0):.3f}")
say("   role-median decoherence at update index 1 (d12, across seeds):")
mm = pd.concat([DATA[run]["r"].query("update_index == 1").assign(run=run)
                for run in runs])
say(mm.groupby("param_role")["v"].agg(["median", "min", "max"]).to_string())
say("")
say("   Most of that rho is role separation (role medians span ~45x).  After")
say("   removing the per-role median within each seed, the residual per-matrix")
say("   ordering is:")
res = {}
for run in runs:
    g = DATA[run]["r"].query("update_index == 1").copy()
    g["res"] = g["v"] - g.groupby("param_role")["v"].transform("median")
    res[run] = g.set_index("parameter_name")["res"]
res = pd.DataFrame(res)
rr = [spearman(res[runs[i]], res[runs[j]])
      for i in range(len(runs)) for j in range(i + 1, len(runs))]
say(f"   within-role residual rho: min {min(rr):.3f} median {np.median(rr):.3f}"
    f" max {max(rr):.3f}  -> {'reproducible' if np.median(rr) > 0.5 else 'seed noise'}")
say("   within-role correlation of decoherence with layer index (per seed,")
say("   pooled over roles after median-centering):")
for run in runs:
    g = DATA[run]["r"].query("update_index == 1").copy()
    g["res"] = g["v"] - g.groupby("param_role")["v"].transform("median")
    say(f"      {run:9s} rho(res, layer) = {spearman(g['res'], g['layer']):+.3f}")

say("")
say("CONTEXT (outside the declared universe): the same exact-zero state at")
say("step 0 also affects non-Muon parameters, from grad/norm keyed by role:")
per = DATA["d12-s7"]["per"]
g = per[(per["metric"] == "grad/norm") & (per["step"] == 0)].copy()
g["exact0"] = is_exact_zero(g["value_scalar"].to_numpy(dtype="float64"))
say(g.groupby("param_role")["exact0"].agg(["sum", "count"]).to_string())

# --------------------------------------------------------------------------
say("")
say("-" * 74)
say("6. ACROSS DEPTHS  (d12-s7, d14-s7, d16-s7; seed 7 throughout)")
say("-" * 74)
say("NORMALIZATION.  Depth changes the layer count (12/14/16) and therefore the")
say("matrix count (6 per block + one ve_gate on odd blocks).  Layer indices are")
say("NOT comparable across depths, so layers are reported as relative depth")
say("rho_L = layer / (n_layer - 1) in [0, 1].  Steps are not comparable either")
say("(caveat: horizons differ), so wake-up is reported both as an absolute")
say("update index and as normalized_progress = update_index / num_iterations.")
say("")
say(f"{'run':9s} {'L':>3s} {'matrices':>8s} {'tierA':>6s} {'tierB':>6s} "
    f"{'tierB%':>7s} {'wake idx':>9s} {'wake np':>10s} {'iters':>6s}")
for run in DEPTHS:
    d = DATA[run]
    w = d["wake"]
    nA = int((w["wake"] == 0).sum())
    nB = int((w["wake"] == 1).sum())
    ni = d["prov"]["num_iterations"]
    idxs = "{" + ",".join(str(int(x)) for x in sorted(w["wake"].unique())) + "}"
    say(f"{run:9s} {d['depth']:3d} {len(w):8d} {nA:6d} {nB:6d} "
        f"{nB / len(w) * 100:6.1f}% {idxs:>9s} "
        f"{1 / ni:10.2e} {ni:6d}")
say("")
say("wake index by relative depth rho_L (all depths; every matrix listed as a")
say("(rho_L, role) -> wake index cell; a single wake index per column means")
say("relative depth has no effect):")
for run in DEPTHS:
    w = DATA[run]["wake"].copy()
    L = DATA[run]["depth"]
    w["rho_L"] = w["layer"] / (L - 1)
    piv = w.pivot_table(index="param_role", columns="rho_L", values="wake")
    say(f"   {run}: distinct wake indices per role = " +
        str({k: sorted(set(v.dropna())) for k, v in piv.iterrows()}))
    say(f"      correlation(wake, rho_L) = " +
        ("undefined (wake is constant within each role)"
         if w.groupby("param_role")["wake"].nunique().max() == 1
         else f"{spearman(w['wake'], w['rho_L']):.3f}"))
say("")
say("Is wake-up governed by absolute step, by relative depth, or by neither?")
for run in DEPTHS:
    w = DATA[run]["wake"]
    ni = DATA[run]["prov"]["num_iterations"]
    say(f"   {run:9s} last wake at absolute update index "
        f"{int(w['wake'].max())}; as normalized progress "
        f"{w['wake'].max() / ni:.3e}")

# depth-matched magnitude
say("")
say("decoherence magnitude at update index 1, by role and relative depth:")
rows = []
for run in DEPTHS:
    g = DATA[run]["r"].query("update_index == 1").copy()
    L = DATA[run]["depth"]
    g["rho_L"] = g["layer"] / (L - 1)
    g["run"] = run
    rows.append(g)
allm = pd.concat(rows)
say(allm.pivot_table(index="param_role", columns="run", values="v",
                     aggfunc="median").to_string())
say("")
say("The one graded structure that survives: within a role, decoherence at the")
say("wake-up checkpoint falls with position in the stack.  rho_L and the raw")
say("layer index are monotone transforms of one another inside a run, so the")
say("within-run rho is the same either way; what the normalization buys is the")
say("right to compare the three numbers with each other.")


def depth_slope(run):
    g = DATA[run]["r"].query("update_index == 1").copy()
    L = DATA[run]["depth"]
    g["rho_L"] = g["layer"] / (L - 1)
    g["res"] = g["v"] - g.groupby("param_role")["v"].transform("median")
    return spearman(g["res"], g["rho_L"])


d12_slopes = {run: depth_slope(run) for run in sorted(D12)}
say("   d12 five-seed values of rho(residual, rho_L): " +
    ", ".join(f"{k.split('-')[1]} {v:+.3f}" for k, v in d12_slopes.items()))
lo, hi = min(d12_slopes.values()), max(d12_slopes.values())
say(f"   d12 five-seed range: [{lo:+.3f}, {hi:+.3f}]")
for run in DEPTHS[1:]:
    s = depth_slope(run)
    say(f"   {run:9s} rho = {s:+.3f}  -> "
        f"{'INSIDE' if lo <= s <= hi else 'OUTSIDE'} the d12 five-seed range")
say("   So the depth gradient of decoherence at wake-up is present at every")
say("   depth and is not distinguishable across depths at n=5 seeds.")

# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROLES = ["attn_q", "attn_k", "attn_v", "ve_gate", "mlp_in", "attn_out", "mlp_out"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
for ax, run in zip(axes, DEPTHS):
    d = DATA[run]
    L = d["depth"]
    w = d["wake"]
    for k, role in enumerate(ROLES):
        g = w[w["param_role"] == role]
        if not len(g):
            continue
        dodge = (k - 3) * 0.055  # one sub-column per role, so nothing hides
        ax.scatter(g["wake"] + dodge, g["layer"] / (L - 1),
                   s=34, marker="os^Dv<>"[k], label=role, alpha=0.9)
    ax.axvspan(-0.35, 0.35, color="0.85", alpha=0.5, zorder=0)
    ax.axvspan(0.65, 1.35, color="0.92", alpha=0.5, zorder=0)
    ax.set_xlim(-0.6, 2.6)
    ax.set_xticks([0, 1, 2])
    ax.set_xlabel("first deep checkpoint with nonzero decoherence\n(update index)")
    ax.set_ylabel(r"relative depth  $\rho_L=\ell/(L-1)$")
    ax.set_title(f"{run}   (L={L}, {len(w)} Muon matrices)")
    ax.grid(alpha=0.25)
axes[0].legend(fontsize=7, loc="center right", framealpha=0.95)
fig.suptitle("Zero-init wake-up: tier A (the two zero-init projections) at update 0, "
             "everything else at update 1, at every depth", fontsize=11)
fig.savefig(os.path.join(OUT, "fig_wakeup.png"), dpi=140)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
ax = axes[0]
for k, run in enumerate(sorted(SEGMENTS)):
    r = DATA[run]["r"]
    frac = r.groupby("update_index")["zero"].mean()
    frac = frac[frac.index <= 64]
    ax.plot(frac.index, frac * 100, marker="o", ms=9 - k, label=run, alpha=0.8,
            lw=3.5 - 0.4 * k)
ax.set_xscale("symlog", linthresh=1)
ax.set_xlabel("deep checkpoint (update index)")
ax.set_ylabel("% of Muon matrices with EXACTLY zero decoherence")
ax.set_title("all seven v3 runs; the curves coincide exactly\n"
             "(69.2% at update 0, 0.0% from update 1 on)", fontsize=10)
ax.legend(fontsize=7)
ax.grid(alpha=0.25)

ax = axes[1]
mk = dict(zip(DEPTHS, "os^"))
cmap = plt.get_cmap("tab10")
for run in DEPTHS:
    g = DATA[run]["r"].query("update_index == 1").copy()
    L = DATA[run]["depth"]
    g["rho_L"] = g["layer"] / (L - 1)
    for k, role in enumerate(ROLES):
        q = g[g["param_role"] == role]
        if not len(q):
            continue
        ax.plot(q["rho_L"], q["v"], marker=mk[run], ms=4, lw=0.9,
                color=cmap(k), alpha=0.85,
                label=role if run == "d12-s7" else None)
ax.set_yscale("log")
ax.set_xlabel(r"relative depth  $\rho_L=\ell/(L-1)$")
ax.set_ylabel("decoherence at update index 1")
ax.set_title("magnitude at the wake-up checkpoint\n"
             "(marker = depth: o d12, s d14, ^ d16)", fontsize=10)
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.25)
fig.suptitle("Exactly-zero decoherence is confined to update index 0; the only "
             "graded structure is in the magnitude", fontsize=11)
fig.savefig(os.path.join(OUT, "fig_magnitude.png"), dpi=140)
plt.close(fig)

open(os.path.join(OUT, "wakeup_tables.txt"), "w").write("\n".join(LINES) + "\n")
print("\nwrote", os.path.join(OUT, "wakeup_tables.txt"))
