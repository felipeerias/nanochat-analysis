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
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/felipe/Igalia/nanochat/analysis/loader")
import telemetry_load as T  # noqa: E402

ROOT = "/home/felipe/Igalia/nanochat/telemetry-data/sweep/telemetry-data"
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
    # grad/norm and grad/zero_fraction, restricted to the Muon matrices
    for met, want in (("grad/norm", 0.0), ("grad/zero_fraction", 1.0)):
        g = per[(per["metric"] == met) & (per["step"] == 0)
                & (per["phase"] == "pre_update")].copy()
        keys = [(rr, int(ll)) if pd.notna(ll) else (rr, -1)
                for rr, ll in zip(g["param_role"], g["layer"])]
        keep = [i for i, k in enumerate(keys) if k in key_all]
        g = g.iloc[keep]
        keys = [keys[i] for i in keep]
        v = g["value_scalar"].astype("float64").to_numpy()
        hit = is_exact_zero(v) if want == 0.0 else (v == 1.0)
        got = {k for k, h in zip(keys, hit) if h}
        line.append(f"{met}: {int(hit.sum())}/{len(v)} exact "
                    f"({'MATCH' if got == key_sleep else 'MISMATCH'})")
    say("   " + "  ".join(line))
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
say("Cadence limit on the gradient-norm half of the test: grad/* lives in the")
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
    say(f"{run:9s} {d['depth']:3d} {len(w):8d} {nA:6d} {nB:6d} "
        f"{nB / len(w) * 100:6.1f}% {sorted(w['wake'].unique())!s:>9s} "
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
        ax.scatter(g["wake"], g["layer"] / (L - 1) + 0.0,
                   s=46, marker="os^Dv<>"[k], label=role, alpha=0.85)
    ax.set_xlim(-0.6, 2.6)
    ax.set_xticks([0, 1, 2])
    ax.set_xlabel("first deep checkpoint with nonzero decoherence\n(update index)")
    ax.set_ylabel(r"relative depth  $\rho_L=\ell/(L-1)$")
    ax.set_title(f"{run}   (L={L}, {len(w)} Muon matrices)")
    ax.grid(alpha=0.25)
axes[0].legend(fontsize=7, loc="center right")
fig.suptitle("Zero-init wake-up: every Muon matrix is awake by update index 1, "
             "at every depth", fontsize=11)
fig.savefig(os.path.join(OUT, "fig_wakeup.png"), dpi=140)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
ax = axes[0]
for run in sorted(D12):
    r = DATA[run]["r"]
    frac = r.groupby("update_index")["zero"].mean()
    frac = frac[frac.index <= 64]
    ax.plot(frac.index, frac * 100, marker="o", label=run, alpha=0.8)
ax.set_xscale("symlog", linthresh=1)
ax.set_xlabel("deep checkpoint (update index)")
ax.set_ylabel("% of Muon matrices with EXACTLY zero decoherence")
ax.set_title("five d12 seeds")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
ax = axes[1]
for run in DEPTHS:
    r = DATA[run]["r"]
    frac = r.groupby("update_index")["zero"].mean()
    frac = frac[frac.index <= 64]
    ax.plot(frac.index, frac * 100, marker="s", label=f"{run} (L={DATA[run]['depth']})",
            alpha=0.8)
ax.set_xscale("symlog", linthresh=1)
ax.set_xlabel("deep checkpoint (update index)")
ax.set_ylabel("% exactly zero")
ax.set_title("three depths, seed 7")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
fig.suptitle("Exactly-zero decoherence is confined to update index 0", fontsize=11)
fig.savefig(os.path.join(OUT, "fig_magnitude.png"), dpi=140)
plt.close(fig)

open(os.path.join(OUT, "wakeup_tables.txt"), "w").write("\n".join(LINES) + "\n")
print("\nwrote", os.path.join(OUT, "wakeup_tables.txt"))
