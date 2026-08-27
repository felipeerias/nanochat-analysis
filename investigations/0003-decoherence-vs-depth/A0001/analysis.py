"""I0003/A0001 — does Muon replay decoherence change with depth?

Protocol: investigations/0003-decoherence-vs-depth/README.md @ e76859c
Channel:  muon/replay_update_relerr, sparse tier, post_update, defined rows,
          one row per matrix (parameter_name) per deep checkpoint.

Writes: matched.csv, per_matrix.csv, decision.txt, and figures.
"""
import json
import os

import numpy as np
import pandas as pd

from loader import telemetry_load as tl  # noqa: E402

ROOT = str(tl.DEFAULT_DATA_ROOT)
OUT = os.path.dirname(os.path.abspath(__file__))
METRIC = "muon/replay_update_relerr"

SEGS = [
    ("d12-s7", 12, 7, "d12-s7-s0-6c217854f4174ce884dfb2b2dcc13c45"),
    ("d12-s8", 12, 8, "d12-s8-s0-2b2e72e4395440029b92226213d137bb"),
    ("d12-s9", 12, 9, "d12-s9-s0-b872e698e81b4a8fa0cd08512f43c2c2"),
    ("d12-s10", 12, 10, "d12-s10-s0-0ffeb05742bb4154a3f1afc202dc3955"),
    ("d12-s11", 12, 11, "d12-s11-s0-5fd1bf807f764954b30658872fc3e2ad"),
    ("d14-s7", 14, 7, "d14-s7-s0-b72acf7ac59942dab902e26fa0b2c80d"),
    ("d16-s7", 16, 7, "d16-s7-s0-8e0a39fc8f164343bd01ef6b915e849f"),
]
D12 = [s[0] for s in SEGS if s[1] == 12]

# ---------------------------------------------------------------- load
frames = []
prov = {}
for run, depth, seed, seg in SEGS:
    sp = tl.read_telemetry(ROOT, seg, "sparse").to_pandas()
    assert set(sp["schema_version"].unique()) == {"3"}, run
    r = sp[sp["metric"] == METRIC].copy()
    n_all = len(r)
    r = tl.defined(r)                      # explicit; none are undefined here
    assert len(r) == n_all, (run, n_all, len(r))
    assert (r["phase"] == "post_update").all()
    r["run"] = run
    r["depth"] = depth
    r["seed"] = seed
    r["layer"] = r["layer"].astype(int)
    frames.append(r[["run", "depth", "seed", "step", "normalized_progress",
                     "value_scalar", "param_role", "parameter_name", "layer",
                     "optimizer_group_id"]])
    prov[run] = json.load(open(os.path.join(ROOT, seg, "provenance.json")))
df = pd.concat(frames, ignore_index=True)

log = open(os.path.join(OUT, "decision.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.write(s + "\n")


say("=" * 78)
say("I0003 / A0001 — muon/replay_update_relerr vs depth")
say("=" * 78)
say(f"\nrows loaded (defined, post_update): {len(df)}")
say(df.groupby(["run", "depth"]).agg(
    rows=("value_scalar", "size"),
    matrices=("parameter_name", "nunique"),
    ckpts=("step", "nunique")).to_string())

# ------------------------------------------------- structural zeros (step 0)
say("\n" + "-" * 78)
say("INIT-TIME STRUCTURAL ZEROS (reported separately, never averaged in)")
say("-" * 78)
zrows = df[df["value_scalar"] == 0.0]
say(f"exact zeros anywhere in the channel: {len(zrows)} of {len(df)} rows")
say("all at step:", sorted(zrows["step"].unique()),
    " (= deep checkpoint at update index 0)")
zt = zrows.groupby(["run", "param_role"]).size().unstack(fill_value=0)
nz = df[(df["step"] == 1) & (df["value_scalar"] != 0.0)]
say("\nzero-valued matrices at the update-0 checkpoint, by role:")
say(zt.to_string())
say("\nnon-zero matrices at the update-0 checkpoint, by role:")
say(nz.groupby(["run", "param_role"]).size().unstack(fill_value=0).to_string())
say("\nfraction of matrices that are structurally zero at update 0:")
for run, depth, _, _ in SEGS:
    a = df[(df["run"] == run) & (df["step"] == 1)]
    say(f"  {run:8s} {len(a[a['value_scalar']==0.0]):3d}/{len(a):3d} "
        f"= {100*len(a[a['value_scalar']==0.0])/len(a):.1f}%")
say("\nDecoherence at update 0 over the ACTIVE (non-zero) matrices only:")
act = df[(df["step"] == 1) & (df["value_scalar"] != 0.0)]
say(act.groupby(["run", "depth"])["value_scalar"].agg(
    ["size", "median", "mean", "min", "max"]).to_string())
say("\nIf the zeros were averaged in blindly the update-0 mean would be:")
say(df[df["step"] == 1].groupby(["run", "depth"])["value_scalar"]
    .agg(["mean", "median"]).to_string())
say("\n=> the update-0 checkpoint is EXCLUDED from every number below.")
say("   It is also not on the matched grid (see next section), so it drops")
say("   out of the decision rule on its own.")

# ------------------------------------------------- matched checkpoint grid
say("\n" + "-" * 78)
say("MATCHED CHECKPOINTS on normalized_progress")
say("-" * 78)
# The deep schedule has the same SHAPE in normalized progress at every depth,
# but the geometric prefix {0,1,2,4,8,...} is defined in STEP space, so those
# points land at different progress at different depths and cannot be matched.
# The ~20-point uniform tail is defined in progress space and does match, up
# to the rounding to an integer step (|delta| < 1/num_iterations).
NOMINAL = np.round(np.arange(0.05, 1.0001, 0.05), 10)
TOL = 0.005      # < half the 0.05 tail spacing; observed deviations < 1e-3
matched = []
for t in NOMINAL:
    row = {"nominal": t}
    ok = True
    for run, depth, _, _ in SEGS:
        g = np.array(sorted(df[df["run"] == run]["normalized_progress"].unique()))
        d = np.abs(g - t)
        if d.min() > TOL:
            ok = False
            break
        # if the landmark and the tail point collide, take the closer one
        row[run] = float(g[d.argmin()])
        row[run + "_dev"] = float(g[d.argmin()] - t)
    if ok:
        matched.append(row)
matched = pd.DataFrame(matched)
say(f"matched checkpoints: {len(matched)} of {len(NOMINAL)} nominal tail points")
devcols = [c for c in matched.columns if c.endswith("_dev")]
say(f"max |progress - nominal| over all runs and points: "
    f"{matched[devcols].abs().to_numpy().max():.6f}")
say("\n(unmatched: the geometric prefix and the recipe landmarks — they are")
say(" step-defined, so d12 step 1 is progress 3.97e-4 but d16 step 1 is")
say(" 1.86e-4. No early-training checkpoint is shared across all depths.)")
say("\nnominal grid: " + ", ".join(f"{t:.2f}" for t in matched["nominal"]))

# --------------------------------------------- per-run per-checkpoint summary
# Per-run summary at a checkpoint = MEDIAN over that run's per-matrix channels.
# (The protocol speaks of "the d14 and d16 medians".)  Mean is carried as a
# robustness check.  Populations differ in size (78 / 91 / 104 matrices) but
# are identical in role composition per layer, so the median is comparable.
recs = []
for _, m in matched.iterrows():
    for run, depth, seed, _ in SEGS:
        sub = df[(df["run"] == run) &
                 (np.isclose(df["normalized_progress"], m[run]))]
        assert len(sub) in (78, 91, 104), (run, m["nominal"], len(sub))
        assert (sub["value_scalar"] > 0).all()
        recs.append({
            "nominal": m["nominal"], "run": run, "depth": depth, "seed": seed,
            "progress": m[run], "n_matrix": len(sub),
            "median": sub["value_scalar"].median(),
            "mean": sub["value_scalar"].mean(),
            "gmean": float(np.exp(np.log(sub["value_scalar"]).mean())),
            "q25": sub["value_scalar"].quantile(.25),
            "q75": sub["value_scalar"].quantile(.75),
        })
S = pd.DataFrame(recs)
S.to_csv(os.path.join(OUT, "matched.csv"), index=False)

# --------------------------------------------------------- the decision rule
say("\n" + "=" * 78)
say("DECISION RULE (fixed before looking; README.md @ e76859c)")
say("=" * 78)


def decide(stat):
    rows = []
    for t in matched["nominal"]:
        a = S[(S["nominal"] == t)]
        d12 = a[a["depth"] == 12][stat].to_numpy()
        lo, hi = d12.min(), d12.max()
        v14 = float(a[a["depth"] == 14][stat].iloc[0])
        v16 = float(a[a["depth"] == 16][stat].iloc[0])
        rows.append({
            "progress": t, "d12_lo": lo, "d12_hi": hi,
            "d12_med": float(np.median(d12)),
            "d12_range_rel": (hi - lo) / np.median(d12),
            "d12_sd_rel": d12.std(ddof=1) / np.median(d12),
            "d14": v14, "d16": v16,
            "d14_out": not (lo <= v14 <= hi),
            "d16_out": not (lo <= v16 <= hi),
            "d14_dir": "above" if v14 > hi else ("below" if v14 < lo else "in"),
            "d16_dir": "above" if v16 > hi else ("below" if v16 < lo else "in"),
            "d14_rel": v14 / np.median(d12) - 1.0,
            "d16_rel": v16 / np.median(d12) - 1.0,
        })
    return pd.DataFrame(rows)


for stat in ("median", "mean", "gmean"):
    D = decide(stat)
    D.to_csv(os.path.join(OUT, f"decision_{stat}.csv"), index=False)
    n = len(D)
    say("\n" + "#" * 70)
    say(f"# per-run summary statistic across matrices: {stat.upper()}"
        + ("   <-- PRIMARY" if stat == "median" else "   (robustness)"))
    say("#" * 70)
    say(D.to_string(float_format=lambda v: f"{v:.5f}", index=False))
    for lbl in ("d14", "d16"):
        out = int(D[lbl + "_out"].sum())
        above = int((D[lbl + "_dir"] == "above").sum())
        below = int((D[lbl + "_dir"] == "below").sum())
        say(f"\n{lbl}: outside the d12 five-seed range at {out}/{n} "
            f"checkpoints ({100*out/n:.0f}%); above {above}, below {below}, "
            f"inside {n-out}")
        say(f"     median offset vs the d12 median: "
            f"{100*D[lbl+'_rel'].median():+.2f}%  "
            f"(range {100*D[lbl+'_rel'].min():+.2f}% to "
            f"{100*D[lbl+'_rel'].max():+.2f}%)")
    both_out = int((D["d14_out"] & D["d16_out"]).sum())
    say(f"\nboth d14 and d16 outside at the same checkpoint: {both_out}/{n}")
    say(f"d12 five-seed RANGE relative to the d12 median: "
        f"median {100*D['d12_range_rel'].median():.2f}%, "
        f"min {100*D['d12_range_rel'].min():.2f}%, "
        f"max {100*D['d12_range_rel'].max():.2f}%")
    say(f"d12 five-seed SD relative to the d12 median:    "
        f"median {100*D['d12_sd_rel'].median():.2f}%, "
        f"min {100*D['d12_sd_rel'].min():.2f}%, "
        f"max {100*D['d12_sd_rel'].max():.2f}%")
    # verdict
    n14, n16 = int(D["d14_out"].sum()), int(D["d16_out"].sum())
    half = n / 2.0
    cons14 = (D[D["d14_out"]]["d14_dir"].nunique() <= 1)
    cons16 = (D[D["d16_out"]]["d16_dir"].nunique() <= 1)
    sup = (n14 > half and n16 > half and cons14 and cons16
           and (D[D["d14_out"]]["d14_dir"].iloc[0] ==
                D[D["d16_out"]]["d16_dir"].iloc[0] if n14 and n16 else False))
    ref = (n - n14 > half) and (n - n16 > half)
    say(f"\n  -> d14 outside at {n14}/{n} (need >{half:.0f} for supported, "
        f"inside {n-n14} need >{half:.0f} for refuted)")
    say(f"  -> d16 outside at {n16}/{n} (inside {n-n16})")
    say(f"  -> verdict under the STRICT parse of 'consistent direction'"
        f" (zero excursions the other way), reading [B] in verdict.txt: "
        + ("SUPPORTED" if sup else ("REFUTED" if ref else "INCONCLUSIVE")))
    b14 = int((D["d14_dir"] == "below").sum())
    b16 = int((D["d16_dir"] == "below").sum())
    say(f"  -> verdict under the COUNT parse (excursions sharing one"
        f" direction > half), reading [A] in verdict.txt, d14 {b14}/{n} and"
        f" d16 {b16}/{n} below: "
        + ("SUPPORTED" if (b14 > half and b16 > half)
           else ("REFUTED" if ref else "INCONCLUSIVE")))
    say("     (verdict.txt applies the rule in full and reports [A].)")

say("\n" + "=" * 78)
say("SEED REFERENCE (I0001/conclusion.md)")
say("=" * 78)
say("I0001 reports muon/replay_update_relerr at 3.5% sd-relative and ~8%")
say("range-relative across the five d12 seeds. The per-checkpoint figures")
say("printed above are the same quantity recomputed on the matched tail.")

log.close()
print("\nwrote", os.path.join(OUT, "decision.txt"))
