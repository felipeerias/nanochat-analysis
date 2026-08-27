"""I0006 / A0001 - stage 7: robustness of the two disclosed choices.

(a) the median-across-rows aggregation (vs mean) for multi-row families;
(b) the z>=3 threshold (vs 2);
(c) the d14 run as an out-of-design consistency check on the headline channels;
(d) a hand recomputation of the loss/train_mean headline straight from the
    parquet, bypassing every helper in this folder.
"""

import json
import os

import numpy as np
import pandas as pd

from loader import telemetry_load as T  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = str(T.DEFAULT_DATA_ROOT)
S = pd.read_parquet(os.path.join(HERE, "series.parquet"))
F = pd.read_csv(os.path.join(HERE, "families.csv"))
META = json.load(open(os.path.join(HERE, "runs.json")))
D12 = sorted(r for r, m in META.items() if m["depth"] == 12)

lines = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


p("## (d) independent recomputation of the loss/train_mean headline")
raw = {}
for r in D12 + ["d14-s7", "d16-s7"]:
    seg = META[r]["segment"]
    df = T.read_telemetry(ROOT, seg, "continuous").to_pandas()
    df = df[(df.metric == "loss/train_mean") & (df.is_defined)]
    raw[r] = df.set_index("step")["value_scalar"].sort_index()
N16 = META["d16-s7"]["num_iterations"]
N12 = META["d12-s7"]["num_iterations"]
for lo, hi, lab in ((0, 400, "step <= 400"), (401, 2520, "401 < step <= 2520")):
    st = np.arange(lo, hi + 1)
    y16 = raw["d16-s7"].reindex(st).to_numpy()
    M = np.vstack([raw[r].reindex(st).to_numpy() for r in D12])
    rel_abs = (y16 - np.nanmedian(M, 0)) / np.nanmedian(M, 0)
    # progress alignment: d12 evaluated at step * N12/N16
    st12 = st * (N12 / N16)
    Mp = np.vstack([np.interp(st12, raw[r].index.to_numpy(), raw[r].to_numpy())
                    for r in D12])
    rel_prog = (y16 - np.nanmedian(Mp, 0)) / np.nanmedian(Mp, 0)
    p(f"  {lab:20s} median rel diff  abs {100*np.nanmedian(rel_abs):+7.3f}%   "
      f"prog {100*np.nanmedian(rel_prog):+7.3f}%")
p("  (compare per_family_region.csv: warmup -2.77 / -18.37, post -1.08 / -7.97)")
sub = F[F.metric == "loss/train_mean"].iloc[0]
p(f"  pipeline values:      warmup abs {100*sub.w_medrel_abs:+.3f}% "
  f"prog {100*sub.w_medrel_prog:+.3f}% | post abs {100*sub.p_medrel_abs:+.3f}% "
  f"prog {100*sub.p_medrel_prog:+.3f}%")

p("")
p("## (a) median vs mean across parameter rows")
seg = META["d12-s7"]["segment"]
sp = T.defined(T.read_telemetry(ROOT, seg, "sparse").to_pandas())
sp = sp[sp["value_scalar"].notna()]
multi = (sp.groupby(["metric", "step"]).size().groupby("metric").max())
multi = multi[multi > 1]
p(f"  {len(multi)} of {sp.metric.nunique()} sparse families have >1 row per step "
  f"(max rows/step {int(multi.max())})")
g = sp[sp.metric.isin(multi.index)].groupby(["metric", "step"])["value_scalar"]
cmp = pd.DataFrame({"med": g.median(), "mean": g.mean()}).reset_index()
with np.errstate(divide="ignore", invalid="ignore"):
    rr = np.abs(cmp["mean"] - cmp["med"]) / np.abs(cmp["med"])
p(f"  |mean - median| / |median| across those rows: median "
  f"{np.nanmedian(rr):.4f}, p90 {np.nanquantile(rr, .9):.4f}, "
  f"p99 {np.nanquantile(rr, .99):.4f}, max {np.nanmax(rr):.4f}")
worst = cmp.assign(r=rr).groupby("metric")["r"].median().nlargest(5)
p("  families most sensitive to the choice (median |mean-med|/|med|):")
for k, v in worst.items():
    p(f"    {k}: {v:.3f}")
p("  -> for most families the choice is immaterial; it is disclosed because for")
p("     per-parameter families the aggregated populations are NOT comparable")
p("     across depths (d16 has 16 blocks, d12 has 12).")

p("")
p("## (b) threshold sensitivity, dynamics families testable under both alignments")
t = F[(F.group == "dynamics") & (F.verdict != "underpowered") &
      (F.verdict_prog != "underpowered")]
for z in (2.0, 3.0, 4.0):
    wa = int((t.w_medabsz_abs >= z).sum())
    wp = int((t.w_medabsz_prog >= z).sum())
    au = int((t.w_align_dz >= z).sum())
    p(f"  z>={z}: clears the band inside the window on the STEP axis {wa:3d}, "
      f"on the PROGRESS axis {wp:3d}; alignment gap itself >= z for {au:3d} "
      f"of {len(t)} families")

p("")
p("## (c) d14 consistency on the headline channels (out of the declared design)")
for m in ("loss/train_mean", "update/loss_before", "muon/replay_update_relerr"):
    row = []
    for dep, run in ((14, "d14-s7"), (16, "d16-s7")):
        N = META[run]["num_iterations"]
        d = S[(S.run == run) & (S.metric == m)].sort_values("step")
        st, pr, v = (d["step"].to_numpy(float), d["progress"].to_numpy(float),
                     d["value"].to_numpy(float))
        ref = [S[(S.run == r) & (S.metric == m)].sort_values("step") for r in D12]
        for tag, xq, xr in (("abs", st, "step"), ("prog", pr, "progress")):
            M = np.vstack([np.interp(xq, x[xr].to_numpy(float),
                                     x["value"].to_numpy(float),
                                     left=np.nan, right=np.nan) for x in ref])
            med = np.nanmedian(M, 0)
            rel = (v - med) / np.abs(med)
            w = (st <= 400) & np.isfinite(rel)
            po = (st > 400) & np.isfinite(rel)
            row.append(f"d{dep} {tag}: w {100*np.median(rel[w]):+6.2f}% "
                       f"p {100*np.median(rel[po]):+6.2f}%")
    p(f"  {m}")
    for x in row:
        p("      " + x)

open(os.path.join(HERE, "robustness.txt"), "w").write("\n".join(lines) + "\n")
