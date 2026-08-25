"""I0006 / A0001 - stage 2: what is actually matchable across depths, and where.

Pure schedule geometry, no metric values. Answers:
  (a) which absolute steps carry a measurement in both d12 and d16, per tier;
  (b) which normalized_progress values do, at what tolerance;
  (c) where the two recipe landmarks (LR warmup end 40, Muon ramp end 400) sit
      on each axis, and how wide the resulting phase-mismatch zone is.
"""

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
S = pd.read_parquet(os.path.join(HERE, "series.parquet"))
META = json.load(open(os.path.join(HERE, "runs.json")))

LR_WARMUP_END = 40
MUON_RAMP_END = 400

lines = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


N = {r: m["num_iterations"] for r, m in META.items()}
p("# run horizons")
for r in sorted(META):
    m = META[r]
    p(f"  {r:8s} depth={m['depth']:2d} seed={m['seed']:2d} N={m['num_iterations']:5d} "
      f"lr_warmup_frac={LR_WARMUP_END/m['num_iterations']:.4%} "
      f"muon_ramp_frac={MUON_RAMP_END/m['num_iterations']:.4%} "
      f"periodic_every={m['telemetry_every']}")

p("")
p("# phase-mismatch zones on the normalized-progress axis (d12 vs d16)")
for name, end in (("lr_warmup_end=40", LR_WARMUP_END), ("muon_ramp_end=400", MUON_RAMP_END)):
    lo, hi = end / 5376, end / 2520
    p(f"  {name}: d16 reaches it at p={lo:.5f}, d12 at p={hi:.5f}; "
      f"progress-aligned comparisons in p in [{lo:.5f},{hi:.5f}] "
      f"({(hi-lo)*100:.2f}% of training) compare unequal schedule phases")
p(f"  conversely, an absolute-step comparison at step s puts d12 at progress "
  f"{5376/2520:.3f}x that of d16 (2520 vs 5376 steps)")

p("")
p("# per-tier x-grid matchability, d12-s7 vs d16-s7")
d12 = S[S.run == "d12-s7"]
d16 = S[S.run == "d16-s7"]
for tier in ("continuous", "periodic", "sparse"):
    a = np.sort(d12[d12.tier == tier]["step"].unique())
    b = np.sort(d16[d16.tier == tier]["step"].unique())
    common = np.intersect1d(a, b)
    pa = np.sort(d12[d12.tier == tier]["progress"].unique())
    pb = np.sort(d16[d16.tier == tier]["progress"].unique())
    # progress match within 0.5 * (median d16 progress spacing)
    tol = 0.5 * np.median(np.diff(pb)) if len(pb) > 2 else 1e-9
    idx = np.searchsorted(pb, pa)
    idx = np.clip(idx, 1, len(pb) - 1)
    near = np.minimum(np.abs(pa - pb[idx - 1]), np.abs(pa - pb[np.minimum(idx, len(pb) - 1)]))
    nmatch = int((near <= tol).sum())
    p(f"  {tier:11s} d12 n={len(a):5d} d16 n={len(b):5d} | common ABSOLUTE steps "
      f"n={len(common):5d} (of which <=400: {int((common<=400).sum())}) | "
      f"progress matches within tol={tol:.2e}: {nmatch}/{len(pa)}")
    if tier == "sparse":
        p(f"    d12 deep steps <=400: {a[a<=400].tolist()}")
        p(f"    d16 deep steps <=400: {b[b<=400].tolist()}")
        p(f"    common deep steps <=400: {common[common<=400].tolist()}")
        p(f"    d12 deep progress <=0.16: "
          f"{[round(x,5) for x in np.sort(d12[(d12.tier=='sparse')&(d12.step<=400)]['progress'].unique())]}")
        p(f"    d16 deep progress <=0.16: "
          f"{[round(x,5) for x in np.sort(d16[(d16.tier=='sparse')&(d16.step<=400)]['progress'].unique())]}")
    if tier == "periodic":
        p(f"    d12 periodic steps <=400: {a[a<=400].tolist()}")
        p(f"    d16 periodic steps <=400: {b[b<=400].tolist()}")
        p(f"    d12 periodic progress (first 6): {[round(x,5) for x in pa[:6]]}")
        p(f"    d16 periodic progress (first 6): {[round(x,5) for x in pb[:6]]}")

p("")
p("# how many measurement points fall inside the absolute warmup window (step<=400)")
for r in sorted(META, key=lambda x: (META[x]['depth'], META[x]['seed'])):
    row = []
    for tier in ("continuous", "periodic", "sparse"):
        sub = S[(S.run == r) & (S.tier == tier)]
        st = sub["step"].unique()
        row.append(f"{tier}={int((st<=400).sum())}/{len(st)}")
    p(f"  {r:8s} " + "  ".join(row))

open(os.path.join(HERE, "structure.txt"), "w").write("\n".join(lines) + "\n")
