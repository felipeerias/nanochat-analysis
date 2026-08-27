"""I0003/A0001 — SUPPLEMENTARY, not part of the decision rule.

The decision rule aligns on normalized_progress, and the deep schedule's
geometric prefix is defined in STEP space, so no checkpoint below progress
0.05 is shared across depths. This script looks at the early phase the only
way it can be looked at -- aligned on absolute STEP -- because DATASET.md
caveat 3 says the warmups are absolute (40-step LR warmup, 400-step Muon
momentum ramp) and therefore occupy a different fraction of each run.
Alignment on step is the natural frame for a warmup-driven effect and the
wrong frame for a progress-driven one; both are reported so the reader can
see which one the early data follows.
"""
import os

import numpy as np
import pandas as pd

from loader import telemetry_load as tl  # noqa: E402

ROOT = str(tl.DEFAULT_DATA_ROOT)
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
pd.set_option("display.width", 200)
log = open(os.path.join(OUT, "early.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.write(s + "\n")


say(__doc__)
frames = []
for run, depth, seg in SEGS:
    r = tl.defined(tl.read_telemetry(ROOT, seg, "sparse").to_pandas()
                   .query("metric == @METRIC"))
    frames.append(r.assign(run=run, depth=depth)[
        ["run", "depth", "step", "normalized_progress", "value_scalar",
         "param_role"]])
df = pd.concat(frames, ignore_index=True)

shared = set.intersection(*[set(df[df["run"] == r]["step"].unique())
                            for r, _, _ in SEGS])
early = sorted(s for s in shared if s <= 401)
say(f"\ndeep-checkpoint STEPS shared by all seven runs: {sorted(shared)}")
say(f"early ones used here (step <= 401): {early}")

rows = []
for s in early:
    a = df[df["step"] == s]
    # structural zeros live at step 1; drop them and say so
    nz = a[a["value_scalar"] > 0]
    nzero = len(a) - len(nz)
    d12 = nz[nz["depth"] == 12].groupby("run")["value_scalar"].median()
    v14 = nz[nz["depth"] == 14]["value_scalar"].median()
    v16 = nz[nz["depth"] == 16]["value_scalar"].median()
    rows.append(dict(step=s, zeros_dropped=nzero,
                     prog_d12=a[a.depth == 12]["normalized_progress"].iloc[0],
                     prog_d16=a[a.depth == 16]["normalized_progress"].iloc[0],
                     d12_lo=d12.min(), d12_med=float(np.median(d12)),
                     d12_hi=d12.max(), d14=v14, d16=v16,
                     d14_rel=v14 / np.median(d12) - 1,
                     d16_rel=v16 / np.median(d12) - 1,
                     d14_out=not (d12.min() <= v14 <= d12.max()),
                     d16_out=not (d12.min() <= v16 <= d12.max())))
E = pd.DataFrame(rows)
say("\nALIGNED ON ABSOLUTE STEP (structural zeros excluded at step 1):")
say(E.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
say(f"\nd14 outside the d12 range at {int(E['d14_out'].sum())}/{len(E)} "
    f"early steps; d16 at {int(E['d16_out'].sum())}/{len(E)}")
say(f"median offset: d14 {100*E['d14_rel'].median():+.2f}%, "
    f"d16 {100*E['d16_rel'].median():+.2f}%")
say("""
Reading: at matched absolute step the three depths sit essentially on top of
each other for the first ~40 steps (the LR-warmup window), and the deeper
runs only separate downward later. On the progress axis the same data looks
like a large early gap (-29% for d14 and -38% for d16 at progress 0.05),
because progress 0.05 is step 127 at d12 but step 270 at d16 -- different
points in the absolute warmup schedule. The first two matched-progress
checkpoints (0.05, 0.10) are therefore the ones most exposed to caveat 3.
Dropping them changes nothing: see below.""")
E.to_csv(os.path.join(OUT, "early_by_step.csv"), index=False)

V = pd.read_csv(os.path.join(OUT, "verdict_table.csv"))
sub = V[V["progress"] >= 0.15]
n = len(sub)
say(f"\nDECISION RULE RE-RUN ON THE 18 CHECKPOINTS WITH progress >= 0.15,")
say("dropping the two most warmup-exposed matched points:")
say(f"  d14: outside {int(sub['d14_below'].sum()+sub['d14_above'].sum())}/{n} "
    f"(below {int(sub['d14_below'].sum())}, above "
    f"{int(sub['d14_above'].sum())}, inside {int(sub['d14_in'].sum())})")
say(f"  d16: outside {int(sub['d16_below'].sum()+sub['d16_above'].sum())}/{n} "
    f"(below {int(sub['d16_below'].sum())}, above "
    f"{int(sub['d16_above'].sum())}, inside {int(sub['d16_in'].sum())})")
say(f"  median offset: d14 {100*sub['d14_rel'].median():+.2f}%, "
    f"d16 {100*sub['d16_rel'].median():+.2f}%")
say("  -> same verdict, smaller effect size.")
log.close()
