"""I0003/A0001 — checkpoint grids, matrix inventory, structural zeros."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/felipe/Igalia/nanochat/nanochat-analysis/loader")
import telemetry_load as tl  # noqa: E402

ROOT = "/home/felipe/Igalia/nanochat/telemetry-data/sweep/telemetry-data"
SEGS = {
    "d12-s7": "d12-s7-s0-6c217854f4174ce884dfb2b2dcc13c45",
    "d12-s8": "d12-s8-s0-2b2e72e4395440029b92226213d137bb",
    "d12-s9": "d12-s9-s0-b872e698e81b4a8fa0cd08512f43c2c2",
    "d12-s10": "d12-s10-s0-0ffeb05742bb4154a3f1afc202dc3955",
    "d12-s11": "d12-s11-s0-5fd1bf807f764954b30658872fc3e2ad",
    "d14-s7": "d14-s7-s0-b72acf7ac59942dab902e26fa0b2c80d",
    "d16-s7": "d16-s7-s0-8e0a39fc8f164343bd01ef6b915e849f",
}
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 300)

grids = {}
for run, seg in SEGS.items():
    sp = tl.read_telemetry(ROOT, seg, "sparse").to_pandas()
    r = tl.defined(sp[sp["metric"] == "muon/replay_update_relerr"])
    ck = sorted(r["normalized_progress"].unique())
    grids[run] = ck
    nmat = r.groupby("step")["parameter_name"].nunique()
    print(f"{run}: schema={sp['schema_version'].unique()} rows={len(r)} "
          f"ckpts={len(ck)} matrices/ckpt={sorted(nmat.unique())} "
          f"roles={sorted(r['param_role'].unique())}")
    z = r.assign(z=r["value_scalar"] == 0.0).groupby(["step", "normalized_progress"])["z"].agg(["sum", "count"])
    zz = z[z["sum"] > 0]
    print("   checkpoints with exact zeros:")
    print(zz.to_string().replace("\n", "\n   "))
    # which roles are zero at the first checkpoint
    first = r[r["step"] == r["step"].min()]
    zr = first[first["value_scalar"] == 0.0]["param_role"].value_counts().to_dict()
    nr = first[first["value_scalar"] != 0.0]["param_role"].value_counts().to_dict()
    print(f"   at first ckpt: zero roles={zr}  nonzero roles={nr}")
    print()

# grid intersection
print("=== grid comparison (rounded to 1e-9) ===")
sets = {k: set(np.round(v, 9)) for k, v in grids.items()}
inter = set.intersection(*sets.values())
print(f"exact intersection over 7 runs: {len(inter)} points")
print(sorted(inter))
for k, v in sets.items():
    print(f"{k}: {len(v)} pts, unique-to-run {len(v - inter)}")
print("\nd12 union grid:", sorted(set.union(*[sets[k] for k in sets if k.startswith('d12')])))
print("\nd14 grid:", sorted(sets["d14-s7"]))
print("\nd16 grid:", sorted(sets["d16-s7"]))
