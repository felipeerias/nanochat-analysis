"""I0003/A0001 — first look at muon/replay_update_relerr shape and columns."""
import os

import pandas as pd

from loader import telemetry_load as tl  # noqa: E402

ROOT = str(tl.DEFAULT_DATA_ROOT)
SEGS = {
    "d12-s7": "d12-s7-s0-6c217854f4174ce884dfb2b2dcc13c45",
    "d12-s8": "d12-s8-s0-2b2e72e4395440029b92226213d137bb",
    "d12-s9": "d12-s9-s0-b872e698e81b4a8fa0cd08512f43c2c2",
    "d12-s10": "d12-s10-s0-0ffeb05742bb4154a3f1afc202dc3955",
    "d12-s11": "d12-s11-s0-5fd1bf807f764954b30658872fc3e2ad",
    "d14-s7": "d14-s7-s0-b72acf7ac59942dab902e26fa0b2c80d",
    "d16-s7": "d16-s7-s0-8e0a39fc8f164343bd01ef6b915e849f",
}

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 200)

seg = SEGS["d12-s7"]
sp = tl.read_telemetry(ROOT, seg, "sparse").to_pandas()
print("sparse rows:", len(sp))
print("columns:", list(sp.columns))
muon = sp[sp["metric"].str.startswith("muon/")]
print("\nmuon metrics in sparse:")
print(muon["metric"].value_counts())

r = sp[sp["metric"] == "muon/replay_update_relerr"]
print("\nreplay_update_relerr rows:", len(r))
print(r.head(20).to_string())
print("\nis_defined:", r["is_defined"].value_counts().to_dict())
print("undefined_reason:", r["undefined_reason"].value_counts(dropna=False).to_dict())
print("phase:", r["phase"].value_counts().to_dict())
print("tier:", r["tier"].value_counts().to_dict())
print("acceptance_arm:", r["acceptance_arm"].value_counts(dropna=False).to_dict())
print("aggregation:", r["aggregation"].value_counts(dropna=False).to_dict())
print("param_role:", r["param_role"].value_counts(dropna=False).to_dict())
print("\nparameter_name unique:", r["parameter_name"].nunique())
print(sorted(r["parameter_name"].unique())[:60])
print("\nlayer:", sorted(r["layer"].dropna().unique()))
print("\nsteps:", sorted(r["step"].unique()))
print("\nnormalized_progress:", sorted(r["normalized_progress"].unique()))
print("\nvalue_scalar describe:")
print(r["value_scalar"].describe())
print("\nzeros:", (r["value_scalar"] == 0).sum())
print("zeros by step:")
print(r.assign(z=r["value_scalar"] == 0).groupby("step")["z"].agg(["sum", "count"]).head(10))
