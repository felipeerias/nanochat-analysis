"""I0005 / A0001 - certified curvature trajectory over training.

Selection (frozen protocol, commit e76859c):
  arm = shadow_fp32 ONLY
  direction = gradient ONLY
  checkpoints where curvature/verdict_code_gradient == 0 (passed)

Writes tidy CSVs into ./out/ for the reporting script. No claims here.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
from loader import telemetry_load as tl  # noqa: E402

ROOT = str(tl.DEFAULT_DATA_ROOT)
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

D12 = {
    "d12-s7": "d12-s7-s0-6c217854f4174ce884dfb2b2dcc13c45",
    "d12-s8": "d12-s8-s0-2b2e72e4395440029b92226213d137bb",
    "d12-s9": "d12-s9-s0-b872e698e81b4a8fa0cd08512f43c2c2",
    "d12-s10": "d12-s10-s0-0ffeb05742bb4154a3f1afc202dc3955",
    "d12-s11": "d12-s11-s0-5fd1bf807f764954b30658872fc3e2ad",
}
OTHER = {  # described, NEVER pooled with d12
    "d14-s7": "d14-s7-s0-b72acf7ac59942dab902e26fa0b2c80d",
    "d16-s7": "d16-s7-s0-8e0a39fc8f164343bd01ef6b915e849f",
}
# d12-iter (legacy schema v1) is excluded by the protocol.

# The declared universe plus every other *scalar* curvature/update channel
# present at those checkpoints (recorded so the universe accounting is honest).
HEADLINE = ["curvature/gHg", "curvature/eta_star", "curvature/dhd",
            "curvature/vhv_gradient", "curvature/e_curv_gradient"]


def one_run(run, seg):
    d = tl.load_segment(ROOT, seg)
    prov = d["provenance"]
    sp = d["tiers"]["sparse"]
    sh = tl.arm(sp, "shadow_fp32")

    # --- per-direction verdicts (all three directions, for the record) ------
    verd = []
    for dname in ("random", "gradient", "update"):
        v = sh[sh["metric"] == f"curvature/verdict_code_{dname}"]
        for _, r in v.iterrows():
            verd.append(dict(run=run, direction=dname, step=int(r["step"]),
                             normalized_progress=float(r["normalized_progress"]),
                             is_defined=bool(r["is_defined"]),
                             undefined_reason=r["undefined_reason"],
                             code=(float(r["value_scalar"])
                                   if r["is_defined"] else np.nan)))
    verd = pd.DataFrame(verd)

    # checkpoint-level shadow verdict, for context only
    ckpt = sh[sh["metric"] == "curvature/shadow_verdict_code"]
    ckpt_v = pd.DataFrame(dict(
        run=run, step=ckpt["step"].astype(int).values,
        normalized_progress=ckpt["normalized_progress"].values,
        code=ckpt["value_scalar"].values))

    # --- the certified checkpoint set -------------------------------------
    g = verd[(verd.direction == "gradient") & verd.is_defined]
    ok_steps = sorted(int(s) for s in g[g.code == 0.0]["step"])

    # --- every scalar curvature/update channel at those checkpoints --------
    cand = sh[sh["metric"].str.startswith(("curvature/", "update/"))
              & sh["step"].isin(ok_steps)
              & (sh["aggregation"] == "scalar")]
    vals = pd.DataFrame(dict(
        run=run, step=cand["step"].astype(int).values,
        normalized_progress=cand["normalized_progress"].values,
        metric=cand["metric"].values,
        value=cand["value_scalar"].values,
        is_defined=cand["is_defined"].values,
        undefined_reason=cand["undefined_reason"].values))

    meta = dict(run=run, segment=seg, depth=prov.get("depth"),
                seed=prov.get("seed"), num_iterations=prov.get("num_iterations"),
                schema_version=int(sp["schema_version"].iloc[0]),
                n_deep=int(len(ckpt_v)), n_certified=len(ok_steps),
                telemetry_config_hash=str(sp["telemetry_config_hash"].iloc[0]),
                estimator_id=str(sh["estimator_id"].dropna().unique()[0]))
    return verd, ckpt_v, vals, meta


def main():
    verds, ckpts, vals, metas = [], [], [], []
    for run, seg in {**D12, **OTHER}.items():
        v, c, x, m = one_run(run, seg)
        verds.append(v); ckpts.append(c); vals.append(x); metas.append(m)
        print(f"{run}: {m['n_certified']}/{m['n_deep']} gradient-certified", flush=True)
    pd.concat(verds).to_csv(os.path.join(OUT, "verdicts.csv"), index=False)
    pd.concat(ckpts).to_csv(os.path.join(OUT, "ckpt_verdicts.csv"), index=False)
    pd.concat(vals).to_csv(os.path.join(OUT, "certified_values.csv"), index=False)
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(metas, f, indent=2, default=str)


if __name__ == "__main__":
    main()
