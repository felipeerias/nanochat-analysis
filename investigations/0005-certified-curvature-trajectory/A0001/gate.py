"""I0005 / A0001 - accounting for the two gates that remove checkpoints.

Gate 1 (protocol selection): curvature/verdict_code_gradient == 0, shadow arm.
Gate 2 (instrument): the eta* reliable-sign gate "reliable-sign-v2" -
  eta* is defined only when gHg > 0 AND rho = gHg/(||g|| ||Hg||) exceeds
  8 * arith_eps. Its exclusions are reported here and are NEVER interpolated.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
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
KEEP = ["curvature/eta_star", "curvature/eta_star_rho",
        "curvature/eta_star_rho_threshold", "curvature/gHg",
        "curvature/gg", "curvature/arith_eps",
        "curvature/verdict_code_gradient"]


def main():
    rows = []
    for run, seg in SEGS.items():
        sh = tl.arm(tl.load_segment(ROOT, seg)["tiers"]["sparse"], "shadow_fp32")
        sub = sh[sh["metric"].isin(KEEP)]
        for _, r in sub.iterrows():
            rows.append(dict(run=run, step=int(r["step"]),
                             p=round(float(r["normalized_progress"]), 6),
                             metric=r["metric"],
                             val=float(r["value_scalar"]) if r["is_defined"] else np.nan,
                             defined=bool(r["is_defined"]),
                             reason=r["undefined_reason"]))
    df = pd.DataFrame(rows)
    # index on the (run, step) pairs that actually exist; pivot_table would
    # otherwise build the full run x step cross product
    w = (df.set_index(["run", "step", "metric"])["val"].unstack("metric")
         .reset_index())
    pmap = df.drop_duplicates(["run", "step"]).set_index(["run", "step"])["p"]
    w["p"] = [pmap.loc[(r, s)] for r, s in zip(w.run, w.step)]
    why = (df[df.metric == "curvature/eta_star"]
           .set_index(["run", "step"])["reason"])
    w["eta_reason"] = [why.get((r, s)) for r, s in zip(w.run, w.step)]
    w["certified"] = w["curvature/verdict_code_gradient"] == 0.0
    w["eta_defined"] = w["curvature/eta_star"].notna()
    w["rho_over_tau"] = (w["curvature/eta_star_rho"]
                         / w["curvature/eta_star_rho_threshold"])
    w.to_csv(os.path.join(OUT, "gate_table.csv"), index=False)

    d12 = w[w.run.str.startswith("d12")]
    print("d12 shadow deep checkpoints:", len(d12))
    print("  gradient-certified:", int(d12.certified.sum()))
    print("  NOT certified:", int((~d12.certified).sum()))
    print()
    print("eta* reliable-sign gate:")
    print("  undefined among ALL d12 shadow checkpoints:",
          int((~d12.eta_defined).sum()))
    print("  undefined among CERTIFIED d12 checkpoints:",
          int((d12.certified & ~d12.eta_defined).sum()))
    print("  reasons (all d12):")
    print(d12[~d12.eta_defined].groupby(["eta_reason"]).size().to_string())
    print("  reasons among certified:",
          d12[d12.certified & ~d12.eta_defined].eta_reason.tolist())
    print()
    sub = d12[d12.certified]
    print("rho margin at CERTIFIED d12 checkpoints (rho / tau, tau = 8*eps):")
    print("  min %.4g   p05 %.4g   median %.4g   max %.4g"
          % (sub.rho_over_tau.min(), sub.rho_over_tau.quantile(.05),
             sub.rho_over_tau.median(), sub.rho_over_tau.max()))
    print("  min rho = %.5g   (tau = %.4g)"
          % (sub["curvature/eta_star_rho"].min(),
             sub["curvature/eta_star_rho_threshold"].iloc[0]))
    print("  min gHg = %.5g   min gg = %.5g"
          % (sub["curvature/gHg"].min(), sub["curvature/gg"].min()))
    print()
    print("NON-certified d12 checkpoints (the excluded head of training):")
    print(d12[~d12.certified][["run", "step", "p", "curvature/gHg",
                               "curvature/eta_star", "eta_reason"]]
          .sort_values(["run", "step"])
          .to_string(index=False, float_format=lambda z: "%.4g" % z))
    print()
    for run in ("d14-s7", "d16-s7"):
        o = w[w.run == run]
        print(f"{run}: {int(o.certified.sum())}/{len(o)} certified; "
              f"eta* undefined {int((~o.eta_defined).sum())} "
              f"(certified & undefined {int((o.certified & ~o.eta_defined).sum())})")


if __name__ == "__main__":
    main()
