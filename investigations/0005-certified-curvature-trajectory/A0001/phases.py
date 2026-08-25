"""I0005 / A0001 - phase-split trajectory statistics.

The recipe's landmarks are the same in normalized progress at every depth:
LR warmup ends at step 40, the Muon momentum ramp at 400, and warmdown starts
at 882/2520 = 0.350 (d12), 1316/3759 = 0.350 (d14), 1882/5376 = 0.350 (d16).
So p = 0.35 is the warmdown boundary at every depth.

Phases used here, on the certified grid:
  ramp      p < 0.16   (through the Muon momentum ramp end, step 401)
  body      0.16 <= p < 0.35   (constant peak LR)
  warmdown  p >= 0.35
"""
import json
import os

import numpy as np
import pandas as pd

from stats import spearman  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
D12 = ["d12-s7", "d12-s8", "d12-s9", "d12-s10", "d12-s11"]
WARMDOWN = 0.35
RAMP_END = 0.16
METRICS = ["curvature/gHg", "curvature/eta_star", "curvature/dhd",
           "curvature/vhv_gradient", "curvature/e_curv_gradient",
           "curvature/gg", "curvature/Hg_norm", "update/direction_norm"]


def main():
    v = pd.read_csv(os.path.join(OUT, "certified_values.csv"))
    v = v[v.is_defined].copy()
    v["p"] = v.normalized_progress.round(6)

    rows = []
    for run in sorted(v.run.unique()):
        for m in METRICS:
            s = v[(v.run == run) & (v.metric == m)].sort_values("p")
            if len(s) < 8:
                continue
            x, y = s.p.values, s.value.values
            ramp = x < RAMP_END
            body = (x >= RAMP_END) & (x < WARMDOWN)
            wd = x >= WARMDOWN
            pre = x < WARMDOWN
            rows.append(dict(
                run=run, metric=m,
                n=len(y), n_ramp=int(ramp.sum()), n_body=int(body.sum()),
                n_wd=int(wd.sum()),
                med_ramp=np.median(y[ramp]), med_body=np.median(y[body]),
                med_wd=np.median(y[wd]),
                med_pre=np.median(y[pre]),
                ratio_wd_over_pre=np.median(y[wd]) / np.median(y[pre]),
                ratio_wd_over_body=np.median(y[wd]) / np.median(y[body]),
                sp_pre=spearman(x[pre], y[pre]),
                sp_wd=spearman(x[wd], y[wd]),
                sp_all=spearman(x, y),
                # plateau: last 8 certified points
                med_tail=np.median(y[-8:]),
                sp_tail=spearman(x[-8:], y[-8:])))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "phases.csv"), index=False)

    d12 = df[df.run.isin(D12)]
    summ = d12.groupby("metric").agg(
        sp_pre_min=("sp_pre", "min"), sp_pre_max=("sp_pre", "max"),
        n_pre_pos=("sp_pre", lambda s: int((s > 0).sum())),
        sp_wd_min=("sp_wd", "min"), sp_wd_max=("sp_wd", "max"),
        n_wd_pos=("sp_wd", lambda s: int((s > 0).sum())),
        sp_tail_min=("sp_tail", "min"), sp_tail_max=("sp_tail", "max"),
        n_tail_pos=("sp_tail", lambda s: int((s > 0).sum())),
        r_min=("ratio_wd_over_pre", "min"), r_med=("ratio_wd_over_pre", "median"),
        r_max=("ratio_wd_over_pre", "max"),
        r_sd_rel=("ratio_wd_over_pre",
                  lambda s: s.std(ddof=1) / s.median())).reset_index()
    summ.to_csv(os.path.join(OUT, "phases_d12_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False, float_format=lambda z: "%.4g" % z))
    print()
    print(df[df.metric == "curvature/gHg"].to_string(
        index=False, float_format=lambda z: "%.4g" % z))
    with open(os.path.join(OUT, "phases_meta.json"), "w") as f:
        json.dump({"warmdown_p": WARMDOWN, "ramp_end_p": RAMP_END,
                   "landmarks_steps_d12": [40, 400, 882]}, f, indent=2)


if __name__ == "__main__":
    main()
