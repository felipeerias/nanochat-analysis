"""I0004 / A0001 — extract acceptance self-consistency errors vs depth.

Selection (from the frozen protocol, README.md @ e76859c):
  - sweep, seven schema-v3 segments (d12-iter excluded: schema v1, no shadow arm)
  - metrics curvature/e_sym_* and curvature/e_lin_*, all three probe directions
  - deep checkpoints (sparse tier, phase == post_update)
  - defined rows only
  - both arms extracted; shadow_fp32 drives the decision

Writes a tidy CSV to rows.csv next to this file.
"""

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
from loader import telemetry_load as T  # noqa: E402

ROOT = str(T.DEFAULT_DATA_ROOT)

DIRECTIONS = ("random", "gradient", "update")
FAMILIES = ("e_sym", "e_lin")
METRICS = [f"curvature/{f}_{d}" for f in FAMILIES for d in DIRECTIONS]
VERDICTS = [f"curvature/verdict_code_{d}" for d in DIRECTIONS]
VERDICT_NAME = {0.0: "passed", 1.0: "inconclusive", 2.0: "failed"}


def segments():
    return [s for s in sorted(os.listdir(ROOT))
            if os.path.isdir(os.path.join(ROOT, s)) and not s.startswith("d12-iter")]


def main():
    recs = []
    meta = []
    for seg in segments():
        d = T.load_segment(ROOT, seg)
        prov = d["provenance"]
        depth = prov["user_config"]["depth"]
        seed = prov["seed"]
        run = prov["manifest_run_id"]
        n_iter = prov["num_iterations"]
        deep = prov["telemetry_deep_steps"]
        sp = d["tiers"]["sparse"]
        assert (sp["schema_version"].astype(str) == "3").all(), seg

        sub = sp[sp["metric"].isin(METRICS + VERDICTS)]
        sub = sub[sub["phase"] == "post_update"]
        n_all = len(sub)
        sub = T.defined(sub)          # explicit: drop honestly-undefined rows
        n_def = len(sub)

        for _, r in sub.iterrows():
            m = r["metric"]
            base = m.split("/", 1)[1]
            fam, direction = base.rsplit("_", 1)
            recs.append(dict(
                run=run, depth=depth, seed=seed, segment=seg,
                arm=r["acceptance_arm"], metric=m, family=fam, direction=direction,
                step=int(r["step"]), progress=float(r["normalized_progress"]),
                value=float(r["value_scalar"]),
                estimator_id=r["estimator_id"], dtype=r["dtype"],
            ))
        meta.append(dict(run=run, depth=depth, seed=seed, n_iter=n_iter,
                         n_deep=len(deep), n_rows_sparse=len(sp),
                         n_sel_rows=n_all, n_sel_defined=n_def,
                         thresholds=str(prov["telemetry_config"]["hvp_thresholds"])))

    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(HERE, "rows.csv"), index=False)
    m = pd.DataFrame(meta)
    m.to_csv(os.path.join(HERE, "runs.csv"), index=False)
    print(m.to_string(index=False))
    print()
    print("rows by arm/family/direction:")
    print(df[df.family != "verdict"].groupby(["arm", "family", "direction"]).size().to_string())
    print("checkpoints per run (shadow, e_sym_gradient):")
    print(df[(df.arm == "shadow_fp32") & (df.metric == "curvature/e_sym_gradient")]
          .groupby("run").size().to_string())


if __name__ == "__main__":
    main()
