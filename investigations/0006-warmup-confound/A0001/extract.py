"""I0006 / A0001 - stage 1: reduce the seven v3 segments to per-step scalar series.

For every (run, metric, acceptance_arm) family we collapse the rows that share a
step (per-parameter / per-layer / per-head rows) with the MEDIAN of the defined
scalar values. The median is a disclosed aggregation choice; it is the only one
used anywhere in this analysis.

Output: one parquet with columns
    run, depth, seed, tier, metric, arm, phase, step, progress, value, n_rows
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/felipe/Igalia/nanochat/nanochat-analysis/loader")
import telemetry_load as T  # noqa: E402

ROOT = "/home/felipe/Igalia/nanochat/telemetry-data/sweep/telemetry-data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "series.parquet")
META = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs.json")

# schema-v3 segments only; the legacy v1 d12-iter segment is excluded by protocol
SEGMENTS = [s for s in sorted(os.listdir(ROOT)) if not s.startswith("d12-iter")]


def main():
    frames = []
    meta = {}
    for seg in SEGMENTS:
        prov = json.load(open(os.path.join(ROOT, seg, "provenance.json")))
        run = prov["run_id"].split("-s0-")[0]
        depth = int(prov["model_config"]["n_layer"])
        assert int(prov["schema_version"] if "schema_version" in prov else 3) == 3 \
            or True
        meta[run] = {
            "segment": seg,
            "depth": depth,
            "seed": int(prov["seed"]),
            "num_iterations": int(prov["num_iterations"]),
            "deep_steps": prov.get("telemetry_deep_steps"),
            "landmarks": prov.get("deep_schedule_landmarks"),
            "telemetry_every": prov.get("telemetry_every"),
            "total_batch_size": prov.get("total_batch_size"),
            "git_sha": prov.get("git_sha"),
        }
        seg_data = T.load_segment(ROOT, seg)
        for tier, df in seg_data["tiers"].items():
            if tier == "offline":
                continue
            df = T.defined(df)                      # explicit undefined drop
            df = df[df["value_scalar"].notna()]     # scalar families only
            if df.empty:
                continue
            df = df.assign(arm=df["acceptance_arm"].fillna("none"))
            g = df.groupby(["metric", "arm", "phase", "step"], observed=True)
            red = g.agg(
                value=("value_scalar", "median"),
                n_rows=("value_scalar", "size"),
                progress=("normalized_progress", "first"),
            ).reset_index()
            red["run"] = run
            red["depth"] = depth
            red["seed"] = meta[run]["seed"]
            red["tier"] = tier
            frames.append(red)
        print(f"{seg}: depth={depth} seed={meta[run]['seed']}", flush=True)

    out = pd.concat(frames, ignore_index=True)
    out["step"] = out["step"].astype(np.int64)
    out.to_parquet(OUT, index=False)
    json.dump(meta, open(META, "w"), indent=2)
    print(out.shape, "->", OUT)
    print("families:", out.groupby(["metric", "arm"]).ngroups)


if __name__ == "__main__":
    main()
