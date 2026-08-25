"""I0004 / A0001 — where the threshold exceedances actually live.

The median is two orders of magnitude below 1e-4, so the campaign question
("would d18/d20 certify?") is a question about the upper tail and about which
checkpoints sit there, not about the median. This script looks at that.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
THRESH = 1e-4


def main():
    df = pd.read_csv(os.path.join(HERE, "rows.csv"))
    err = df[(df.family != "verdict_code") & (df.arm == "shadow_fp32")]

    print("=" * 78)
    print("A. Upper tail of shadow e_sym / e_lin, per run and direction")
    print("=" * 78)
    for fam in ("e_sym", "e_lin"):
        for d in ("random", "gradient", "update"):
            s = err[(err.family == fam) & (err.direction == d)]
            t = s.groupby(["depth", "run"])["value"].agg(
                p50="median",
                p90=lambda v: np.quantile(v, 0.90),
                pmax="max",
                n_over=lambda v: int((v > THRESH).sum()),
                n="size")
            print(f"\n--- shadow {fam}_{d} ---")
            print(t.to_string(float_format=lambda v: f"{v:.4e}"))

    print("\n" + "=" * 78)
    print("B. Every shadow checkpoint over 1e-4, any family/direction")
    print("=" * 78)
    over = err[err.value > THRESH].sort_values(["depth", "run", "step"])
    print(over[["run", "depth", "family", "direction", "step", "progress",
                "value"]].to_string(index=False,
                                    float_format=lambda v: f"{v:.4e}"))

    print("\n" + "=" * 78)
    print("C. e_sym_gradient by checkpoint rank: is the tail early or late?")
    print("=" * 78)
    s = err[(err.family == "e_sym") & (err.direction == "gradient")]
    for run, g in s.groupby("run"):
        g = g.sort_values("step")
        top = g.nlargest(4, "value")
        print(f"{run:8s} 4 largest at steps "
              + ", ".join(f"{int(r.step)}(p={r.progress:.3f}, {r.value:.2e})"
                          for r in top.itertuples()))

    print("\n" + "=" * 78)
    print("D. Restricted to the first 8 deep checkpoints (what a SHORT run sees)")
    print("=" * 78)
    for fam in ("e_sym", "e_lin"):
        for d in ("random", "gradient", "update"):
            s2 = err[(err.family == fam) & (err.direction == d)]
            early = s2.sort_values("step").groupby(["depth", "run"]).head(8)
            t = early.groupby(["depth"])["value"].agg(
                p50="median", pmax="max",
                n_over=lambda v: int((v > THRESH).sum()), n="size")
            print(f"shadow {fam}_{d}:")
            print(t.to_string(float_format=lambda v: f"{v:.4e}"))

    print("\n" + "=" * 78)
    print("E. Checkpoint-level worst-over-directions e_sym (what the verdict uses)")
    print("=" * 78)
    w = (err[err.family == "e_sym"].groupby(["depth", "run", "step"])["value"]
         .max().reset_index())
    t = w.groupby(["depth", "run"])["value"].agg(
        p50="median", p90=lambda v: np.quantile(v, .9), pmax="max",
        n_over=lambda v: int((v > THRESH).sum()), n="size")
    print(t.to_string(float_format=lambda v: f"{v:.4e}"))

    print("\n" + "=" * 78)
    print("F. Fraction of checkpoints where the gradient direction PASSED")
    print("=" * 78)
    v = df[(df.family == "verdict_code") & (df.arm == "shadow_fp32")
           & (df.direction == "gradient")]
    t = v.assign(passed=v.value == 0).groupby(["depth", "run"])["passed"].agg(
        ["sum", "size", "mean"])
    print(t.to_string())
    print("\nfailing gradient checkpoints (verdict != passed), with e_sym:")
    es = err[(err.family == "e_sym") & (err.direction == "gradient")]
    j = v.merge(es[["run", "step", "value"]], on=["run", "step"],
                suffixes=("_verdict", "_esym"))
    bad = j[j.value_verdict != 0].sort_values(["depth", "run", "step"])
    print(bad[["run", "depth", "step", "progress", "value_verdict",
               "value_esym"]].to_string(index=False,
                                        float_format=lambda x: f"{x:.4e}"))


if __name__ == "__main__":
    main()
