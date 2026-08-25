"""I0004 / A0001 — seed-7-matched depth comparison.

d14 and d16 have only seed 7. d12-s7 is the seed-matched d12 run, so the
three-depth ladder can be walked at a fixed seed. This removes seed as a
confound at the cost of throwing away the only error bar we have.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
THRESH = 1e-4
EARLY = [1, 2, 3, 5]


def main():
    df = pd.read_csv(os.path.join(HERE, "rows.csv"))
    err = df[df.family != "verdict_code"]
    s7 = err[err.seed == 7]

    for arm in ("shadow_fp32", "native"):
        print("=" * 78)
        print(f"Seed 7 only, arm={arm}: median over ALL deep checkpoints")
        print("=" * 78)
        for fam in ("e_sym", "e_lin"):
            for d in ("random", "gradient", "update"):
                p = (s7[(s7.arm == arm) & (s7.family == fam)
                        & (s7.direction == d)]
                     .groupby("depth")["value"].median())
                mono = p.iloc[0] < p.iloc[1] < p.iloc[2]
                print(f"  {fam}_{d}: " + "  ".join(
                    f"d{int(k)}={v:.3e}" for k, v in p.items())
                    + f"   monotone-increasing? {mono}")
        print()

    print("=" * 78)
    print("Seed 7 only, shadow: median over the first four deep checkpoints")
    print("=" * 78)
    e = s7[(s7.arm == "shadow_fp32") & s7.step.isin(EARLY)]
    for fam in ("e_sym", "e_lin"):
        for d in ("random", "gradient", "update"):
            p = (e[(e.family == fam) & (e.direction == d)]
                 .groupby("depth")["value"].median())
            mono = p.iloc[0] < p.iloc[1] < p.iloc[2]
            print(f"  {fam}_{d}: " + "  ".join(
                f"d{int(k)}={v:.3e}" for k, v in p.items())
                + f"   monotone-increasing? {mono}")

    print("\n" + "=" * 78)
    print("Seed 7 only, shadow e_sym_gradient: counts over 1e-4")
    print("=" * 78)
    g = s7[(s7.arm == "shadow_fp32") & (s7.family == "e_sym")
           & (s7.direction == "gradient")]
    print(g.groupby("depth")["value"].agg(
        n="size", p50="median", pmax="max",
        n_over=lambda v: int((v > THRESH).sum())).to_string(
        float_format=lambda v: f"{v:.4e}"))


if __name__ == "__main__":
    main()
