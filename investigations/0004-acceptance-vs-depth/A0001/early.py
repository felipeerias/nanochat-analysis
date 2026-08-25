"""I0004 / A0001 — the early-training transient.

EXPLORATORY. This slice was chosen after seeing that every shadow-arm
threshold exceedance in e_sym_gradient sits in the first five updates. It is
not part of the frozen protocol and is reported as exploratory.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
THRESH = 1e-4
EARLY_STEPS = [1, 2, 3, 5]          # post-update labels of deep steps 0,1,2,4


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    b = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    a = y.mean() - b * x.mean()
    r = y - (a + b * x)
    ss = ((y - y.mean()) ** 2).sum()
    return a, b, 1 - (r ** 2).sum() / ss if ss > 0 else float("nan")


def main():
    df = pd.read_csv(os.path.join(HERE, "rows.csv"))
    err = df[(df.family != "verdict_code") & (df.arm == "shadow_fp32")]
    s = err[(err.family == "e_sym") & (err.direction == "gradient")]
    e = s[s.step.isin(EARLY_STEPS)]

    print("=" * 78)
    print("A. e_sym_gradient (shadow) at the first four deep checkpoints")
    print("=" * 78)
    piv = e.pivot_table(index=["depth", "run"], columns="step", values="value")
    piv["median"] = piv.median(axis=1)
    piv["max"] = piv[EARLY_STEPS].max(axis=1)
    print(piv.to_string(float_format=lambda v: f"{v:.4e}"))

    print("\nApply the frozen decision rule to this EARLY slice (exploratory):")
    for stat in ("median", "max"):
        d12 = piv.xs(12, level="depth")[stat].values
        d14 = float(piv.xs(14, level="depth")[stat].iloc[0])
        d16 = float(piv.xs(16, level="depth")[stat].iloc[0])
        d12m = float(np.median(d12))
        sup = d16 > d12.max() and d12m < d14 < d16
        inside = d12.min() <= d16 <= d12.max()
        v = "supported" if sup else ("refuted" if inside else "inconclusive")
        print(f"  {stat}: d12={d12m:.3e} [{d12.min():.3e},{d12.max():.3e}] "
              f"d14={d14:.3e} d16={d16:.3e} -> {v}")
        sd_rel = d12.std(ddof=1) / d12m
        print(f"        d12 sd-relative={sd_rel:.1%}, "
              f"d16/d12med={d16 / d12m:.2f}x = {(d16/d12m - 1)/sd_rel:.1f} sd")

    print("\n" + "=" * 78)
    print("B. Extrapolating the EARLY slice to the 1e-4 threshold")
    print("=" * 78)
    for stat in ("median", "max"):
        pts = piv[stat].groupby("depth").median()
        x, y = pts.index.values.astype(float), pts.values
        a, b, r2 = ols(x, np.log(y))
        print(f"\n{stat}: " + "  ".join(f"d{int(k)}={v:.3e}"
                                        for k, v in pts.items()))
        print(f"  log-linear slope {b:+.4f}/depth (x{np.exp(2*b):.2f} per +2), "
              f"R2={r2:.3f}")
        if b > 0:
            print(f"  crossing 1e-4 at depth {(np.log(THRESH) - a) / b:.1f}")
        # pairwise slopes: the honest measure of how unstable this is
        for i, j in ((0, 1), (1, 2), (0, 2)):
            bb = (np.log(y[j]) - np.log(y[i])) / (x[j] - x[i])
            aa = np.log(y[i]) - bb * x[i]
            c = (np.log(THRESH) - aa) / bb if bb > 0 else float("nan")
            print(f"    pair d{int(x[i])}->d{int(x[j])}: slope {bb:+.4f}, "
                  f"crossing depth {c:.1f}")

    print("\n" + "=" * 78)
    print("C. Same early slice, other directions and e_lin (context)")
    print("=" * 78)
    for fam in ("e_sym", "e_lin"):
        for d in ("random", "gradient", "update"):
            s2 = err[(err.family == fam) & (err.direction == d)]
            e2 = s2[s2.step.isin(EARLY_STEPS)]
            p = e2.groupby(["depth", "run"])["value"].median().groupby(
                "depth").median()
            print(f"  {fam}_{d}: " + "  ".join(f"d{int(k)}={v:.3e}"
                                               for k, v in p.items()))

    print("\n" + "=" * 78)
    print("D. Excluding the first four checkpoints entirely (steps > 5)")
    print("=" * 78)
    for fam in ("e_sym", "e_lin"):
        for d in ("random", "gradient", "update"):
            s2 = err[(err.family == fam) & (err.direction == d)]
            r = s2[s2.step > 5]
            m = r.groupby(["depth", "run"])["value"].median()
            d12 = m.xs(12, level="depth").values
            d14 = float(m.xs(14, level="depth").iloc[0])
            d16 = float(m.xs(16, level="depth").iloc[0])
            d12m = float(np.median(d12))
            sup = d16 > d12.max() and d12m < d14 < d16
            inside = d12.min() <= d16 <= d12.max()
            v = "supported" if sup else ("refuted" if inside else "inconclusive")
            print(f"  {fam}_{d}: d12={d12m:.3e} [{d12.min():.3e},"
                  f"{d12.max():.3e}] d14={d14:.3e} d16={d16:.3e} -> {v}")


if __name__ == "__main__":
    main()
