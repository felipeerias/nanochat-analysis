"""I0004 / A0001 — the extrapolation the protocol asks for, and its stability.

The protocol asks at what depth the median e_sym_gradient crosses 1e-4, with
an explicit statement of uncertainty. This script produces the numbers that
support the answer "the extrapolation is not supportable", plus the same
computation for the three channels that DID move monotonically, so the
"how far away is 1e-4 for those?" question is answered quantitatively.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
THRESH = 1e-4


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    b = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    a = y.mean() - b * x.mean()
    r = y - (a + b * x)
    ss = ((y - y.mean()) ** 2).sum()
    return a, b, (1 - (r ** 2).sum() / ss) if ss > 0 else float("nan")


def cross(a, b, log_x=False):
    if b <= 0:
        return None
    d = (np.log(THRESH) - a) / b
    return float(np.exp(d)) if log_x else float(d)


def main():
    df = pd.read_csv(os.path.join(HERE, "rows.csv"))
    s = df[(df.arm == "shadow_fp32") & (df.family != "verdict_code")]
    med = s.groupby(["family", "direction", "depth", "run"])["value"].median()

    print("=" * 78)
    print("A. Headline: median e_sym_gradient. Every reasonable variant of the")
    print("   three-point fit, so the reader can see the spread of answers.")
    print("=" * 78)
    m = med.loc["e_sym", "gradient"]
    d12 = m.loc[12].values
    y14, y16 = float(m.loc[14].iloc[0]), float(m.loc[16].iloc[0])
    x = np.array([12., 14., 16.])

    variants = []
    for lab, y12 in [("d12 = five-seed median", float(np.median(d12))),
                     ("d12 = five-seed min", float(d12.min())),
                     ("d12 = five-seed max", float(d12.max())),
                     ("d12 = seed-7 only (seed-matched)",
                      float(m.loc[12].loc["d12-s7"]))]:
        y = np.array([y12, y14, y16])
        a, b, r2 = ols(x, np.log(y))
        c = cross(a, b)
        variants.append((lab, b, r2, c))
        print(f"  {lab:34s} slope {b:+.4f}/depth  R2={r2:.3f}  "
              + (f"crossing depth {c:.0f}" if c else
                 "NO CROSSING (trend is downward)"))

    print("\n  leave-one-out on the three depth points (d12 = five-seed median):")
    y = np.array([float(np.median(d12)), y14, y16])
    for drop in range(3):
        keep = [i for i in range(3) if i != drop]
        b = (np.log(y[keep[1]]) - np.log(y[keep[0]])) / (x[keep[1]] - x[keep[0]])
        a = np.log(y[keep[0]]) - b * x[keep[0]]
        c = cross(a, b)
        print(f"    drop d{int(x[drop])}: slope {b:+.4f}/depth  "
              + (f"crossing depth {c:.0f}" if c else "NO CROSSING"))

    print("\n  how big a change is even needed?")
    print(f"    d12 five-seed median            {np.median(d12):.3e}")
    print(f"    factor to reach 1e-4            {THRESH/np.median(d12):.0f}x")
    print(f"    observed d12->d16 change        {y16/np.median(d12)-1:+.1%}")
    sd_rel = d12.std(ddof=1) / np.median(d12)
    print(f"    d12 seed sd-relative (I0001 stat) {sd_rel:.1%}")
    print(f"    I0001 detectability bar (2-3x sd) {2*sd_rel:.0%}-{3*sd_rel:.0%}")

    print("\n" + "=" * 78)
    print("B. The three channels that DID move monotonically: how far is 1e-4?")
    print("=" * 78)
    for fam, dr in (("e_sym", "random"), ("e_lin", "random"), ("e_lin", "update")):
        mm = med.loc[fam, dr]
        v12 = float(np.median(mm.loc[12].values))
        y = np.array([v12, float(mm.loc[14].iloc[0]), float(mm.loc[16].iloc[0])])
        a, b, r2 = ols(x, np.log(y))
        c = cross(a, b)
        pa, pb, pr2 = ols(np.log(x), np.log(y))
        pc = cross(pa, pb, log_x=True)
        print(f"\n  {fam}_{dr}: d12={y[0]:.3e} d14={y[1]:.3e} d16={y[2]:.3e}  "
              f"({y[2]/y[0]:.2f}x over +4 depth)")
        print(f"    exponential in depth: slope {b:+.4f}/depth R2={r2:.3f} -> "
              + (f"crossing depth {c:.0f}" if c else "no crossing"))
        print(f"    power law in depth:   exponent {pb:+.3f} R2={pr2:.3f} -> "
              + (f"crossing depth {pc:.0f}" if pc else "no crossing")
              + f"   (sqrt(depth) would be +0.500)")

    print("\n" + "=" * 78)
    print("C. Extrapolating the early transient (steps 1,2,3,5), exploratory")
    print("=" * 78)
    e = s[(s.family == "e_sym") & (s.direction == "gradient")
          & s.step.isin([1, 2, 3, 5])]
    em = e.groupby(["depth", "run"])["value"].median()
    y = np.array([float(np.median(em.loc[12].values)),
                  float(em.loc[14].iloc[0]), float(em.loc[16].iloc[0])])
    print(f"  d12={y[0]:.3e} d14={y[1]:.3e} d16={y[2]:.3e}")
    print("  d16 is already OVER 1e-4 here, and d14 is BELOW d12: the sequence")
    print("  is not monotone, so a crossing depth is not defined by a fit.")
    a, b, r2 = ols(x, np.log(y))
    print(f"  (for the record, the 3-point log-linear fit: slope {b:+.4f}, "
          f"R2={r2:.3f}, crossing depth {cross(a, b):.1f})")
    print("  the two adjacent-pair slopes it averages are "
          f"{(np.log(y[1])-np.log(y[0]))/2:+.3f} and "
          f"{(np.log(y[2])-np.log(y[1]))/2:+.3f} per unit depth "
          "- opposite signs, "
          f"{abs((np.log(y[2])-np.log(y[1]))/(np.log(y[1])-np.log(y[0]))):.1f}x "
          "apart in magnitude.")

    print("\n" + "=" * 78)
    print("D. Seed-uncertainty envelope on the crossing depth")
    print("=" * 78)
    print("d14 and d16 have ONE seed each, so their points carry an unmeasured")
    print("seed error. Take the d12 sd-relative of the same channel as a proxy,")
    print("perturb d14 and d16 by +/- that, and sweep d12 over its five seeds.")
    print("45 variants per channel.\n")
    for fam, dr in (("e_sym", "gradient"), ("e_sym", "random"),
                    ("e_lin", "random"), ("e_lin", "update")):
        mm = med.loc[fam, dr]
        v12s = mm.loc[12].values
        sd = v12s.std(ddof=1) / np.median(v12s)
        y14, y16 = float(mm.loc[14].iloc[0]), float(mm.loc[16].iloc[0])
        cs = []
        for v12 in v12s:
            for f14 in (1 - sd, 1.0, 1 + sd):
                for f16 in (1 - sd, 1.0, 1 + sd):
                    a, b, _ = ols(x, np.log([v12, y14 * f14, y16 * f16]))
                    cs.append(cross(a, b) or np.inf)
        c = np.array(cs, float)
        fin = c[np.isfinite(c)]
        print(f"  {fam}_{dr}: d12 sd-relative={sd:5.1%}   "
              f"{len(fin)}/{len(c)} variants cross at all")
        if len(fin):
            print(f"      crossing depth among those: min={fin.min():.0f}  "
                  f"median={np.median(fin):.0f}  max={fin.max():.0f}")
        else:
            print("      no variant crosses at any depth")


if __name__ == "__main__":
    main()
