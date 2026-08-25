"""I0004 / A0001 — apply the frozen decision rule and attempt the extrapolation.

Reads rows.csv (produced by extract.py). Prints every number that goes into
result.md. No scipy: the fits are ordinary least squares by hand.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
THRESH = 1e-4
DIRECTIONS = ("random", "gradient", "update")
FAMILIES = ("e_sym", "e_lin")


def load():
    df = pd.read_csv(os.path.join(HERE, "rows.csv"))
    return df[df.family != "verdict_code"], df[df.family == "verdict_code"]


def per_run_median(df, arm, family, direction, progress_min=None):
    s = df[(df.arm == arm) & (df.family == family) & (df.direction == direction)]
    if progress_min is not None:
        s = s[s.progress >= progress_min]
    g = s.groupby(["depth", "run"])["value"]
    return g.median().reset_index().rename(columns={"value": "median"}), g.size()


def decision(med):
    """The rule fixed before looking, verbatim:

    Supported : d16 median > max over the five d12 seeds AND d12<d14<d16 medians
    Refuted   : d16 median lies inside the d12 five-seed range
    Inconclusive: anything else (incl. non-monotone)
    """
    d12 = med[med.depth == 12]["median"].values
    d14 = float(med[med.depth == 14]["median"].iloc[0])
    d16 = float(med[med.depth == 16]["median"].iloc[0])
    d12_med = float(np.median(d12))
    lo, hi = float(d12.min()), float(d12.max())
    above = d16 > hi
    monotone = d12_med < d14 < d16
    inside = lo <= d16 <= hi
    if above and monotone:
        verdict = "supported"
    elif inside:
        verdict = "refuted"
    else:
        verdict = "inconclusive"
    return dict(verdict=verdict, d12_median=d12_med, d12_min=lo, d12_max=hi,
                d14=d14, d16=d16, d16_above_d12max=above, monotone=monotone,
                d16_inside_d12_range=inside)


def ols(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    b = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    a = y.mean() - b * x.mean()
    resid = y - (a + b * x)
    dof = n - 2
    if dof > 0:
        s2 = (resid ** 2).sum() / dof
        se_b = np.sqrt(s2 / ((x - x.mean()) ** 2).sum())
    else:
        s2, se_b = float("nan"), float("nan")
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else float("nan")
    return a, b, se_b, r2, resid


def banner(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


def main():
    df, verd = load()

    banner("1. Per-run medians over deep checkpoints (all checkpoints)")
    table = []
    for arm in ("shadow_fp32", "native"):
        for fam in FAMILIES:
            for d in DIRECTIONS:
                med, n = per_run_median(df, arm, fam, d)
                res = decision(med)
                table.append(dict(arm=arm, family=fam, direction=d, **res))
                print(f"\n--- {arm} / {fam}_{d} ---")
                print(med.to_string(index=False,
                                    formatters={"median": lambda v: f"{v:.4e}"}))
                print(f"  decision: {res['verdict']}  "
                      f"(d16>{res['d12_max']:.3e}? {res['d16_above_d12max']}; "
                      f"monotone? {res['monotone']})")
    dec = pd.DataFrame(table)
    dec.to_csv(os.path.join(HERE, "decisions.csv"), index=False)

    banner("2. Decision summary, all 12 channels")
    print(dec.to_string(index=False, float_format=lambda v: f"{v:.4e}"))

    banner("3. Headline channel: shadow_fp32 e_sym_gradient, full detail")
    med, n = per_run_median(df, "shadow_fp32", "e_sym", "gradient")
    s = df[(df.arm == "shadow_fp32") & (df.family == "e_sym")
           & (df.direction == "gradient")]
    stats = s.groupby(["depth", "run"])["value"].agg(
        ["size", "min", "median", "mean", "max",
         lambda v: (v > THRESH).sum()])
    stats.columns = ["n_ckpt", "min", "median", "mean", "max", "n_over_1e-4"]
    print(stats.to_string(float_format=lambda v: f"{v:.4e}"))
    print("\nfraction of checkpoints over the 1e-4 threshold, by depth:")
    print(s.assign(over=s.value > THRESH).groupby("depth")["over"]
          .agg(["sum", "size", "mean"]).to_string())

    banner("4. Sensitivity: matched normalized-progress window (>= 0.2)")
    print("Caveat 3 in DATASET.md: warmups are absolute, so the geometric"
          "\nprefix of deep checkpoints covers different progress fractions."
          "\nRecompute medians over checkpoints with normalized_progress >= 0.2.")
    for fam in FAMILIES:
        for d in DIRECTIONS:
            med2, n2 = per_run_median(df, "shadow_fp32", fam, d, progress_min=0.2)
            res2 = decision(med2)
            print(f"  shadow {fam}_{d}: n/run={sorted(set(n2.values))} "
                  f"d12med={res2['d12_median']:.3e} "
                  f"[{res2['d12_min']:.3e},{res2['d12_max']:.3e}] "
                  f"d14={res2['d14']:.3e} d16={res2['d16']:.3e} -> {res2['verdict']}")

    banner("5. Sensitivity: last-10-checkpoint medians (late training)")
    for fam in FAMILIES:
        for d in DIRECTIONS:
            s2 = df[(df.arm == "shadow_fp32") & (df.family == fam)
                    & (df.direction == d)]
            late = (s2.sort_values("step").groupby(["depth", "run"])
                    .tail(10).groupby(["depth", "run"])["value"].median()
                    .reset_index().rename(columns={"value": "median"}))
            r = decision(late)
            print(f"  shadow {fam}_{d}: d12med={r['d12_median']:.3e} "
                  f"[{r['d12_min']:.3e},{r['d12_max']:.3e}] "
                  f"d14={r['d14']:.3e} d16={r['d16']:.3e} -> {r['verdict']}")

    banner("6. Seed-noise floor at d12 (this dataset, headline channel)")
    for fam in FAMILIES:
        for d in DIRECTIONS:
            med3, _ = per_run_median(df, "shadow_fp32", fam, d)
            v = med3[med3.depth == 12]["median"].values
            sd_rel = v.std(ddof=1) / np.median(v)
            rng_rel = (v.max() - v.min()) / np.median(v)
            d16 = float(med3[med3.depth == 16]["median"].iloc[0])
            eff = (d16 - np.median(v)) / np.median(v)
            print(f"  {fam}_{d}: sd-relative={sd_rel:6.1%}  range-relative="
                  f"{rng_rel:6.1%}  d16-vs-d12 effect={eff:+7.1%}  "
                  f"effect/sd={(eff/sd_rel if sd_rel else float('nan')):.2f}")

    banner("7. Extrapolation attempt: median e_sym_gradient (shadow) vs depth")
    med, _ = per_run_median(df, "shadow_fp32", "e_sym", "gradient")
    pts = med.groupby("depth")["median"].median()
    x = pts.index.values.astype(float)
    y = pts.values
    print("points:", {int(a): f"{b:.4e}" for a, b in zip(x, y)})

    a, b, se_b, r2, resid = ols(x, y)
    print(f"\nlinear   y = {a:.4e} + {b:.4e}*depth   R2={r2:.4f} "
          f"se(slope)={se_b:.3e}")
    if b > 0:
        print(f"  crossing 1e-4 at depth {(THRESH - a) / b:.1f}")
        for k in (1, 2):
            for sgn, lab in ((+1, "steeper"), (-1, "shallower")):
                bb = b + sgn * k * se_b
                if bb > 0:
                    print(f"    slope {sgn*k:+d} se ({lab}): depth "
                          f"{(THRESH - a) / bb:.1f}" if False else
                          f"    slope {sgn*k:+d}se: crossing depth "
                          f"{(THRESH - (y.mean() - bb * x.mean())) / bb:.1f}")

    la, lb, lse, lr2, lresid = ols(x, np.log(y))
    print(f"\nlog-linear ln y = {la:.4f} + {lb:.4f}*depth  R2={lr2:.4f} "
          f"se(slope)={lse:.4f}   (x{np.exp(2*lb):.2f} per +2 depth)")
    if lb > 0:
        print(f"  crossing 1e-4 at depth {(np.log(THRESH) - la) / lb:.1f}")
        for sgn in (+1, -1):
            bb = lb + sgn * lse
            aa = np.log(y).mean() - bb * x.mean()
            if bb > 0:
                print(f"    slope {sgn:+d}se: crossing depth "
                      f"{(np.log(THRESH) - aa) / bb:.1f}")

    lla, llb, llse, llr2, _ = ols(np.log(x), np.log(y))
    print(f"\npower-law  ln y = {lla:.4f} + {llb:.4f}*ln depth  R2={llr2:.4f} "
          f"se(exp)={llse:.4f}")
    if llb > 0:
        print(f"  crossing 1e-4 at depth "
              f"{np.exp((np.log(THRESH) - lla) / llb):.1f}")

    print("\nLeave-one-out on the three depth points (log-linear):")
    for drop in range(3):
        keep = [i for i in range(3) if i != drop]
        xx, yy = x[keep], np.log(y[keep])
        bb = (yy[1] - yy[0]) / (xx[1] - xx[0])
        aa = yy[0] - bb * xx[0]
        cross = (np.log(THRESH) - aa) / bb if bb > 0 else float("nan")
        print(f"  drop d{int(x[drop])}: slope={bb:+.4f}/depth  "
              f"crossing depth={cross:.1f}")

    print("\nSeed-band sensitivity: refit the log-linear through the d12 seed"
          "\nextremes instead of the d12 median (d14/d16 have one seed each):")
    d12v = med[med.depth == 12]["median"].values
    for lab, v12 in (("d12=min", d12v.min()), ("d12=median", np.median(d12v)),
                     ("d12=max", d12v.max())):
        yy = np.log(np.array([v12, y[1], y[2]]))
        aa, bb, _, rr, _ = ols(x, yy)
        cross = (np.log(THRESH) - aa) / bb if bb > 0 else float("nan")
        print(f"  {lab}: slope={bb:+.4f}/depth R2={rr:.3f} crossing={cross:.1f}")

    banner("8. Verdict context (per-direction, shadow arm)")
    v = verd[verd.arm == "shadow_fp32"]
    tab = (v.assign(passed=v.value == 0)
           .groupby(["depth", "run", "direction"])["passed"]
           .agg(["sum", "size"]))
    print(tab.to_string())

    banner("9. Native arm medians (reported, not decisive)")
    for fam in FAMILIES:
        for d in DIRECTIONS:
            medn, _ = per_run_median(df, "native", fam, d)
            p = medn.groupby("depth")["median"].median()
            print(f"  native {fam}_{d}: " +
                  "  ".join(f"d{int(k)}={val:.3e}" for k, val in p.items()))


if __name__ == "__main__":
    main()
