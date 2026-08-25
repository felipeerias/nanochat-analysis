"""I0006 / A0001 - stage 4: classify each family and emit the unsafe list.

Classification (protocol's three labels, plus two the protocol did not name but
the data forces):

  not different       |z| below the seed band in both regions
  warmup-dominated    |z| clears the band inside step<=400 and is at least 3x
                      larger there than after  ->  UNSAFE for depth claims
  post-dominated      the mirror image (difference only after the window)
  uniformly different |z| clears the band in both regions, similar size
  underpowered        fewer than 5 usable comparison points in a region

Independently of the three-way label, a family is also UNSAFE when the two
alignments disagree by more than the seed band (median |z_abs - z_prog| >= 3),
because then the cross-depth answer is a function of the analyst's x-axis.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
Z_SIG = 3.0          # I0001 practical rule: clear 2-3x the seed sd
DOM = 3.0            # ratio for "dominated"
MIN_PTS = 5

RESOURCE = ("step/", "overhead/", "mem/", "telemetry/")
CONFIG = ("optim/", "batch/")


def group_of(m):
    if m.startswith(RESOURCE):
        return "resource"
    if m.startswith(CONFIG):
        return "config"
    if m in ("loss/finite", "grad/finite"):
        return "config"
    return "dynamics"


def main():
    r = pd.read_csv(os.path.join(HERE, "per_family_region.csv"))
    r = r[r.region != "SKIPPED"]
    idx = ["metric", "arm", "tier"]
    w = r[r.region == "warmup"].set_index(idx)
    p = r[r.region == "post"].set_index(idx)
    keys = w.index.union(p.index)

    out = []
    for k in keys:
        rec = {"metric": k[0], "arm": k[1], "tier": k[2], "group": group_of(k[0])}
        for tag, tbl in (("w", w), ("p", p)):
            if k in tbl.index:
                row = tbl.loc[k]
                for c in ("n_abs", "n_prog", "n_pts_both", "medabsz_abs",
                          "medabsz_prog", "n_inf_abs", "n_inf_prog",
                          "medabsrel_abs", "medabsrel_prog", "medrel_abs",
                          "medrel_prog", "seedsdrel_abs", "align_dz",
                          "align_dz_max", "align_drel", "sign_flip_frac",
                          "frac_zero_sd_abs", "frac_exact_eq_abs",
                          "frac_exact_eq_prog", "medgap_abs", "medgap_prog"):
                    rec[f"{tag}_{c}"] = row[c]
            else:
                for c in ("n_abs", "n_prog", "n_pts_both", "medabsz_abs",
                          "medabsz_prog", "n_inf_abs", "n_inf_prog",
                          "medabsrel_abs", "medabsrel_prog", "medrel_abs",
                          "medrel_prog", "seedsdrel_abs", "align_dz",
                          "align_dz_max", "align_drel", "sign_flip_frac",
                          "frac_zero_sd_abs", "frac_exact_eq_abs",
                          "frac_exact_eq_prog", "medgap_abs", "medgap_prog"):
                    rec[f"{tag}_{c}"] = np.nan
        out.append(rec)
    F = pd.DataFrame(out)

    # deterministic = no seed spread anywhere -> z is undefined, use rel diff
    F["deterministic"] = (F["w_frac_zero_sd_abs"].fillna(1) == 1) & \
                         (F["p_frac_zero_sd_abs"].fillna(1) == 1)

    def label(row, align="abs"):
        nw = row[f"w_n_{align}"]
        npp = row[f"p_n_{align}"]
        if not (nw >= MIN_PTS) or not (npp >= MIN_PTS):
            return "underpowered"
        if row["deterministic"]:
            # identical schedule/config channel, or a genuine recipe difference
            ew = row[f"w_frac_exact_eq_{align}"]
            ep = row[f"p_frac_exact_eq_{align}"]
            if ew == 1 and ep == 1:
                return "identical (deterministic)"
            if ew == 1 and ep < 1:
                return "post-dominated (deterministic)"
            if ew < 1 and ep == 1:
                return "warmup-dominated (deterministic)"
            return "uniformly different (deterministic)"
        zw, zp = row[f"w_medabsz_{align}"], row[f"p_medabsz_{align}"]
        if not np.isfinite(zw) or not np.isfinite(zp):
            return "unquantifiable"
        if zw < Z_SIG and zp < Z_SIG:
            return "not different"
        if zw >= Z_SIG and zw >= DOM * max(zp, 1e-12):
            return "warmup-dominated"
        if zp >= Z_SIG and zp >= DOM * max(zw, 1e-12):
            return "post-dominated"
        return "uniformly different"

    F["verdict"] = F.apply(lambda r: label(r, "abs"), axis=1)
    F["verdict_prog"] = F.apply(lambda r: label(r, "prog"), axis=1)
    F["verdict_flip"] = (F["verdict"] != F["verdict_prog"])

    # alignment instability: does the answer depend on the x-axis?
    F["align_unstable_warmup"] = F["w_align_dz"] >= Z_SIG
    F["align_unstable_post"] = F["p_align_dz"] >= Z_SIG
    F["align_unstable"] = F["align_unstable_warmup"] | F["align_unstable_post"]
    # untestable: the measurement cadence puts too few d16 samples inside the
    # absolute warmup window to say anything about it
    F["untestable_warmup"] = F["w_n_abs"].fillna(0) < MIN_PTS

    reasons = []
    for _, r in F.iterrows():
        why = []
        if r["verdict"].startswith("warmup-dominated"):
            why.append("warmup-dominated")
        if r["align_unstable_warmup"]:
            why.append("alignment-unstable in window")
        if r["align_unstable_post"]:
            why.append("alignment-unstable after window")
        if r["verdict_flip"]:
            why.append("verdict flips with alignment")
        if r["untestable_warmup"]:
            why.append("window untestable (<%d pts)" % MIN_PTS)
        reasons.append("; ".join(why))
    F["why_unsafe"] = reasons
    F["unsafe"] = F["why_unsafe"] != ""

    F = F.sort_values(["group", "verdict", "metric", "arm"])
    F.to_csv(os.path.join(HERE, "families.csv"), index=False)

    print("=== universe ===")
    print(f"scalar families keyed (metric, acceptance_arm): {len(F)}")
    print(F.groupby(["group"]).size().to_string())
    print()
    print("=== verdicts under ABSOLUTE-STEP alignment ===")
    print(pd.crosstab(F["verdict"], F["group"]).to_string())
    print()
    print("=== verdicts under NORMALIZED-PROGRESS alignment ===")
    print(pd.crosstab(F["verdict_prog"], F["group"]).to_string())
    print()
    print("=== verdict flips (abs -> prog), families testable under both ===")
    t = F[(F.verdict != "underpowered") & (F.verdict_prog != "underpowered")]
    print(f"{int(t.verdict_flip.sum())} of {len(t)} testable families change verdict")
    print(pd.crosstab(t["verdict"], t["verdict_prog"]).to_string())
    print()
    print("=== alignment disagreement, median |d12(step) - d12(progress)| in "
          "pooled seed sigmas ===")
    for g in ("dynamics", "resource", "config"):
        s = F[F.group == g]
        print(f"  {g:9s} warmup median={s.w_align_dz.median():.2f} "
              f"p90={s.w_align_dz.quantile(.9):.2f} | "
              f"post median={s.p_align_dz.median():.2f} "
              f"p90={s.p_align_dz.quantile(.9):.2f}")
    print()
    print("=== UNSAFE for depth claims ===")
    u = F[F["unsafe"]]
    print(f"{len(u)} of {len(F)} families flagged")
    print(pd.crosstab(u["group"], u["verdict"]).to_string())
    print()
    print("reason counts:")
    from collections import Counter
    c = Counter(x for r in F["why_unsafe"] for x in r.split("; ") if x)
    for k, v in c.most_common():
        print(f"  {k}: {v}")
    print()
    print("=== SAFE families (dynamics group) ===")
    s = F[(~F.unsafe) & (F.group == "dynamics")]
    print(f"{len(s)} of {int((F.group=='dynamics').sum())} dynamics families")
    for _, r in s.iterrows():
        print(f"  {r.metric}|{r.arm:11s} {r.tier:10s} {r.verdict}")


if __name__ == "__main__":
    main()
