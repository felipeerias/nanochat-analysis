"""I0005 / A0001 - full universe accounting.

Every scalar channel present at the certified checkpoints is scored, so the
result can state how many channels were tested rather than only the five the
protocol names. Caveat 9 (multiple comparisons) applies: this is a
descriptive sweep, and only the five declared channels are headline.
"""
import os

import numpy as np
import pandas as pd

from stats import spearman  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
D12 = ["d12-s7", "d12-s8", "d12-s9", "d12-s10", "d12-s11"]
WARMDOWN = 0.35
DECLARED = {"curvature/gHg", "curvature/eta_star", "curvature/dhd",
            "curvature/vhv_gradient", "curvature/e_curv_gradient"}


def main():
    v = pd.read_csv(os.path.join(OUT, "certified_values.csv"))
    v = v[v.run.isin(D12)].copy()
    v["p"] = v.normalized_progress.round(6)

    allm = sorted(v.metric.unique())
    defined_any = sorted(v[v.is_defined].metric.unique())
    # direction scope
    def scope(m):
        if m.endswith("_random"):
            return "random (out of scope)"
        if m.endswith("_update"):
            return "update (out of scope)"
        if m.endswith("_gradient"):
            return "gradient"
        return "direction-agnostic"

    rows = []
    d = v[v.is_defined]
    for m in defined_any:
        s = d[d.metric == m]
        if s.run.nunique() < 5:
            continue
        sp_all, sp_pre, sp_wd, ratios = [], [], [], []
        for run in D12:
            r = s[s.run == run].sort_values("p")
            if len(r) < 10:
                continue
            x, y = r.p.values, r.value.values
            sp_all.append(spearman(x, y))
            pre, wd = x < WARMDOWN, x >= WARMDOWN
            if pre.sum() >= 4 and wd.sum() >= 4:
                sp_pre.append(spearman(x[pre], y[pre]))
                sp_wd.append(spearman(x[wd], y[wd]))
                mp, mw = np.median(y[pre]), np.median(y[wd])
                ratios.append(mw / mp if mp != 0 else np.nan)
        if len(sp_all) < 5:
            continue
        # across-seed relative sd at matched checkpoints (LEVEL comparison)
        common = [p for p in sorted(s.p.unique())
                  if s[s.p == p].run.nunique() == 5]
        sds = []
        for p in common:
            arr = s[s.p == p].value.values
            med = np.median(arr)
            if med != 0:
                sds.append(abs(np.std(arr, ddof=1) / med))
        rows.append(dict(
            metric=m, scope=scope(m),
            declared=m in DECLARED,
            n_ckpt_per_run=int(np.median(s.groupby("run").size())),
            sp_all_min=min(sp_all), sp_all_max=max(sp_all),
            n_seeds_same_sign_all=max((np.array(sp_all) > 0).sum(),
                                      (np.array(sp_all) < 0).sum()),
            sp_wd_min=min(sp_wd) if sp_wd else np.nan,
            sp_wd_max=max(sp_wd) if sp_wd else np.nan,
            n_seeds_same_sign_wd=(max((np.array(sp_wd) > 0).sum(),
                                      (np.array(sp_wd) < 0).sum())
                                  if sp_wd else np.nan),
            ratio_wd_pre_med=np.median(ratios) if ratios else np.nan,
            ratio_wd_pre_min=np.min(ratios) if ratios else np.nan,
            ratio_wd_pre_max=np.max(ratios) if ratios else np.nan,
            seed_sd_rel_med=np.median(sds) if sds else np.nan))
    t = pd.DataFrame(rows).sort_values(
        ["scope", "metric"]).reset_index(drop=True)
    t.to_csv(os.path.join(OUT, "universe.csv"), index=False)

    print("scalar channels present at certified checkpoints (shadow arm):",
          len(allm))
    print("  with at least one DEFINED value:", len(defined_any))
    print("  never defined (so unavailable):",
          sorted(set(allm) - set(defined_any)))
    print("  scored (defined in all 5 runs, >=10 points):", len(t))
    print()
    print(t.groupby("scope").size().to_string())
    print()
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 200)
    show = t[t.scope != "random (out of scope)"]
    show = show[show.scope != "update (out of scope)"]
    print(show[["metric", "declared", "n_ckpt_per_run", "sp_all_min",
                "sp_all_max", "n_seeds_same_sign_all", "sp_wd_min", "sp_wd_max",
                "n_seeds_same_sign_wd", "ratio_wd_pre_med", "seed_sd_rel_med"]]
          .to_string(index=False, float_format=lambda z: "%.3g" % z))


if __name__ == "__main__":
    main()
