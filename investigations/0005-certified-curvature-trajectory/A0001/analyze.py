"""I0005 / A0001 - trajectory statistics on the certified curvature set.

Reads out/*.csv written by extract.py. No scipy: Spearman and the
bootstrap-free summaries are implemented directly on numpy.

Two kinds of statement, kept separate throughout:
  SHAPE  - within one run, over normalized_progress (seed noise does not limit)
  LEVEL  - a value compared between runs (limited by I0001)
"""
import json
import os

import numpy as np
import pandas as pd

from stats import spearman  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
D12 = ["d12-s7", "d12-s8", "d12-s9", "d12-s10", "d12-s11"]

HEADLINE = ["curvature/gHg", "curvature/eta_star", "curvature/dhd",
            "curvature/vhv_gradient", "curvature/e_curv_gradient"]
SUPPORT = ["curvature/gg", "curvature/Hg_norm", "curvature/eta_star_rho",
           "update/direction_norm", "curvature/curv_snr_gradient",
           "curvature/e_fd_gradient", "curvature/e_sym_gradient",
           "curvature/e_lin_gradient"]


def main():
    v = pd.read_csv(os.path.join(OUT, "certified_values.csv"))
    v = v[v.is_defined]
    meta = json.load(open(os.path.join(OUT, "meta.json")))
    report = {}

    # ---------- the common d12 certified grid ---------------------------
    d12 = v[v.run.isin(D12)].copy()
    d12["p"] = d12.normalized_progress.round(6)
    grids = {r: set(s.p) for r, s in d12[d12.metric == "curvature/gHg"].groupby("run")}
    common = sorted(set.intersection(*grids.values()))
    report["n_common_grid"] = len(common)
    report["common_grid"] = common

    metrics = HEADLINE + SUPPORT
    long = d12[d12.metric.isin(metrics)]
    wide = long.pivot_table(index=["metric", "p"], columns="run",
                            values="value").reset_index()

    # ---------- band table on the common grid ---------------------------
    band_rows = []
    for m in metrics:
        sub = wide[(wide.metric == m) & wide.p.isin(common)].sort_values("p")
        for _, r in sub.iterrows():
            x = np.array([r[c] for c in D12], float)
            x = x[np.isfinite(x)]
            med = np.median(x)
            band_rows.append(dict(
                metric=m, p=r["p"], n=len(x), median=med,
                lo=x.min(), hi=x.max(), sd=x.std(ddof=1),
                sd_rel=abs(x.std(ddof=1) / med) if med != 0 else np.nan,
                range_rel=abs((x.max() - x.min()) / med) if med != 0 else np.nan))
    band = pd.DataFrame(band_rows)
    band.to_csv(os.path.join(OUT, "bands.csv"), index=False)

    # ---------- SHAPE: per-run monotonicity and turning points ----------
    shape_rows = []
    for m in metrics:
        for run in D12:
            s = long[(long.metric == m) & (long.run == run)].sort_values("p")
            x, y = s.p.values, s.value.values
            if len(y) < 5:
                continue
            pos = np.all(y > 0)
            rho_all = spearman(x, y)
            # split at the LR-warmup/Muon-ramp landmark region vs the body
            early = x <= 0.16   # <= step 401 at d12 (Muon momentum ramp end)
            late = ~early
            shape_rows.append(dict(
                metric=m, run=run, n=len(y),
                first=y[0], last=y[-1], last_over_first=y[-1] / y[0] if y[0] else np.nan,
                vmin=y.min(), vmax=y.max(), p_at_max=x[int(np.argmax(y))],
                p_at_min=x[int(np.argmin(y))],
                max_over_min=y.max() / y.min() if y.min() != 0 else np.nan,
                all_positive=pos,
                spearman_all=rho_all,
                spearman_early=spearman(x[early], y[early]) if early.sum() >= 4 else np.nan,
                spearman_late=spearman(x[late], y[late]) if late.sum() >= 4 else np.nan,
                frac_steps_up=float(np.mean(np.diff(y) > 0))))
    shape = pd.DataFrame(shape_rows)
    shape.to_csv(os.path.join(OUT, "shape.csv"), index=False)

    # ---------- SHAPE agreement across seeds ---------------------------
    agree_rows = []
    for m in metrics:
        s = shape[shape.metric == m]
        if s.empty:
            continue
        agree_rows.append(dict(
            metric=m,
            spearman_all_min=s.spearman_all.min(), spearman_all_max=s.spearman_all.max(),
            n_seeds_rising=int((s.spearman_all > 0).sum()),
            spearman_late_min=s.spearman_late.min(), spearman_late_max=s.spearman_late.max(),
            n_seeds_rising_late=int((s.spearman_late > 0).sum()),
            p_at_max_min=s.p_at_max.min(), p_at_max_max=s.p_at_max.max(),
            p_at_min_min=s.p_at_min.min(), p_at_min_max=s.p_at_min.max(),
            last_over_first_min=s.last_over_first.min(),
            last_over_first_med=s.last_over_first.median(),
            last_over_first_max=s.last_over_first.max(),
            max_over_min_med=s.max_over_min.median()))
    agree = pd.DataFrame(agree_rows)
    agree.to_csv(os.path.join(OUT, "shape_agreement.csv"), index=False)

    # ---------- across-seed spread summary vs I0001 --------------------
    spread = band.groupby("metric").agg(
        sd_rel_med=("sd_rel", "median"), sd_rel_max=("sd_rel", "max"),
        range_rel_med=("range_rel", "median")).reset_index()
    spread.to_csv(os.path.join(OUT, "spread_vs_i0001.csv"), index=False)

    # ---------- pairwise seed-rank concordance of the SHAPE ------------
    conc = []
    for m in metrics:
        sub = wide[(wide.metric == m) & wide.p.isin(common)].sort_values("p")
        cols = [c for c in D12 if c in sub.columns]
        rs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = sub[cols[i]].values, sub[cols[j]].values
                ok = np.isfinite(a) & np.isfinite(b)
                rs.append(spearman(a[ok], b[ok]))
        conc.append(dict(metric=m, pair_spearman_min=np.min(rs),
                         pair_spearman_med=np.median(rs),
                         pair_spearman_max=np.max(rs), n_pairs=len(rs)))
    pd.DataFrame(conc).to_csv(os.path.join(OUT, "seed_shape_concordance.csv"),
                              index=False)

    # ---------- consistency check: vhv_gradient == gHg/gg --------------
    a = long[long.metric == "curvature/vhv_gradient"].set_index(["run", "p"]).value
    g = long[long.metric == "curvature/gHg"].set_index(["run", "p"]).value
    gg = long[long.metric == "curvature/gg"].set_index(["run", "p"]).value
    rel = ((a - g / gg).abs() / a.abs()).dropna()
    report["vhv_vs_gHg_over_gg_max_relerr"] = float(rel.max())
    e = long[long.metric == "curvature/eta_star"].set_index(["run", "p"]).value
    rel2 = ((e - gg / g).abs() / e.abs()).dropna()
    report["etastar_vs_gg_over_gHg_max_relerr"] = float(rel2.max())

    # ---------- d14 / d16, described separately, never pooled ----------
    other = v[~v.run.isin(D12)].copy()
    other["p"] = other.normalized_progress.round(6)
    orows = []
    for m in HEADLINE + ["curvature/gg"]:
        for run in sorted(other.run.unique()):
            s = other[(other.metric == m) & (other.run == run)].sort_values("p")
            if len(s) < 5:
                continue
            y = s.value.values
            orows.append(dict(metric=m, run=run, n=len(y), first=y[0], last=y[-1],
                              last_over_first=y[-1] / y[0] if y[0] else np.nan,
                              vmax=y.max(), p_at_max=s.p.values[int(np.argmax(y))],
                              spearman_all=spearman(s.p.values, y),
                              spearman_late=spearman(s.p.values[s.p.values > 0.16],
                                                     y[s.p.values > 0.16])))
    pd.DataFrame(orows).to_csv(os.path.join(OUT, "depth_described.csv"), index=False)

    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in report if k != "common_grid"}, indent=2))
    print(f"\nmeta: {[(m['run'], m['n_certified'], m['n_deep']) for m in meta]}")


if __name__ == "__main__":
    main()
