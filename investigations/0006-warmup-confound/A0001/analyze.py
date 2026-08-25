"""I0006 / A0001 - stage 3: the d12->d16 difference under two alignments.

Design
------
Every comparison point is an ACTUAL d16 measurement. Only the d12 reference it
is compared against changes:

  absolute-step alignment    d16 sample at step s  vs  d12 seeds at step s
  normalized-progress align. d16 sample at prog p  vs  d12 seeds at prog p

The d12 reference is linearly interpolated onto the requested x inside each
seed's measured range (never extrapolated). Using the same y16 in both means
the difference between the two answers is attributable to alignment alone.

Effect size is expressed in units of the d12 five-seed standard deviation at
that same x (ddof=1), i.e. the local seed noise floor:

  z(x) = ( y16(x) - median_k y12_k(x) ) / sd_k y12_k(x)

Regions follow the protocol: warmup = d16 sample at absolute step <= 400,
post   = absolute step > 400.
"""

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
S = pd.read_parquet(os.path.join(HERE, "series.parquet"))
META = json.load(open(os.path.join(HERE, "runs.json")))

D12_RUNS = sorted(r for r, m in META.items() if m["depth"] == 12)
assert len(D12_RUNS) == 5
MUON_RAMP_END = 400

# z threshold: I0001's practical rule is that an effect must clear ~2-3x the
# seed sd before five runs can distinguish it from seed noise. We use 3.
Z_SIG = 3.0


def series(run, metric, arm):
    d = S[(S.run == run) & (S.metric == metric) & (S.arm == arm)]
    if d.empty:
        return None
    d = d.sort_values("step")
    return (d["step"].to_numpy(float), d["progress"].to_numpy(float),
            d["value"].to_numpy(float))


def interp_in_range(xq, x, y):
    """Linear interpolation, NaN outside [x.min(), x.max()]. Also returns the
    width of the bracketing interval used, as an interpolation-coarseness flag."""
    out = np.full(len(xq), np.nan)
    gap = np.full(len(xq), np.nan)
    if len(x) < 2:
        return out, gap
    ok = (xq >= x[0]) & (xq <= x[-1])
    out[ok] = np.interp(xq[ok], x, y)
    j = np.clip(np.searchsorted(x, xq[ok], side="left"), 1, len(x) - 1)
    gap[ok] = x[j] - x[j - 1]
    return out, gap


def compare(metric, arm):
    """Return per-grid-point records for one family, both alignments."""
    s16 = series("d16-s7", metric, arm)
    if s16 is None:
        return None
    st16, pr16, y16 = s16
    ref = {}
    for r in D12_RUNS:
        s = series(r, metric, arm)
        if s is None:
            return None
        ref[r] = s

    recs = {}
    for align, xq, key in (("abs", st16, 0), ("prog", pr16, 1)):
        cols, gaps = [], []
        for r in D12_RUNS:
            x = ref[r][key]
            v, g = interp_in_range(xq, x, ref[r][2])
            cols.append(v)
            gaps.append(g)
        M = np.vstack(cols)                       # 5 x n
        med = np.nanmedian(M, axis=0)
        sd = np.nanstd(M, axis=0, ddof=1)
        # A seed spread below 1e-9 of the level is floating-point residue from
        # the interpolation, not variance: treat the channel as deterministic
        # there. Without this guard a config channel yields |z| ~ 1e14.
        with np.errstate(divide="ignore", invalid="ignore"):
            sd = np.where(sd <= 1e-9 * np.abs(med), 0.0, sd)
        n_ok = np.sum(~np.isnan(M), axis=0)
        gap = np.nanmedian(np.vstack(gaps), axis=0)
        recs[align] = dict(med=med, sd=sd, n_ok=n_ok, gap=gap)

    n = len(st16)
    out = pd.DataFrame({
        "step": st16, "progress": pr16, "y16": y16,
        "warmup": st16 <= MUON_RAMP_END,
    })
    for align in ("abs", "prog"):
        r = recs[align]
        d = out["y16"].to_numpy() - r["med"]
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(r["sd"] > 0, d / r["sd"],
                         np.where(np.abs(d) > 0, np.inf, 0.0))
            rel = np.where(np.abs(r["med"]) > 1e-12, d / np.abs(r["med"]), np.nan)
            sdrel = np.where(np.abs(r["med"]) > 1e-12,
                             r["sd"] / np.abs(r["med"]), np.nan)
        out[f"med12_{align}"] = r["med"]
        out[f"sd12_{align}"] = r["sd"]
        out[f"sdrel12_{align}"] = sdrel
        out[f"z_{align}"] = z
        out[f"rel_{align}"] = rel
        out[f"nok_{align}"] = r["n_ok"]
        out[f"gap_{align}"] = r["gap"]
    out = out[(out["nok_abs"] == 5) | (out["nok_prog"] == 5)]
    return out if len(out) else None


def summarize(df, mask, label):
    d = df[mask]
    if len(d) == 0:
        return None
    both = (d["nok_abs"] == 5) & (d["nok_prog"] == 5)
    d_both = d[both]
    r = {
        "region": label,
        "n_pts": len(d),
        "n_pts_both": int(both.sum()),
        "step_lo": int(d["step"].min()), "step_hi": int(d["step"].max()),
    }
    for a in ("abs", "prog"):
        dd = d[d[f"nok_{a}"] == 5]
        fin = np.isfinite(dd[f"z_{a}"])
        r[f"n_{a}"] = len(dd)
        r[f"medabsz_{a}"] = float(np.nanmedian(np.abs(dd[f"z_{a}"][fin]))) if fin.any() else np.nan
        r[f"n_inf_{a}"] = int((~fin).sum())
        r[f"medrel_{a}"] = float(np.nanmedian(dd[f"rel_{a}"]))
        r[f"medabsrel_{a}"] = float(np.nanmedian(np.abs(dd[f"rel_{a}"])))
        r[f"seedsdrel_{a}"] = float(np.nanmedian(dd[f"sdrel12_{a}"]))
        r[f"medgap_{a}"] = float(np.nanmedian(dd[f"gap_{a}"]))
        r[f"frac_zero_sd_{a}"] = float((dd[f"sd12_{a}"] == 0).mean()) if len(dd) else np.nan
        r[f"frac_exact_eq_{a}"] = float(
            (dd["y16"] == dd[f"med12_{a}"]).mean()) if len(dd) else np.nan
    # Alignment disagreement. Because both alignments compare the SAME d16
    # sample, (y16 - med12_abs) - (y16 - med12_prog) = med12_prog - med12_abs:
    # the disagreement is entirely a property of the d12 reference series, i.e.
    # how far apart d12-at-the-same-step and d12-at-the-same-progress are.
    # Expressed in pooled d12 seed sigmas.
    if len(d_both):
        gap = d_both["med12_prog"] - d_both["med12_abs"]
        pooled = np.sqrt((d_both["sd12_abs"] ** 2 + d_both["sd12_prog"] ** 2) / 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            dz = np.where(pooled > 0, np.abs(gap) / pooled,
                          np.where(np.abs(gap) > 0, np.inf, 0.0))
            dr = np.where(np.abs(d_both["med12_abs"]) > 1e-12,
                          gap / np.abs(d_both["med12_abs"]), np.nan)
        fin = np.isfinite(dz)
        r["align_dz"] = float(np.nanmedian(dz[fin])) if fin.any() else np.nan
        r["align_dz_max"] = float(np.nanmax(dz[fin])) if fin.any() else np.nan
        r["align_dz_ninf"] = int((~fin).sum())
        r["align_drel"] = float(np.nanmedian(np.abs(dr)))
        # sign flip: does the direction of the d16-d12 difference change?
        sa, sp = np.sign(d_both["rel_abs"]), np.sign(d_both["rel_prog"])
        m = np.isfinite(sa) & np.isfinite(sp)
        r["sign_flip_frac"] = float((sa[m] != sp[m]).mean()) if m.any() else np.nan
    else:
        r.update(align_dz=np.nan, align_dz_max=np.nan, align_drel=np.nan,
                 align_dz_ninf=0, sign_flip_frac=np.nan)
    return r


def main():
    fam = (S.groupby(["metric", "arm"])
             .agg(depths=("depth", lambda s: tuple(sorted(set(s)))),
                  runs=("run", lambda s: len(set(s))),
                  tier=("tier", lambda s: "|".join(sorted(set(s)))))
             .reset_index())
    universe_all = len(fam)
    fam = fam[fam["depths"].apply(lambda t: set(t) >= {12, 14, 16})]
    fam = fam[fam["runs"] == 7]
    print(f"scalar families total {universe_all}; present at all three depths "
          f"in all seven runs: {len(fam)}")

    rows, points = [], []
    for _, f in fam.iterrows():
        df = compare(f["metric"], f["arm"])
        if df is None:
            rows.append({"metric": f["metric"], "arm": f["arm"], "tier": f["tier"],
                         "region": "SKIPPED", "n_pts": 0})
            continue
        df = df.assign(metric=f["metric"], arm=f["arm"], tier=f["tier"])
        points.append(df)
        for mask, label in ((df["warmup"], "warmup"), (~df["warmup"], "post")):
            r = summarize(df, mask, label)
            if r is None:
                continue
            r["metric"], r["arm"], r["tier"] = f["metric"], f["arm"], f["tier"]
            rows.append(r)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(HERE, "per_family_region.csv"), index=False)
    pd.concat(points, ignore_index=True).to_parquet(
        os.path.join(HERE, "points.parquet"), index=False)
    print("wrote per_family_region.csv", res.shape)


if __name__ == "__main__":
    main()
