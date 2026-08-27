"""E4: the update-effectiveness records -- the dataset's only realized 'value of
an applied update'.  p1 = g^T d is exactly s = lambda^T d with lambda replaced
by the MYOPIC costate (minus the fixed-probe gradient at theta_s).
Tests whether a first/second-order local model predicts the realized change.
"""
import os
import pandas as pd, numpy as np
from loader import telemetry_load as T
from _paths import DATA_ROOT, OUTPUT_ROOT
ROOT = DATA_ROOT
OUT = OUTPUT_ROOT
SEGS = sorted(d for d in os.listdir(ROOT) if not d.startswith('d12-iter'))
M = ['update/p1','update/p2','update/actual','update/residual_p1','update/residual_p2',
     'update/normalized_residual','update/direction_norm','update/loss_before','update/loss_after',
     'curvature/gg','curvature/gHg','curvature/dhd','curvature/eta_star','curvature/Hg_norm',
     'curvature/verdict_code_gradient','curvature/verdict_code_update','curvature/eta_star_rho']
rows = []
for seg in SEGS:
    d = T.load_segment(ROOT, seg); sp = d['tiers']['sparse']; run = d['provenance']['manifest_run_id']
    for armname in ('native','shadow_fp32'):
        a = T.arm(sp, armname)
        sub = a[a['metric'].isin(M)]
        p = sub.pivot_table(index='step', columns='metric', values='value_scalar')
        dfl = sub.pivot_table(index='step', columns='metric', values='is_defined', aggfunc='first')
        prog = sub.groupby('step')['normalized_progress'].first()
        for st in p.index:
            r = dict(run=run, arm=armname, step=int(st), progress=float(prog[st]))
            for m in M:
                r[m.split('/')[1] if m.startswith('update') else m.replace('curvature/','c_')] = (
                    float(p.loc[st, m]) if m in p.columns and bool(dfl.loc[st, m]) else np.nan)
            rows.append(r)
df = pd.DataFrame(rows); df.to_csv(f'{OUT}/update_value.csv', index=False)
pd.set_option('display.width', 250, 'display.max_columns', 40)

def r2(y, yhat):
    m = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[m], yhat[m]
    return 1 - ((y-yhat)**2).sum()/((y-y.mean())**2).sum(), m.sum()

print('=== does the LOCAL model predict the REALIZED probe-loss change? ===')
for armname in ('native','shadow_fp32'):
    s = df[df.arm == armname]
    for pred in ('p1','p2'):
        v, n = r2(s['actual'].values, s[pred].values)
        rel = np.abs((s['actual']-s[pred])/s['actual'].abs().clip(lower=1e-12))
        print(f'  {armname:12s} {pred} vs actual: R2={v:.5f}  n={n}  '
              f'median |rel err| = {np.nanmedian(rel):.4f}   p90 = {np.nanpercentile(rel,90):.4f}')
print('\n=== normalized residual (a - p2)/max(|a|,|p2|) by progress third, shadow arm ===')
s = df[df.arm == 'shadow_fp32'].copy()
s['third'] = pd.cut(s.progress, [-.01, .33, .66, 1.01], labels=['early','mid','late'])
print(s.groupby('third', observed=True)['normalized_residual']
       .agg(['count','median', lambda x: np.nanpercentile(np.abs(x),90)])
       .rename(columns={'<lambda_0>':'p90_abs'}).to_string())
print('\n=== restricted to CERTIFIED shadow checkpoints (gradient direction passed) ===')
cert = s[s['c_verdict_code_gradient'] == 0.0]
print(f'  n = {len(cert)} of {len(s)} shadow checkpoints')
for pred in ('p1','p2'):
    v, n = r2(cert['actual'].values, cert[pred].values)
    print(f'  {pred} vs actual: R2={v:.6f}  n={n}')
print('\n=== is realized value predictable from LOCAL geometry alone? ===')
# regress the realized loss decrease on local scalars available BEFORE the update
s2 = s.dropna(subset=['actual','c_gg','c_gHg','c_dhd','direction_norm'])
y = -s2['actual'].values   # loss decrease = value
for name, x in [('|g|^2 (c_gg)', s2.c_gg.values), ('gHg', s2.c_gHg.values),
                ('dHd', s2.c_dhd.values), ('|delta|', s2.direction_norm.values),
                ('p1 = g.delta', -s2.p1.values), ('eta* ', s2.c_eta_star.values)]:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() > 5:
        c = np.corrcoef(x[m], y[m])[0,1]
        cs = np.corrcoef(pd.Series(x[m]).rank(), pd.Series(y[m]).rank())[0,1]
        print(f'  corr(value, {name:14s}) = {c:+.4f}   spearman {cs:+.4f}   n={m.sum()}')
print('\n=== per-run headline: median |1 - p2/actual| , shadow arm ===')
print(s.assign(relerr=np.abs(1-s.p2/s.actual)).groupby('run')['relerr'].median().to_string())
