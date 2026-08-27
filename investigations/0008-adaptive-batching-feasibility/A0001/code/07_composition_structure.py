"""Structure of the only data-composition channel the dataset has, and whether
composition over a window predicts progress on the FIXED probe."""
import os, pickle
import numpy as np, pandas as pd
from loader import telemetry_load as T
from _paths import D12_CONT, DATA_ROOT, OUTPUT_ROOT
ROOT = DATA_ROOT
OUT = OUTPUT_ROOT
SP = D12_CONT
out = pickle.load(open(SP,'rb'))
c = out['d12-s7']
print('=== is the batch stream stationary?  (loader is deterministic: no RNG, shard order) ===')
for col in ['batch/bos_count','batch/mean_segment_length','batch/segments_per_row_max']:
    v = c[col].values; st = c.index.values
    print(f'  {col:32s} corr with step = {np.corrcoef(st, v)[0,1]:+.4f}  '
          f'cv = {v.std()/v.mean():.4f}  '
          f'block(252)-mean sd/within sd = {pd.Series(v).groupby(np.arange(len(v))//252).mean().std()/ (v.std()/np.sqrt(252)):.2f}')
print('  (last column > 1 would mean slow shard-level drift beyond iid sampling)')

print('\n=== does batch composition over a 101-step window predict fixed-probe progress? ===')
SEGS = sorted(d for d in os.listdir(ROOT) if not d.startswith('d12-iter'))
rows=[]
for seg in SEGS:
    d = T.load_segment(ROOT, seg); run = d['provenance']['manifest_run_id']
    cont = d['tiers']['continuous']; per = d['tiers']['periodic']
    comp = {m: T.metric(cont,m).set_index('step')['value_scalar']
            for m in ['batch/bos_count','batch/mean_segment_length']}
    pl = T.defined(T.metric(per,'probe/loss'))
    for pid, g in pl.groupby('probe_id'):
        g = g.sort_values('step')
        s = g['step'].values; v = g['value_scalar'].values
        for i in range(1, len(s)):
            lo, hi = s[i-1], s[i]
            w = {k: float(x.loc[lo:hi-1].mean()) for k,x in comp.items()}
            rows.append(dict(run=run, probe=pid[:8], step=int(hi), d_probe=float(v[i]-v[i-1]),
                             progress=float(g['normalized_progress'].values[i]), **w))
df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/probe_window.csv', index=False)
for pid, g in df.groupby('probe'):
    # remove the training-progress trend, then ask if composition explains the remainder
    g = g.dropna()
    y = g['d_probe'].values
    trend = np.polyval(np.polyfit(g['progress'].values, y, 3), g['progress'].values)
    r = y - trend
    for k in ['batch/bos_count','batch/mean_segment_length']:
        x = g[k].values; x = (x-x.mean())/x.std()
        print(f'  probe {pid}: corr(detrended probe-loss change, mean {k:26s}) = '
              f'{np.corrcoef(x, r)[0,1]:+.4f}  n={len(g)}')
    print(f'    (raw sd of the probe-loss change {y.std():.5f}; after removing the progress trend {r.std():.5f})')

print('\n=== the same test done WITHIN each run (the pooled version above is confounded) ===')
for run, g in df.groupby('run'):
    g = g[g.probe == g.probe.iloc[0]].dropna().sort_values('step')
    y = g['d_probe'].values
    trend = np.polyval(np.polyfit(g['progress'].values, y, 3), g['progress'].values)
    r = y - trend
    cs = []
    for k in ['batch/bos_count','batch/mean_segment_length']:
        x = g[k].values; x = (x-x.mean())/x.std()
        cs.append(np.corrcoef(x, r)[0,1])
    print(f'  {run:8s} n={len(g):3d}  corr(bos)={cs[0]:+.3f}  corr(seglen)={cs[1]:+.3f}')
print('\n=== is the DOCUMENT STREAM identical across all seven runs? ===')
ref = None
for seg in SEGS:
    d = T.load_segment(ROOT, seg); run = d['provenance']['manifest_run_id']
    b = T.metric(d['tiers']['continuous'],'batch/bos_count').set_index('step')['value_scalar']
    if ref is None: ref, refrun = b, run; continue
    n = min(len(ref), len(b))
    print(f'  {refrun} vs {run:8s}: first {n} steps bitwise identical = '
          f'{np.array_equal(ref.values[:n], b.values[:n])}')
