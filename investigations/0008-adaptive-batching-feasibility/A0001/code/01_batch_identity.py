import sys, os
sys.path.insert(0, '/home/felipe/Igalia/nanochat/nanochat-analysis/loader')
import pandas as pd, numpy as np, telemetry_load as T
ROOT = '/home/felipe/Igalia/nanochat/telemetry-data/sweep/telemetry-data'
D12 = sorted(d for d in os.listdir(ROOT) if d.startswith('d12-s'))
cols = ['batch/bos_count','batch/mean_segment_length','batch/segments_per_row_mean',
        'batch/segments_per_row_max','batch/valid_targets','loss/train_mean']
out = {}
for s in D12:
    d = T.load_segment(ROOT, s)
    c = d['tiers']['continuous']
    sub = c[c['metric'].isin(cols)][['metric','step','phase','value_scalar']]
    p = sub.pivot_table(index='step', columns='metric', values='value_scalar')
    out[s.split('-s0-')[0]] = p
    print(s.split('-s0-')[0], 'steps', p.index.min(), p.index.max(), 'phase(s)', sub['phase'].unique())
keys = list(out)
base = out[keys[0]]
print('\n--- cross-seed identity of batch composition (d12) ---')
for k in keys[1:]:
    o = out[k].reindex(base.index)
    for c in cols:
        if c in base.columns and c in o.columns:
            same = np.allclose(base[c].values, o[c].values, rtol=0, atol=0, equal_nan=True)
            corr = np.corrcoef(base[c].values, o[c].values)[0,1]
            print(f'{keys[0]} vs {k:8s} {c:32s} identical={same}  corr={corr:.6f}')
    break
# full pairwise for bos_count
print('\nbos_count pairwise identical matrix:')
for a in keys:
    print(' ', a, [int(np.array_equal(out[a]['batch/bos_count'].values, out[b]['batch/bos_count'].values)) for b in keys])
print('\nvalid_targets constant?', {k: (out[k]['batch/valid_targets'].nunique()) for k in keys})
print('bos_count describe (d12-s7):'); print(out['d12-s7']['batch/bos_count'].describe())
print('mean_segment_length describe (d12-s7):'); print(out['d12-s7']['batch/mean_segment_length'].describe())
import pickle
pickle.dump(out, open('/home/felipe/Igalia/nanochat/nanochat-analysis/investigations/0008-adaptive-batching-feasibility/A0001/d12_cont.pkl','wb'))
