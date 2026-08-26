import sys, os, json
sys.path.insert(0, '/home/felipe/Igalia/nanochat/nanochat-analysis/loader')
import pandas as pd, numpy as np
import telemetry_load as T

ROOT = '/home/felipe/Igalia/nanochat/telemetry-data/sweep/telemetry-data'
SEGS = sorted(d for d in os.listdir(ROOT) if not d.startswith('d12-iter'))

rows = []
for s in SEGS:
    d = T.load_segment(ROOT, s)
    for tier, df in d['tiers'].items():
        g = df.groupby('metric').agg(n=('metric','size'),
                                     ndef=('is_defined','sum'),
                                     nsteps=('step','nunique'))
        vec = df.groupby('metric')['value_vector'].apply(
            lambda c: int(np.mean([len(v) if v is not None else 0 for v in c.head(50)])))
        for m in g.index:
            rows.append(dict(seg=s, tier=tier, metric=m, n=int(g.loc[m,'n']),
                             ndef=int(g.loc[m,'ndef']), nsteps=int(g.loc[m,'nsteps']),
                             veclen=int(vec.loc[m])))
inv = pd.DataFrame(rows)
inv.to_csv('/home/felipe/Igalia/nanochat/nanochat-analysis/investigations/0008-adaptive-batching-feasibility/A0001/metric_inventory.csv', index=False)
piv = inv.groupby(['tier','metric']).agg(segs=('seg','nunique'), n_tot=('n','sum'),
                                         def_frac=('ndef', lambda x: 0), veclen=('veclen','max'))
piv['def_frac'] = (inv.groupby(['tier','metric'])['ndef'].sum() /
                   inv.groupby(['tier','metric'])['n'].sum()).round(3)
pd.set_option('display.max_rows', 500, 'display.width', 200)
print(piv.to_string())
print('TOTAL metric families:', inv['metric'].nunique())
