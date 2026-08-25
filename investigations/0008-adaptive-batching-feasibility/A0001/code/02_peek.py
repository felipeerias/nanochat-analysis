import sys, os
sys.path.insert(0, '/home/felipe/Igalia/nanochat/analysis/loader')
import pandas as pd, numpy as np, telemetry_load as T
ROOT = '/home/felipe/Igalia/nanochat/telemetry-data/sweep/telemetry-data'
SEG = 'd12-s7-s0-6c217854f4174ce884dfb2b2dcc13c45'
d = T.load_segment(ROOT, SEG)
per = d['tiers']['periodic']; sp = d['tiers']['sparse']
sk = T.metric(per, 'sketch/grad')
print('sketch/grad steps:', sorted(sk['step'].unique()))
print('roles/layers:'); print(sk.groupby(['param_role','layer']).size().head(60).to_string())
print('n (role,layer) keys:', sk.groupby(['param_role','layer'],dropna=False).ngroups)
print('\nbatch_unit values:', sk['batch_unit'].unique() if 'batch_unit' in sk.columns else 'n/a')
print('\ncolumns:', list(per.columns))
pg = T.metric(sp,'sketch/probe_grad')
print('\nprobe_grad steps:', sorted(pg['step'].unique())[:40], '...n=',pg['step'].nunique())
print('probe_grad (role,layer) keys:', pg.groupby(['param_role','layer'],dropna=False).ngroups)
nc = T.metric(per,'noise/pairwise_cosines')
print('\nnoise pairwise cosines steps:', sorted(nc['step'].unique()))
print('example vector:', np.array(nc.iloc[5]['value_vector']))
print('per_sub_sq_norm example:', np.array(T.metric(per,'noise/per_sub_sq_norm').iloc[5]['value_vector']))
print('\nsparse update metrics sample:')
print(sp[sp['metric'].isin(['update/p1','update/p2','update/actual','curvature/gg','curvature/gHg'])]
      [['metric','step','acceptance_arm','value_scalar','is_defined']].head(20).to_string())
