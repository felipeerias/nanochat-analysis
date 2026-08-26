"""Measured telemetry cost, to price the proposed additional instrumentation."""
import sys, os
sys.path.insert(0,'/home/felipe/Igalia/nanochat/nanochat-analysis/loader')
import numpy as np, pandas as pd, telemetry_load as T
ROOT='/home/felipe/Igalia/nanochat/telemetry-data/sweep/telemetry-data'
SEGS=sorted(d for d in os.listdir(ROOT) if not d.startswith('d12-iter'))
rows=[]
for seg in SEGS:
    d=T.load_segment(ROOT,seg); run=d['provenance']['manifest_run_id']
    off=d['tiers']['offline']; cont=d['tiers']['continuous']
    nit=d['provenance']['num_iterations']
    dt=T.metric(cont,'step/observed_dt')['value_scalar'].astype(float)
    dtx=T.metric(cont,'step/dt_excl_serialized_telemetry')['value_scalar'].astype(float)
    tot_wall=dt.sum()
    r=dict(run=run, iters=nit, wall_s=tot_wall, median_step_s=float(dt.median()),
           median_step_excl_s=float(dtx.median()))
    for _,x in off.iterrows():
        if x['metric'].startswith('overhead/total/'):
            r[x['metric'].replace('overhead/total/','oh_')]=float(x['value_scalar'])
    rows.append(r)
df=pd.DataFrame(rows)
pd.set_option('display.width',260,'display.max_columns',40)
oh=[c for c in df.columns if c.startswith('oh_')]
df['oh_sum']=df[oh].sum(axis=1)
df['oh_pct']=100*df.oh_sum/df.wall_s
print(df[['run','iters','wall_s','median_step_s','oh_sum','oh_pct']].to_string(index=False))
print('\nper-section total seconds (and % of wall):')
sub=df.set_index('run')[oh]
pct=(100*sub.div(df.set_index('run').wall_s,axis=0))
print(pd.concat([sub.round(1), pct.round(3).add_suffix('_%')],axis=1).T.to_string())
print('\n=== unit costs used for the instrumentation estimate (d12) ===')
d12=df[df.run=='d12-s7'].iloc[0]
n_periodic=25; n_deep=30
print(f'  median training step                 : {d12.median_step_s*1000:7.1f} ms')
print(f'  noise diagnostic (K=8 sub-batches)   : {1000*d12.oh_noise/n_periodic:7.1f} ms per periodic checkpoint '
      f'= {1000*d12.oh_noise/n_periodic/8:.1f} ms per sub-batch fwd/bwd')
print(f'  periodic gradient scan + sketches    : {1000*d12["oh_grads_ready/periodic_scan"]/n_periodic:7.1f} ms')
print(f'  probe forwards (2 x 16 rows)         : {1000*d12.oh_probes/n_periodic:7.1f} ms')
print(f'  probe-gradient sketch (deep)         : {1000*d12.oh_probe_grad_sketch/n_deep:7.1f} ms')
print(f'  exact-gradient calibration (deep)    : {1000*d12["oh_grads_ready/calibration"]/n_deep:7.1f} ms')
print(f'  update effectiveness + HVP (deep)    : {1000*d12.oh_update_effectiveness/n_deep:7.1f} ms')
print(f'  shadow fp32 acceptance arm (deep)    : {1000*d12.oh_shadow_acceptance/n_deep:7.1f} ms')
