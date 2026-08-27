"""The spec's section-7 trio, at MATCHED gaps (both live on the deep schedule):
  calib/grad_cosine_prev        exact cos between successive LOGICAL-BATCH gradients
                                -> data change + parameter motion
  sketch/probe_grad_cosine_prev cos between successive FIXED-PROBE gradients
                                -> parameter motion alone
The gap between deep checkpoints runs 1,1,2,4,8,... so the short gaps isolate
the pure data term at essentially frozen parameters.
"""
import os
import numpy as np, pandas as pd
from loader import telemetry_load as T
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from _paths import DATA_ROOT, OUTPUT_ROOT
ROOT=DATA_ROOT
OUT=OUTPUT_ROOT
SEGS=sorted(d for d in os.listdir(ROOT) if not d.startswith('d12-iter'))
rows=[]
for seg in SEGS:
    d=T.load_segment(ROOT,seg); sp=d['tiers']['sparse']; run=d['provenance']['manifest_run_id']
    nit=d['provenance']['num_iterations']
    def ser(m):
        r=T.metric(sp,m)
        return {int(s):(float(v) if bool(f) else np.nan) for s,v,f in
                zip(r['step'],r['value_scalar'],r['is_defined'])}
    cb=ser('calib/grad_cosine_prev'); cp=ser('sketch/probe_grad_cosine_prev')
    gn=ser('calib/grad_norm')
    stb=sorted(cb); stp=sorted(cp)
    for i,s in enumerate(stb):
        gap = s-stb[i-1] if i>0 else np.nan
        rows.append(dict(run=run, step=s, gap=gap, progress=s/nit,
                         cos_batch=cb[s], grad_norm=gn.get(s,np.nan)))
    for i,s in enumerate(stp):
        gap = s-stp[i-1] if i>0 else np.nan
        rows.append(dict(run=run, step=s, gap=gap, progress=s/nit, cos_probe=cp[s]))
df=pd.DataFrame(rows).groupby(['run','step','gap','progress'],dropna=False).first().reset_index()
df.to_csv(f'{OUT}/grad_change_decomposition.csv',index=False)
pd.set_option('display.width',220,'display.max_rows',300)
print('=== successive-gradient cosine by GAP (all seven runs pooled) ===')
g=df.dropna(subset=['gap'])
tab=g.groupby('gap').agg(n=('cos_batch','count'),
                         batch_med=('cos_batch','median'),
                         probe_n=('cos_probe','count'),
                         probe_med=('cos_probe','median'))
print(tab.to_string())
print('\n=== the same, split by phase of training (gaps >= 100 steps) ===')
gg=g[g.gap>=100]
for lab,sel in (('early prog<0.25',gg.progress<0.25),('mid',(gg.progress>=0.25)&(gg.progress<0.75)),
                ('late prog>=0.75',gg.progress>=0.75)):
    s=gg[sel]
    print(f'  {lab:16s} batch cos median {s.cos_batch.median():+.4f} (n={s.cos_batch.count():3d})   '
          f'probe cos median {s.cos_probe.median():+.4f} (n={s.cos_probe.count():3d})')
print('\n=== SHORT gaps: parameters barely move, so this is the pure DATA term ===')
sg=g[g.gap<=4]
print(sg[['run','step','gap','progress','cos_batch','cos_probe']].to_string(index=False))
fig,ax=plt.subplots(figsize=(6,4))
for k,lab,m in (('cos_batch','logical-batch gradient (data + parameter motion)','o'),
                ('cos_probe','fixed-probe gradient (parameter motion only)','s')):
    s=g.dropna(subset=[k])
    ax.scatter(s.gap, s[k], s=10, marker=m, alpha=.55, label=lab)
ax.set_xscale('symlog'); ax.axhline(0,color='k',lw=.5)
ax.set_xlabel('gap between checkpoints (steps)'); ax.set_ylabel('cosine with previous checkpoint')
ax.legend(fontsize=7); ax.set_title('what makes the gradient change: data, not parameter motion')
fig.tight_layout(); fig.savefig(f'{OUT}/fig/grad_change.png',dpi=130)
print('\nwrote fig/grad_change.png')
