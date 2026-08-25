"""g_k^T g_nominal proxy, per slice.

From the noise records alone, EXACTLY (no new sketch needed except the recorded
cosines):   g_i . gbar = ( ||g_i||^2 + sum_{j!=i} g_i.g_j ) / K
with ||g_i||^2 from noise/per_sub_sq_norm, g_i.g_j from noise/pairwise_cosines
x the exact norms, and ||gbar||^2 = signal_raw + s2/K.
Then: does slice INDEX carry persistent structure?  The K sub-batches are
CONTIGUOUS row blocks of the device batch and the loader's best-fit packer is
deterministic and length-biased, so exchangeability is not guaranteed - and if
it fails, the estimator's iid assumption fails with it.
"""
import sys, os
sys.path.insert(0,'/home/felipe/Igalia/nanochat/analysis/loader')
import numpy as np, pandas as pd, telemetry_load as T
ROOT='/home/felipe/Igalia/nanochat/telemetry-data/sweep/telemetry-data'
OUT='/home/felipe/Igalia/nanochat/analysis/investigations/0008-adaptive-batching-feasibility/A0001'
SEGS=sorted(d for d in os.listdir(ROOT) if not d.startswith('d12-iter'))
rows=[]
for seg in SEGS:
    d=T.load_segment(ROOT,seg); per=d['tiers']['periodic']; run=d['provenance']['manifest_run_id']
    K=int(d['provenance']['telemetry_config']['noise_K'])
    def sc(m):
        r=T.metric(per,m); return {int(s):float(v) for s,v,f in zip(r['step'],r['value_scalar'],r['is_defined']) if f}
    s2=sc('noise/s2'); sig=sc('noise/signal_raw')
    pss={int(r['step']):np.asarray(r['value_vector'],float) for _,r in T.metric(per,'noise/per_sub_sq_norm').iterrows()}
    pcv={int(r['step']):np.asarray(r['value_vector'],float) for _,r in T.defined(T.metric(per,'noise/pairwise_cosines')).iterrows()}
    prog={int(r['step']):float(r['normalized_progress']) for _,r in T.metric(per,'noise/s2').iterrows()}
    for st in sorted(pcv):
        sq=pss[st]; cos=pcv[st]; nrm=np.sqrt(sq)
        C=np.zeros((K,K)); iu=np.triu_indices(K,1)
        C[iu]=cos; C=C+C.T
        G=C*np.outer(nrm,nrm)            # inner products, diagonal 0
        gbar2=sig[st]+s2[st]/K
        for i in range(K):
            dot=(sq[i]+G[i].sum())/K
            rows.append(dict(run=run, step=st, progress=prog[st], slice=i,
                             cos_with_nominal=dot/np.sqrt(sq[i]*gbar2),
                             sq_norm=sq[i]))
df=pd.DataFrame(rows); df.to_csv(f'{OUT}/slice_vs_nominal.csv',index=False)
pd.set_option('display.width',220)
print('=== cos( 4-row slice gradient , logical mean gradient ) ===')
q=pd.cut(df.progress,[-.01,.2,.4,.6,.8,1.01],labels=['0-.2','.2-.4','.4-.6','.6-.8','.8-1'])
print(df.groupby(q,observed=True)['cos_with_nominal'].agg(['count','median','std','min','max']).round(4).to_string())
print('\n=== spread ACROSS slices within a checkpoint (the "do groups differ?" signal) ===')
w=df.groupby(['run','step','progress'])['cos_with_nominal'].agg(['mean','std']).reset_index()
w['cv']=w['std']/w['mean']
qq=pd.cut(w.progress,[-.01,.2,.4,.6,.8,1.01],labels=['0-.2','.2-.4','.4-.6','.6-.8','.8-1'])
print(w.groupby(qq,observed=True)[['mean','std','cv']].median().round(4).to_string())
print('\n=== is slice INDEX persistent?  (variance of the slice-index mean vs within) ===')
for lab,sel in (('all',slice(None)),):
    piv=df.pivot_table(index=['run','step'],columns='slice',values='cos_with_nominal')
    z=piv.sub(piv.mean(axis=1),axis=0)          # remove the checkpoint mean
    between=z.mean(axis=0)                       # per-slice-index mean deviation
    within=z.std(axis=0)
    print('  slice index      :', ' '.join(f'{i:8d}' for i in piv.columns))
    print('  mean deviation   :', ' '.join(f'{v:+8.4f}' for v in between))
    print('  within-index sd  :', ' '.join(f'{v:8.4f}' for v in within))
    se=within/np.sqrt(len(z))
    print('  |mean|/s.e.      :', ' '.join(f'{abs(b)/s:8.2f}' for b,s in zip(between,se)))
    F=(len(z)*between.var(ddof=1))/ (z.values.var(ddof=1))
    print(f'  between/within variance ratio (F-like) = {F:.3f}  '
          f'(1.0 = perfectly exchangeable; n_checkpoints={len(z)})')
print('\n  norms too:')
piv2=df.pivot_table(index=['run','step'],columns='slice',values='sq_norm')
z2=(piv2.div(piv2.mean(axis=1),axis=0))
print('  mean relative ||g_i||^2 by slice index:', ' '.join(f'{v:.4f}' for v in z2.mean(axis=0)))
print('  s.e.:', ' '.join(f'{v:.4f}' for v in z2.std(axis=0)/np.sqrt(len(z2))))

print('\n=== per-run replication of the slice-index pattern ===')
print('  (the seven runs consume the SAME rows, so a real positional effect must')
print('   reproduce in every run; the 175 checkpoints are not independent)')
piv=df.pivot_table(index=['run','step'],columns='slice',values='cos_with_nominal')
z=piv.sub(piv.mean(axis=1),axis=0)
per_run=z.groupby(level=0).mean()
print(per_run.round(4).to_string())
sgn=np.sign(per_run.values)
print('\n  sign agreement across the 7 runs, per slice index:',
      ' '.join(f'{int(abs(s.sum())):d}/7' for s in sgn.T))
print('  a positional artifact would show 7/7; sampling noise gives ~4/7')
