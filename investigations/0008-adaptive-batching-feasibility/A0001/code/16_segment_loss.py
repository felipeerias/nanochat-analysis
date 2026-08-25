"""The closest thing to a per-group L_k: per-segment loss on the FROZEN probes.

probe/content_loss_sums and probe/segment_lengths give the summed content-only
CE and the length of each packed segment (whole documents, ~40 per probe) at
every periodic checkpoint. The documents have NO identity, but they are the
SAME documents at every checkpoint of a run, so their RELATIVE difficulty can
be tracked. Q1 asks whether relative usefulness re-orders during training; this
answers the strictly weaker question of whether relative DIFFICULTY re-orders.
"""
import sys, os
sys.path.insert(0,'/home/felipe/Igalia/nanochat/analysis/loader')
import numpy as np, pandas as pd, telemetry_load as T
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT='/home/felipe/Igalia/nanochat/telemetry-data/sweep/telemetry-data'
OUT='/home/felipe/Igalia/nanochat/analysis/investigations/0008-adaptive-batching-feasibility/A0001'
SEGS=sorted(d for d in os.listdir(ROOT) if not d.startswith('d12-iter'))
def spearman(a,b):
    ra=pd.Series(a).rank().values; rb=pd.Series(b).rank().values
    return float(np.corrcoef(ra,rb)[0,1])
res=[]; curves={}
for seg in SEGS:
    d=T.load_segment(ROOT,seg); per=d['tiers']['periodic']; run=d['provenance']['manifest_run_id']
    nit=d['provenance']['num_iterations']
    cl=T.defined(T.metric(per,'probe/content_loss_sums'))
    sl=T.defined(T.metric(per,'probe/segment_lengths'))
    for pid in cl['probe_id'].unique():
        a=cl[cl.probe_id==pid].sort_values('step'); b=sl[sl.probe_id==pid].sort_values('step')
        steps=a['step'].values
        L=np.array([np.asarray(v,float) for v in a['value_vector']])
        N=np.array([np.asarray(v,float) for v in b['value_vector']])
        keep=(N[0]>0)
        M=np.where(N>0, L/np.maximum(N,1), np.nan)[:,keep]     # per-segment mean CE
        curves[(run,pid[:8])]=(steps/nit, M)
        # re-ordering: spearman of the per-segment loss vector between checkpoints
        first,last=M[1],M[-1]
        ok=np.isfinite(first)&np.isfinite(last)
        rho_fl=spearman(first[ok],last[ok])
        rhos=[spearman(M[i][np.isfinite(M[i])&np.isfinite(M[i+1])],
                       M[i+1][np.isfinite(M[i])&np.isfinite(M[i+1])]) for i in range(1,len(M)-1)]
        # does relative difficulty change more than the segments' own improvement?
        drop=(M[1]-M[-1])
        res.append(dict(run=run, probe=pid[:8], n_seg=int(keep.sum()),
                        rho_first_last=rho_fl, rho_adjacent_med=float(np.median(rhos)),
                        loss_first=float(np.nanmean(M[1])), loss_last=float(np.nanmean(M[-1])),
                        drop_mean=float(np.nanmean(drop)), drop_sd=float(np.nanstd(drop,ddof=1)),
                        cv_across_seg_first=float(np.nanstd(M[1],ddof=1)/np.nanmean(M[1])),
                        cv_across_seg_last=float(np.nanstd(M[-1],ddof=1)/np.nanmean(M[-1]))))
R=pd.DataFrame(res); R.to_csv(f'{OUT}/segment_loss.csv',index=False)
pd.set_option('display.width',240)
print(R.round(4).to_string(index=False))
print('\nrho_first_last  = Spearman between the per-segment loss ranking at the first and')
print('                  last periodic checkpoint (1.0 = no re-ordering at all)')
print('drop_mean/sd    = per-segment loss reduction over the run: mean and spread')
print(f'\npooled: median rho(first,last) = {R.rho_first_last.median():.3f}; '
      f'median adjacent-checkpoint rho = {R.rho_adjacent_med.median():.3f}')
print(f'per-segment loss reduction: mean {R.drop_mean.mean():.3f} nats, '
      f'across-segment sd {R.drop_sd.mean():.3f} nats '
      f'({100*R.drop_sd.mean()/R.drop_mean.mean():.0f}% of the mean)')
fig,ax=plt.subplots(1,2,figsize=(11,3.8))
p,M=curves[('d12-s7', list(k[1] for k in curves if k[0]=='d12-s7')[0])]
for j in range(M.shape[1]): ax[0].plot(p,M[:,j],lw=.7,alpha=.7)
ax[0].set_title('per-segment probe loss, d12-s7 (each line = one fixed document)')
ax[0].set_xlabel('normalized progress'); ax[0].set_ylabel('mean CE (nats)')
rk=np.apply_along_axis(lambda r: pd.Series(r).rank().values,1,M)
for j in range(M.shape[1]): ax[1].plot(p,rk[:,j],lw=.7,alpha=.7)
ax[1].set_title('their RANK: the ordering is nearly frozen')
ax[1].set_xlabel('normalized progress'); ax[1].set_ylabel('rank (1 = easiest)')
fig.tight_layout(); fig.savefig(f'{OUT}/fig/segment_loss.png',dpi=130)
print('\nwrote fig/segment_loss.png')
