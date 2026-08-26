"""Headline figures + the group-size arithmetic behind the instrumentation ask."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT='/home/felipe/Igalia/nanochat/nanochat-analysis/investigations/0008-adaptive-batching-feasibility/A0001'
sb=pd.read_csv(f'{OUT}/subbatch_geometry.csv'); fl=pd.read_csv(f'{OUT}/sketch_floor.csv')
uv=pd.read_csv(f'{OUT}/update_value.csv')
m=sb.merge(fl[['run','step','analytic_full_sd','null_block_sd']],on=['run','step'],how='left')
m.to_csv(f'{OUT}/subbatch_with_floor.csv',index=False)

print('=== slice-to-slice gradient alignment vs the sketch noise floor ===')
q=pd.cut(m.progress,[-.01,.2,.4,.6,.8,1.01],labels=['0-.2','.2-.4','.4-.6','.6-.8','.8-1'])
t=m.groupby(q,observed=True).agg(n=('cos_mean','size'), cos=('cos_mean','median'),
        floor=('analytic_full_sd','median'), b_noise=('b_noise','median'))
t['snr']=t.cos/t.floor
t['implied_B_rows']=4*(1-t.cos)/t.cos          # b(1-cos)/cos with b = 4 rows
print(t.round(4).to_string())
print('\n  (b = 4 rows per slice; implied_B_rows = the row-noise scale implied by the')
print('   measured cosine; it must be compared with the 256-row logical batch)')

print('\n=== rows per group needed for a target per-group gradient cosine ===')
for prog,B in zip(t.index, t.implied_B_rows):
    line=f'  progress {str(prog):6s} B={B:8.1f} rows :'
    for tgt in (0.3,0.5,0.8):
        line+=f'   cos>={tgt}: {B*tgt/(1-tgt):9.1f} rows'
    print(line)

fig,ax=plt.subplots(1,3,figsize=(15,4))
for run,g in m.groupby('run'):
    g=g.sort_values('progress')
    ax[0].plot(g.progress,g.cos_mean,lw=1,alpha=.85,label=run)
ax[0].fill_between([0,1],-m.analytic_full_sd.median()*2,m.analytic_full_sd.median()*2,
                   color='0.75',alpha=.6,label='+/-2 x sketch noise floor')
ax[0].set_xlabel('normalized progress'); ax[0].set_ylabel('mean cos(g_i,g_j)')
ax[0].set_title('disjoint 4-row data slices at fixed theta\n(the only g_i.g_j the dataset has)')
ax[0].legend(fontsize=6); ax[0].axhline(0,color='k',lw=.5)

s=uv[uv.arm=='shadow_fp32'].dropna(subset=['actual','p1','p2'])
ax[1].scatter(s.actual,s.p1,s=9,alpha=.6,label='p1 = g.delta (first order)')
ax[1].scatter(s.actual,s.p2,s=9,alpha=.6,label='p2 = g.delta + 0.5 delta.H.delta')
lim=[s.actual.min(),s.actual.max()]; ax[1].plot(lim,lim,'k--',lw=.7)
ax[1].set_xlabel('realized probe-loss change'); ax[1].set_ylabel('predicted')
ax[1].set_title('a first-order value model is not enough\n(R2 -0.57 vs 0.87)'); ax[1].legend(fontsize=7)

ax[2].scatter(s.progress,np.abs(s.normalized_residual),s=10,alpha=.6)
ax[2].set_yscale('log'); ax[2].set_xlabel('normalized progress')
ax[2].set_ylabel('|(a - p2)/max(|a|,|p2|)|')
ax[2].set_title('the local QUADRATIC model becomes\nvery accurate late in training')
fig.tight_layout(); fig.savefig(f'{OUT}/fig/headline.png',dpi=130)
print('\nwrote fig/headline.png')
