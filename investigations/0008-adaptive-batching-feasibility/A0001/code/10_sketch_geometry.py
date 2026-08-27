"""E5/E6: what the count-sketched gradients actually support.

(a) g_k^T g_nominal proxy: cos( fixed-probe gradient , logical-batch gradient ).
    The frozen gradient probe is a fixed 1-4-row held-out slice -- a degenerate
    "group" -- and the logical batch is the nominal mixture. The two sketches
    live at different cadences, so we bracket in step and quote the range,
    with the probe-gradient's own decorrelation over that gap as the error bar.
(b) decorrelation of the gradient field: data+parameter motion (batch sketch)
    vs parameter motion alone (fixed-probe sketch).
(c) per-role / per-layer share of the gradient over training.
"""
import os
import numpy as np, pandas as pd
from loader import telemetry_load as T
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from _paths import DATA_ROOT, OUTPUT_ROOT
ROOT=DATA_ROOT
OUT=OUTPUT_ROOT
SEGS = sorted(d for d in os.listdir(ROOT) if not d.startswith('d12-iter'))

def gather(df_sk, df_sq):
    """step -> (keys, matrix[k_blocks,4096], sqnorms)"""
    sq = {(int(r.step), r.param_role, (-1 if pd.isna(r.layer) else int(r.layer))): float(r.value_scalar)
          for r in T.defined(df_sq).itertuples()}
    out = {}
    for st, g in T.defined(df_sk).groupby('step'):
        ks, vs, ns = [], [], []
        for r in g.itertuples():
            key = (r.param_role, -1 if pd.isna(r.layer) else int(r.layer))
            s = sq.get((int(st),)+key)
            if s is None: continue
            ks.append(key); vs.append(np.asarray(r.value_vector, np.float64)); ns.append(s)
        out[int(st)] = (ks, np.array(vs), np.array(ns))
    return out

def full_cos(a, b):
    ka, Va, na = a; kb, Vb, nb = b
    common = sorted(set(ka) & set(kb), key=str)
    ia = {k: i for i, k in enumerate(ka)}; ib = {k: i for i, k in enumerate(kb)}
    A = Va[[ia[k] for k in common]]; B = Vb[[ib[k] for k in common]]
    na2 = na[[ia[k] for k in common]].sum(); nb2 = nb[[ib[k] for k in common]].sum()
    if na2 <= 0 or nb2 <= 0: return np.nan
    return float((A*B).sum()/np.sqrt(na2*nb2))

rows_align, rows_dec, rows_share = [], [], []
for seg in SEGS:
    d = T.load_segment(ROOT, seg); run = d['provenance']['manifest_run_id']
    per, sp = d['tiers']['periodic'], d['tiers']['sparse']
    G = gather(T.metric(per,'sketch/grad'), T.metric(per,'sketch/grad_sq_norm'))
    P = gather(T.metric(sp,'sketch/probe_grad'), T.metric(sp,'sketch/probe_grad_sq_norm'))
    nit = d['provenance']['num_iterations']
    gs, ps = sorted(G), sorted(P)
    # (a) probe-vs-batch alignment, bracketed
    for s in gs:
        lo = max([p for p in ps if p <= s], default=None)
        hi = min([p for p in ps if p >= s], default=None)
        for tag, p in (('below', lo), ('above', hi)):
            if p is None: continue
            rows_align.append(dict(run=run, step=s, probe_step=p, gap=abs(p-s), side=tag,
                                   progress=s/nit, cos=full_cos(G[s], P[p])))
    # (b) decorrelation vs step gap
    for i in range(len(gs)):
        for j in range(i+1, len(gs)):
            rows_dec.append(dict(run=run, kind='batch_grad', s0=gs[i], s1=gs[j],
                                 gap=gs[j]-gs[i], progress=gs[i]/nit, cos=full_cos(G[gs[i]], G[gs[j]])))
    for i in range(len(ps)):
        for j in range(i+1, len(ps)):
            rows_dec.append(dict(run=run, kind='probe_grad', s0=ps[i], s1=ps[j],
                                 gap=ps[j]-ps[i], progress=ps[i]/nit, cos=full_cos(P[ps[i]], P[ps[j]])))
    # (c) per-role share of squared gradient norm
    for s in gs:
        ks, V, n2 = G[s]
        tot = n2.sum()
        agg = {}
        for (role, layer), v in zip(ks, n2): agg[role] = agg.get(role, 0.0)+v
        for role, v in agg.items():
            rows_share.append(dict(run=run, step=s, progress=s/nit, role=role, share=v/tot))
A = pd.DataFrame(rows_align); D = pd.DataFrame(rows_dec); S = pd.DataFrame(rows_share)
A.to_csv(f'{OUT}/probe_vs_batch_alignment.csv', index=False)
D.to_csv(f'{OUT}/grad_decorrelation.csv', index=False)
S.to_csv(f'{OUT}/role_share.csv', index=False)
pd.set_option('display.width', 220, 'display.max_rows', 200)

print('=== (a) cos( fixed 1-4-row probe gradient , logical-batch gradient ) ===')
A2 = A[A.gap <= 130]
print(A2.groupby(pd.cut(A2.progress,[-.01,.25,.5,.75,1.01],labels=['q1','q2','q3','q4']),
                 observed=True)['cos'].agg(['count','median','min','max']).to_string())
print('\n  by run (all gaps <=130 steps):')
print(A2.groupby('run')['cos'].agg(['count','median']).to_string())
print(f'\n  step-0 exact match (same theta, no gap): ')
print(A[A.gap==0][['run','step','cos']].to_string(index=False))

print('\n=== (b) gradient-field decorrelation: data+parameter vs parameter-only ===')
for kind in ('batch_grad','probe_grad'):
    k = D[(D.kind==kind)]
    kk = k[(k.gap>=90)&(k.gap<=135)]
    print(f'  {kind:11s} cos over a ~100-130 step gap: median {kk.cos.median():.4f} '
          f'(n={len(kk)}); early(prog<.25) {kk[kk.progress<.25].cos.median():.4f}, '
          f'late(prog>.75) {kk[kk.progress>.75].cos.median():.4f}')
print('  -> the difference is the DATA contribution to gradient change')

print('\n=== (c) per-role share of squared gradient norm, d12 mean over seeds ===')
s12 = S[S.run.str.startswith('d12')]
pt = s12.pivot_table(index='step', columns='role', values='share', aggfunc='mean')
pt = pt[pt.mean().sort_values(ascending=False).index[:8]]
print(pt.iloc[[0,1,6,12,18,24]].round(4).to_string())
drift = (pt.iloc[-1]-pt.iloc[1]).abs().sort_values(ascending=False)
print('\n  largest |change in share| between step 101 and the last checkpoint:')
print(drift.head(6).round(4).to_string())

fig, ax = plt.subplots(1,3, figsize=(15,3.8))
for run, g in A2.groupby('run'):
    g = g.sort_values('progress'); ax[0].plot(g.progress, g.cos, lw=.9, alpha=.8, label=run)
ax[0].set_title('cos(probe gradient, batch gradient)'); ax[0].set_xlabel('normalized progress')
ax[0].axhline(0, color='k', lw=.5); ax[0].legend(fontsize=6)
for kind, m in (('batch_grad','o'), ('probe_grad','s')):
    k = D[(D.kind==kind)&(D.gap>=90)&(D.gap<=135)]
    ax[1].plot(k.progress, k.cos, m, ms=2.5, alpha=.5, label=kind)
ax[1].legend(fontsize=7); ax[1].set_title('gradient cosine over a ~100-130 step gap')
ax[1].set_xlabel('normalized progress'); ax[1].axhline(0, color='k', lw=.5)
for c in pt.columns: ax[2].plot(pt.index, pt[c], lw=1, label=c)
ax[2].set_title('share of |g|^2 by parameter role (d12)'); ax[2].set_xlabel('step')
ax[2].set_yscale('log'); ax[2].legend(fontsize=6)
fig.tight_layout(); fig.savefig(f'{OUT}/fig/sketch_geometry.png', dpi=130)
print('\nwrote fig/sketch_geometry.png')
