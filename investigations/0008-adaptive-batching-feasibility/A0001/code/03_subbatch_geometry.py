"""E2: the K-way sub-batch gradient capture -- the only g_i^T g_j analogue.

Exact algebra (spec section 6):
  s2 = (sum_i ||g_i||^2 - K||gbar||^2)/(K-1),  signal_raw = ||gbar||^2 - s2/K
  => mean over unordered pairs (i<j) of  g_i . g_j  ==  signal_raw   EXACTLY.
So we can validate the SKETCHED pairwise cosines against an exact scalar.
"""
import sys, os, json
sys.path.insert(0, '/home/felipe/Igalia/nanochat/analysis/loader')
import pandas as pd, numpy as np, telemetry_load as T
ROOT = '/home/felipe/Igalia/nanochat/telemetry-data/sweep/telemetry-data'
SEGS = sorted(d for d in os.listdir(ROOT) if not d.startswith('d12-iter'))
OUT = '/home/felipe/Igalia/nanochat/analysis/investigations/0008-adaptive-batching-feasibility/A0001'

recs = []
for seg in SEGS:
    d = T.load_segment(ROOT, seg); per = d['tiers']['periodic']
    prov = d['provenance']; run = prov['manifest_run_id']
    B = prov['device_batch_size']
    def sc(m):
        r = T.metric(per, m)
        return {int(s): (float(v) if bool(dfl) else np.nan)
                for s, v, dfl in zip(r['step'], r['value_scalar'], r['is_defined'])}
    s2 = sc('noise/s2'); sig = sc('noise/signal_raw'); bn = sc('noise/b_noise')
    rt = sc('noise/row_trace')
    pss = {int(r['step']): np.asarray(r['value_vector'], float)
           for _, r in T.metric(per, 'noise/per_sub_sq_norm').iterrows()}
    pcv = {int(r['step']): (np.asarray(r['value_vector'], float) if bool(r['is_defined']) else None)
           for _, r in T.metric(per, 'noise/pairwise_cosines').iterrows()}
    npg = T.metric(per, 'noise/pairwise_cosines')
    K = int(npg['sample_count'].iloc[0]) if 'sample_count' in npg.columns else 8
    prog = {int(r['step']): float(r['normalized_progress'])
            for _, r in T.metric(per, 'noise/s2').iterrows()}
    b = B // K
    for st in sorted(pss):
        sq = pss[st]; cos = pcv.get(st)
        if cos is None: continue
        iu = [(i, j) for i in range(K) for j in range(i+1, K)]
        # exact mean pair inner product
        gbar2 = sig[st] + s2[st]/K          # ||gbar||^2 reconstructed
        I_exact = sig[st]                    # == mean_{i<j} g_i . g_j  (exact)
        denom = np.array([np.sqrt(sq[i]*sq[j]) for i, j in iu])
        I_sketch = float(np.mean(cos*denom))
        # cosine predicted from exact algebra, matched pair by pair
        cos_pred = I_exact/denom
        recs.append(dict(run=run, step=st, progress=prog[st], K=K, b=b,
                         s2=s2[st], signal_raw=sig[st], gbar2=gbar2, row_trace=rt[st],
                         b_noise=bn[st],
                         mean_sq_norm_sub=float(np.mean(sq)),
                         cos_mean=float(np.mean(cos)), cos_sd=float(np.std(cos, ddof=1)),
                         cos_min=float(cos.min()), cos_max=float(cos.max()),
                         cos_pred_mean=float(np.mean(cos_pred)),
                         I_exact=I_exact, I_sketch=I_sketch,
                         rel_err_inner=(I_sketch-I_exact)/abs(I_exact) if I_exact != 0 else np.nan,
                         abs_err_cos=float(np.mean(cos-cos_pred)),
                         sd_err_cos=float(np.std(cos-cos_pred, ddof=1)),
                         pred_cos_from_bnoise=1.0/(1.0+bn[st]/b) if np.isfinite(bn[st]) else np.nan))
df = pd.DataFrame(recs)
df.to_csv(f'{OUT}/subbatch_geometry.csv', index=False)
pd.set_option('display.width', 250, 'display.max_columns', 40)
print('rows:', len(df), ' runs:', df.run.nunique())
print('\n== sketched-vs-exact pairwise inner product (all checkpoints, all runs) ==')
e = df['rel_err_inner'].dropna()
print(f'relative error of the SKETCHED mean pair inner product vs the EXACT value:')
print(f'  median {np.median(e):+.4f}   mean {e.mean():+.4f}   sd {e.std():.4f}   '
      f'p05 {np.percentile(e,5):+.4f}  p95 {np.percentile(e,95):+.4f}  n={len(e)}')
ce = df['sd_err_cos']
print(f'per-pair cosine error (sketched - exact): mean bias {df.abs_err_cos.mean():+.5f}, '
      f'typical sd within a checkpoint {ce.median():.5f}')
print('\n== cosine between disjoint random data slices (b=%d rows each) ==' % df.b.iloc[0])
g = df.groupby('run').agg(cos_mean=('cos_mean','mean'), cos_first=('cos_mean','first'),
                          cos_last=('cos_mean','last'), b_noise_med=('b_noise','median'),
                          within_ckpt_sd=('cos_sd','mean'))
print(g.to_string())
print('\n== consistency: measured mean cosine vs 1/(1+B_noise/b) ==')
m = df.dropna(subset=['pred_cos_from_bnoise'])
print(f'  corr = {np.corrcoef(m.cos_mean, m.pred_cos_from_bnoise)[0,1]:.5f}   '
      f'median |rel diff| = {np.median(np.abs(m.cos_mean-m.pred_cos_from_bnoise)/m.pred_cos_from_bnoise):.4f}  n={len(m)}')
print('\n== trajectory of slice-to-slice gradient alignment (d12-s7) ==')
print(df[df.run=='d12-s7'][['step','progress','cos_mean','cos_sd','signal_raw','s2','b_noise']].to_string(index=False))
