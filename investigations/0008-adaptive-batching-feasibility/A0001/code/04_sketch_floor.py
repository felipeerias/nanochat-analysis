"""E1: how precisely can a CountSketch cosine resolve g_i^T g_j?

Null construction: two DIFFERENT (role, layer) blocks occupy DISJOINT parameter
coordinates, so their true inner product is exactly 0. Any nonzero sketched
cosine between them is pure estimator noise. This gives an assumption-free
noise floor for every sketch-derived cosine in the dataset.
"""
import os
import pandas as pd, numpy as np
from loader import telemetry_load as T
from _paths import DATA_ROOT, OUTPUT_ROOT
ROOT = DATA_ROOT
OUT = OUTPUT_ROOT
SEGS = sorted(d for d in os.listdir(ROOT) if not d.startswith('d12-iter'))
rng = np.random.default_rng(20260825)

rows = []
for seg in SEGS:
    d = T.load_segment(ROOT, seg); per = d['tiers']['periodic']
    run = d['provenance']['manifest_run_id']; k = d['provenance']['telemetry_config']['sketch_k']
    sk = T.defined(T.metric(per, 'sketch/grad'))
    sq = T.defined(T.metric(per, 'sketch/grad_sq_norm'))
    sqm = {(int(r.step), r.param_role, (-1 if pd.isna(r.layer) else int(r.layer))): float(r.value_scalar)
           for r in sq.itertuples()}
    for st, grp in sk.groupby('step'):
        keys, vecs, norms = [], [], []
        for r in grp.itertuples():
            key = (r.param_role, -1 if pd.isna(r.layer) else int(r.layer))
            s = sqm.get((int(st),) + key)
            if s is None or s <= 0: continue
            keys.append(key); vecs.append(np.asarray(r.value_vector, dtype=np.float64)); norms.append(s)
        V = np.array(vecs); n2 = np.array(norms); nb = len(keys)
        # --- (a) empirical per-block null: cosines between disjoint blocks
        G = V @ V.T
        den = np.sqrt(np.outer(n2, n2))
        C = G/den
        iu = np.triu_indices(nb, 1)
        null_block = C[iu]
        # --- (b) empirical FULL-VECTOR null: random disjoint half-split
        mask = rng.random(nb) < 0.5
        if 0 < mask.sum() < nb:
            a = V[mask].sum(0); bb = V[~mask].sum(0)
            na = n2[mask].sum(); nbn = n2[~mask].sum()
            null_full = float(a @ bb/np.sqrt(na*nbn))
        else:
            null_full = np.nan
        # --- (c) ANALYTIC full-vector floor: sd = sqrt(sum_B ||a_B||^2||b_B||^2 / k)/(|a||b|)
        analytic_full = float(np.sqrt((n2**2).sum()/k)/n2.sum())
        rows.append(dict(run=run, step=int(st), n_blocks=nb, k=k,
                         null_block_sd=float(null_block.std(ddof=1)),
                         null_block_max=float(np.abs(null_block).max()),
                         analytic_block_sd=float(1/np.sqrt(k)),
                         null_full_half=null_full,
                         analytic_full_sd=analytic_full,
                         eff_blocks=float(n2.sum()**2/(n2**2).sum())))
df = pd.DataFrame(rows); df.to_csv(f'{OUT}/sketch_floor.csv', index=False)
pd.set_option('display.width', 220)
print('=== per-block sketch cosine null (true value exactly 0) ===')
print(f'  measured sd  : {df.null_block_sd.mean():.5f}  (per checkpoint mean over {df.n_blocks.iloc[0]*(df.n_blocks.iloc[0]-1)//2} disjoint block pairs)')
print(f'  analytic 1/sqrt(k) with k={df.k.iloc[0]}: {df.analytic_block_sd.iloc[0]:.5f}')
print(f'  worst |cos| seen on a truly-zero pair: {df.null_block_max.max():.4f}')
print('\n=== FULL-VECTOR sketch cosine floor (the normalization used by noise/pairwise_cosines) ===')
h = df.null_full_half.dropna()
print(f'  empirical half-split null: mean {h.mean():+.5f}  sd {h.std():.5f}  max|.| {h.abs().max():.5f}  n={len(h)}')
print(f'  analytic floor sd        : median {df.analytic_full_sd.median():.5f}  '
      f'range {df.analytic_full_sd.min():.5f}-{df.analytic_full_sd.max():.5f}')
print(f'  effective number of blocks carrying gradient mass: median {df.eff_blocks.median():.1f} of {df.n_blocks.iloc[0]}')
print('\nper run:')
print(df.groupby('run').agg(null_block_sd=('null_block_sd','mean'),
                            analytic_full_sd=('analytic_full_sd','median'),
                            eff_blocks=('eff_blocks','median')).to_string())
