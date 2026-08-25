"""The within-device-batch noise-scale estimate does not reconcile with the
observed across-step decorrelation of the logical-batch gradient."""
import numpy as np, pandas as pd
OUT='/home/felipe/Igalia/nanochat/analysis/investigations/0008-adaptive-batching-feasibility/A0001'
D=pd.read_csv(f'{OUT}/grad_change_decomposition.csv')
sb=pd.read_csv(f'{OUT}/subbatch_geometry.csv')
g=D.dropna(subset=['gap'])
late=g[(g.gap>=100)&(g.progress>=0.75)]
cb=late.cos_batch.dropna(); cp=late.cos_probe.dropna()
print('late training (progress >= 0.75), checkpoint gaps >= 100 steps')
print(f'  logical-batch gradient cosine : median {cb.median():+.4f}  p95 {np.percentile(cb,95):+.4f}  n={len(cb)}')
print(f'  fixed-probe gradient cosine   : median {cp.median():+.4f}  p05 {np.percentile(cp,5):+.4f}  n={len(cp)}')
c_par = cp.median()
for lab,c in (('median',cb.median()), ('p95 (conservative)', np.percentile(cb,95))):
    cc = c/c_par                       # remove the parameter-motion factor
    B = 256*(1/cc-1) if cc>0 else np.inf
    print(f'  parameter-motion-corrected batch cosine ({lab}) = {cc:+.4f} '
          f'-> implied row-noise scale >= {B:,.0f} rows' if np.isfinite(B) else
          f'  parameter-motion-corrected batch cosine ({lab}) = {cc:+.4f} -> unbounded')
bn = sb[sb.progress>=0.75].b_noise.median()
print(f'\n  but noise/b_noise (estimated INSIDE one 32-row device batch) says {bn:.0f} rows,')
print(f'  which predicts a 256-row batch-to-batch cosine of {1/(1+bn/256):.3f} -- not observed.')
print('  Rows inside a device batch are consecutive best-fit-packed documents from')
print('  one parquet row group (the loader has no RNG at all), so the K sub-batches')
print('  are a CLUSTERED sample, not independent corpus draws: b_noise is a lower bound.')
