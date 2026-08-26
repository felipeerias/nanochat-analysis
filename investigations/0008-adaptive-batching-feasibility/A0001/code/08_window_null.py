"""Is the window-composition / probe-progress association real, or what two
autocorrelated 24-point series do by accident?  Circular-shift null (preserves
the autocorrelation of both series)."""
import numpy as np, pandas as pd
OUT='/home/felipe/Igalia/nanochat/nanochat-analysis/investigations/0008-adaptive-batching-feasibility/A0001'
df = pd.read_csv(f'{OUT}/probe_window.csv')
print('note: all seven runs consume the IDENTICAL batch stream, so these are NOT')
print('seven independent tests -- d12x5 are literally the same windows.\n')
for run, g in df.groupby('run'):
    g = g[g.probe == g.probe.iloc[0]].dropna().sort_values('step')
    y = g['d_probe'].values
    r = y - np.polyval(np.polyfit(g['progress'].values, y, 3), g['progress'].values)
    x = g['batch/bos_count'].values
    x = (x-x.mean())/x.std(); r = (r-r.mean())/r.std()
    obs = float(np.corrcoef(x, r)[0,1])
    null = np.array([np.corrcoef(x, np.roll(r, k))[0,1] for k in range(1, len(x))])
    p = float((np.abs(null) >= abs(obs)).mean())
    print(f'  {run:8s} n={len(x):3d}  r={obs:+.3f}  circular-shift null: '
          f'|r| p95={np.percentile(np.abs(null),95):.3f}  p={p:.3f}')
