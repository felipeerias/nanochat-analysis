"""Robustness of the window-composition / probe-progress association."""
import numpy as np, pandas as pd
OUT='/home/felipe/Igalia/nanochat/analysis/investigations/0008-adaptive-batching-feasibility/A0001'
df = pd.read_csv(f'{OUT}/probe_window.csv')
def test(g, order, drop, pred='batch/bos_count', use_log=False):
    g = g.dropna().sort_values('step')
    if drop: g = g.iloc[drop:]
    y = g['d_probe'].values
    t = np.log(g['progress'].values) if use_log else g['progress'].values
    r = y - np.polyval(np.polyfit(t, y, order), t)
    x = g[pred].values
    if len(x) < 8 or x.std() == 0: return np.nan, len(x), np.nan
    x = (x-x.mean())/x.std(); rr = (r-r.mean())/r.std()
    obs = float(np.corrcoef(x, rr)[0,1])
    null = np.array([np.corrcoef(x, np.roll(rr, k))[0,1] for k in range(1, len(x))])
    return obs, len(x), float((np.abs(null) >= abs(obs)).mean())
print('predictor = window-mean batch/bos_count ; response = fixed-probe loss change\n')
hdr = f'{"run":9s}{"probe":9s}' + ''.join(f'{f"ord{o},drop{d}":>16s}' for o in (3,5) for d in (0,3,6))
print(hdr)
for (run, probe), g in df.groupby(['run','probe']):
    line = f'{run:9s}{probe:9s}'
    for o in (3,5):
        for d in (0,3,6):
            r, n, p = test(g, o, d)
            line += f'{f"{r:+.2f}(p{p:.2f},n{n})":>16s}'
    print(line)
print('\nsame, with the trend fitted against LOG progress (better for an exponential decay):')
for (run, probe), g in df.groupby(['run','probe']):
    line = f'{run:9s}{probe:9s}'
    for o in (3,5):
        for d in (0,3):
            r, n, p = test(g, o, d, use_log=True)
            line += f'{f"{r:+.2f}(p{p:.2f},n{n})":>16s}'
    print(line)
print('\nalternative predictor = window-mean batch/mean_segment_length (order 5, drop 3):')
for (run, probe), g in df.groupby(['run','probe']):
    r, n, p = test(g, 5, 3, pred='batch/mean_segment_length')
    print(f'  {run:9s}{probe:9s} r={r:+.3f} p={p:.3f} n={n}')
