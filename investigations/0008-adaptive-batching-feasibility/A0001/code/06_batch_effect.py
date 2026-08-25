"""E3: how much of the trajectory is attributable to WHICH batch was used?

Structural fact established in 01: all five d12 seeds consume the IDENTICAL
batch sequence (batch/bos_count agrees bitwise at all 2520 steps). So at step s
the batch is a constant across seeds and the component of the detrended loss
that is COMMON to all five seeds is exactly the batch-attributable component.
That common component is an upper bound on any batch/group value signal
recoverable from this dataset's loss channel.
"""
import sys, os, pickle
sys.path.insert(0, '/home/felipe/Igalia/nanochat/analysis/loader')
import numpy as np, pandas as pd, telemetry_load as T
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = '/home/felipe/Igalia/nanochat/analysis/investigations/0008-adaptive-batching-feasibility/A0001'
SP = '/tmp/claude-1000/-home-felipe-Igalia-nanochat-nanochat/7b7044b9-04da-4203-ac58-6157ccf23646/scratchpad/d12_cont.pkl'
out = pickle.load(open(SP, 'rb'))
runs = sorted(out)
W = 51
L = pd.DataFrame({r: out[r]['loss/train_mean'] for r in runs}).sort_index()
L = L.loc[40:]                       # drop the LR-warmup transient (caveat 3)
R = L - L.rolling(W, center=True, min_periods=W).median()
R = R.dropna()
comp = pd.DataFrame({c: out[runs[0]][c] for c in
                     ['batch/bos_count','batch/mean_segment_length',
                      'batch/segments_per_row_mean','batch/segments_per_row_max']}).loc[R.index]

shared = R.mean(axis=1)                       # batch-attributable component
idio = R.sub(shared, axis=0)                  # seed-idiosyncratic
n = len(runs)
var_tot = R.values.var(ddof=1)
var_shared_raw = shared.var(ddof=1)
var_idio = idio.values.var(ddof=1)*n/(n-1)
var_shared = var_shared_raw - var_idio/n      # unbiased shared-variance estimate
print('=== variance decomposition of the detrended per-step training loss (d12, 5 seeds) ===')
print(f'  steps used                 : {len(R)}  (steps 40..{R.index.max()}, {W}-step centred median detrend)')
print(f'  total residual variance    : {var_tot:.6e}   (sd = {np.sqrt(var_tot):.5f} nats)')
print(f'  BATCH-attributable (shared): {var_shared:.6e}   = {100*var_shared/var_tot:.2f}% of it')
print(f'  seed-idiosyncratic         : {var_idio:.6e}   = {100*var_idio/var_tot:.2f}%')
print(f'  intraclass correlation     : {var_shared/(var_shared+var_idio):.4f}')
print(f'  batch-driven loss sd       : {np.sqrt(max(var_shared,0)):.5f} nats '
      f'({100*np.sqrt(max(var_shared,0))/L.loc[R.index].values.mean():.3f}% of the loss level)')

print('\n=== does recorded batch composition explain the batch-attributable component? ===')
X = comp.copy()
X = (X - X.mean())/X.std()
Xd = np.column_stack([np.ones(len(X))] + [X[c].values for c in X.columns])
beta, *_ = np.linalg.lstsq(Xd, shared.values, rcond=None)
pred = Xd @ beta
r2 = 1 - ((shared.values-pred)**2).sum()/((shared.values-shared.mean())**2).sum()
print(f'  OLS of shared residual on {list(X.columns)}')
print(f'  R2 = {r2:.4f}   (fraction of the batch-attributable variance the recorded')
print(f'       batch descriptors explain = {100*r2*var_tot/var_shared:.1f}% of it)')
for c, b in zip(X.columns, beta[1:]):
    print(f'    corr(shared, {c:32s}) = {np.corrcoef(shared.values, X[c].values)[0,1]:+.4f}   beta={b:+.5f}')

print('\n=== persistence: does the batch at step s move the trajectory AFTER s? ===')
def xcorr(a, b, lags):
    a = (a-a.mean())/a.std(); b = (b-b.mean())/b.std()
    return [float(np.mean(a[:len(a)-k]*b[k:])) if k > 0 else float(np.mean(a*b)) for k in lags]
lags = list(range(0, 21))
ac_shared = xcorr(shared.values, shared.values, lags)
ac_comp = xcorr(comp['batch/bos_count'].values, comp['batch/bos_count'].values, lags)
xc = xcorr(comp['batch/bos_count'].values, shared.values, lags)
print('  lag :', ' '.join(f'{k:6d}' for k in lags[:11]))
print('  acf(shared resid) :', ' '.join(f'{v:+6.3f}' for v in ac_shared[:11]))
print('  acf(bos_count)    :', ' '.join(f'{v:+6.3f}' for v in ac_comp[:11]))
print('  xcorr(bos -> res) :', ' '.join(f'{v:+6.3f}' for v in xc[:11]))
np.save(f'{OUT}/xcorr.npy', np.array([ac_shared, ac_comp, xc]))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(R.index, shared.values, lw=.4, color='#3b6ea5')
ax[0].set_title('batch-attributable detrended loss (5-seed mean, d12)')
ax[0].set_xlabel('step'); ax[0].set_ylabel('nats')
ax[1].plot(lags, ac_shared, 'o-', ms=3, label='acf shared loss residual')
ax[1].plot(lags, ac_comp, 's-', ms=3, label='acf batch/bos_count')
ax[1].plot(lags, xc, '^-', ms=3, label='xcorr bos_count -> residual')
ax[1].axhline(0, color='k', lw=.5); ax[1].set_xlabel('lag (steps)')
ax[1].legend(fontsize=7); ax[1].set_title('no persistence beyond lag 0')
fig.tight_layout(); fig.savefig(f'{OUT}/fig/batch_effect.png', dpi=130)
print('\nwrote fig/batch_effect.png')
