"""Speedup comparison: baseline (30ep scratch) vs warm10 vs warm10_bs200, x {no-forget, variance max, variance min}."""
import glob, ast, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
R = 'results/'
def paths(v, f):
    if v == 'baseline':
        return glob.glob(R + ('DMPNN/none' if f == 'none' else f'DMPNN_scoremethod/variance_{f}') + '/seed*')
    return glob.glob(R + (f'DMPNN_speedup/{v}' if f == 'none' else f'DMPNN_speedup_forget/{v}/variance_{f}') + '/seed*')
def load(d):
    try: df = pd.read_csv(d + '/al_traj.csv')
    except Exception: return None
    df['train_size'] = df['id_prior_al'].apply(lambda x: len(ast.literal_eval(x))) + df['id_add'].apply(lambda x: len(ast.literal_eval(x)))
    return df
V = ['baseline', 'warm10', 'warm10_bs200']; F = ['none', 'max', 'min']
col = {'baseline': '#7f7f7f', 'warm10': '#1f77b4', 'warm10_bs200': '#d95f02'}
data = {(v, f): [d for d in map(load, sorted(paths(v, f))) if d is not None] for v in V for f in F}
last = {k: [d['mcc_selector'].iloc[-1] for d in ds] for k, ds in data.items()}
# Fig A: final MCC bars with seeds + p-value vs same-forget baseline
fig, ax = plt.subplots(figsize=(9, 4.5)); w = 0.26
for i, v in enumerate(V):
    for j, f in enumerate(F):
        x = j + (i - 1) * w; y = last[(v, f)]
        if not y: continue
        ax.bar(x, np.mean(y), w * 0.9, color=col[v], label=v if j == 0 else None)
        ax.scatter([x] * len(y), y, c='k', s=12, zorder=3)
        if v != 'baseline' and len(y) > 1 and len(last[('baseline', f)]) > 1:
            p = stats.ttest_ind(y, last[('baseline', f)], equal_var=False).pvalue
            ax.text(x, max(y) + 0.015, f'p={p:.2f}', ha='center', fontsize=7)
        ax.text(x, 0.01, f'n={len(y)}', ha='center', fontsize=7, color='w')
ax.set_xticks(range(3)); ax.set_xticklabels(['no forget', 'variance, max', 'variance, min'])
ax.set_ylabel('Final MCC'); ax.set_ylim(0, 0.7); ax.legend(); ax.set_title('DMPNN speedups: final MCC (Welch p vs baseline, same forgetting)')
plt.tight_layout(); plt.savefig('plot9_speedup_final_mcc.png', dpi=150); plt.close()
# Fig B: curves
fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, f in zip(axs, F):
    for v in V:
        ds = data[(v, f)]
        if not ds: continue
        n = min(len(d) for d in ds)
        m = np.mean([d['mcc_selector'].iloc[:n].values for d in ds], 0); x = ds[0]['train_size'].iloc[:n]
        ax.plot(x, m, color=col[v], label=f'{v} (n={len(ds)})')
    ax.set_title(f); ax.set_xlabel('train size')
axs[0].set_ylabel('MCC (mean over seeds)'); axs[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig('plot10_speedup_curves.png', dpi=150); plt.close()
for k, y in last.items(): print(k, np.round(y, 3), 'mean %.3f' % np.mean(y) if y else '')
