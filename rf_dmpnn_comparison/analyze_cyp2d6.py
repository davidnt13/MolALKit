"""CYP2D6_Veith: RF (add-one/forget-one from no-forget MCC peak) and DMPNN (batch 20, forget_size 40), 4 arms each, 3 seeds."""
import glob, ast, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
R = 'results/CYP2D6_Veith/'
def trajs(pat):
    out = []
    for d in sorted(glob.glob(R + pat + '/seed*')):
        df = pd.read_csv(d + '/al_traj.csv', usecols=['n_iter', 'mcc_selector', 'roc_auc_selector'])
        out.append(df)
    return out
arms = {'RF': {'none': 'RF_noforget_batch1', 'random': 'RF_addone_forgetone/forget_random', 'max': 'RF_addone_forgetone/max_oob_uncertainty', 'min': 'RF_addone_forgetone/min_oob_uncertainty'},
        'DMPNN': {'none': 'DMPNN/none', 'random': 'DMPNN/forget_random', 'max': 'DMPNN/max', 'min': 'DMPNN/min'}}
data = {(m, a): trajs(p) for m, d in arms.items() for a, p in d.items()}
def stat(v, base):
    return stats.ttest_ind(v, base, equal_var=False).pvalue if len(v) > 1 and len(base) > 1 else np.nan
rows = []
# RF peaks per seed -> windows
peaks = [int(np.argmax(d['mcc_selector'].values)) for d in data[('RF', 'none')]]
print('RF no-forget peak rows per seed', peaks, 'peak MCC', [round(d['mcc_selector'].max(), 3) for d in data[('RF', 'none')]])
for (m, a), ds in data.items():
    if m == 'RF':
        fin = [d['mcc_selector'].iloc[-1] for d in ds]
        post = [d['mcc_selector'].values[p:].mean() for d, p in zip(ds, peaks)]           # mean over post-peak iterations
        late = [d['mcc_selector'].values[-500:].mean() for d in ds]                         # last 500 iterations
        auc = [d['roc_auc_selector'].iloc[-1] for d in ds]
        rows.append(dict(model=m, arm=a, n=len(ds), final_mcc=np.mean(fin), post_peak_mean=np.mean(post), last500_mean=np.mean(late), final_auc=np.mean(auc),
                         p_final_vs_none=stat(fin, [d['mcc_selector'].iloc[-1] for d in data[('RF', 'none')]]),
                         p_last500_vs_none=stat(late, [d['mcc_selector'].values[-500:].mean() for d in data[('RF', 'none')]])))
    else:
        fin = [d['mcc_selector'].iloc[-1] for d in ds]; l10 = [d['mcc_selector'].values[-10:].mean() for d in ds]
        auc = [d['roc_auc_selector'].iloc[-1] for d in ds]; mx = [d['mcc_selector'].max() for d in ds]
        rows.append(dict(model=m, arm=a, n=len(ds), final_mcc=np.mean(fin), last10_mean=np.mean(l10), max_mcc=np.mean(mx), final_auc=np.mean(auc),
                         p_last10_vs_none=stat(l10, [d['mcc_selector'].values[-10:].mean() for d in data[('DMPNN', 'none')]])))
df = pd.DataFrame(rows); df.to_csv('cyp2d6_veith_summary.csv', index=False)
pd.set_option('display.width', 220); print(df.round(3).to_string(index=False))
# cross-model: best DMPNN vs best RF
d_none = [d['mcc_selector'].values[-10:].mean() for d in data[('DMPNN', 'none')]]
print('DMPNN none last10 per seed', np.round(d_none, 3))
r_pk = [d['mcc_selector'].max() for d in data[('RF', 'none')]]
print('RF none peak MCC per seed', np.round(r_pk, 3), ' Welch p DMPNN last10 vs RF peak: %.3f' % stats.ttest_ind(d_none, r_pk, equal_var=False).pvalue)
col = {'none': '#444', 'random': '#2ca02c', 'max': '#d95f02', 'min': '#1f77b4'}
fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
for a in col:
    ds = data[('RF', a)]; n = min(len(d) for d in ds)
    sm = np.mean([pd.Series(d['mcc_selector'].values[:n]).rolling(50, min_periods=1).mean().values for d in ds], 0)
    axs[0].plot(np.arange(n), sm, color=col[a], label=a, lw=1)
axs[0].set_xscale('log'); axs[0].set_xlabel('iteration (RF, batch 1)'); axs[0].set_ylabel('MCC (50-iter rolling mean, 3 seeds)'); axs[0].set_title('RF, CYP2D6_Veith'); axs[0].legend()
for a in col:
    ds = data[('DMPNN', a)]; m = np.mean([d['mcc_selector'].values[:100] for d in ds], 0)
    axs[1].plot(np.arange(100), m, color=col[a], label=a)
axs[1].set_xlabel('iteration (DMPNN, batch 20)'); axs[1].set_title('DMPNN, CYP2D6_Veith (mean of 3 seeds)'); axs[1].legend()
plt.tight_layout(); plt.savefig('plot12_cyp2d6_veith_curves.png', dpi=150)
