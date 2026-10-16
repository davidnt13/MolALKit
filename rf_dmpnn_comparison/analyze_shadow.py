"""Warm-start 'memory' diagnostic: warm model vs cold (from-scratch) models trained on the identical training set.
cold_a/cold_b differ only in torch seed => cold-vs-cold gives the noise floor; warm-vs-cold above that floor = memory of history."""
import glob, re, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import matthews_corrcoef, roc_auc_score
rows = []
for f in sorted(glob.glob('results/DMPNN_shadow/*/*/seed*/shadow/iter_*.npz')):
    m = re.search(r'DMPNN_shadow/([^/]+)/([^/]+)/seed(\d+)/shadow/iter_(\d+)', f); var, mode, seed, it = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
    d = np.load(f); y = d['y_val']; w, a, b = d['warm_val'], d['cold_a_val'], d['cold_b_val']
    lab = lambda p: (p > 0.5).astype(int)
    r = dict(variant=var, mode=mode, seed=seed, iter=it, n_train=int(d['n_train']),
             rho_warm_cold=np.mean([spearmanr(w, a)[0], spearmanr(w, b)[0]]), rho_cold_cold=spearmanr(a, b)[0],
             disagree_warm_cold=np.mean([(lab(w) != lab(a)).mean(), (lab(w) != lab(b)).mean()]), disagree_cold_cold=(lab(a) != lab(b)).mean(),
             mcc_warm=matthews_corrcoef(y, lab(w)), mcc_cold=np.mean([matthews_corrcoef(y, lab(a)), matthews_corrcoef(y, lab(b))]),
             auc_warm=roc_auc_score(y, w), auc_cold=np.mean([roc_auc_score(y, a), roc_auc_score(y, b)]), n_forgot=len(d['forgot_ids']))
    if r['n_forgot'] > 0:  # memory of forgotten molecules: how well each model still fits them (accuracy against their true labels)
        yf = d['y_forgot']; acc = lambda p: (lab(p) == yf).mean()
        r['forgot_acc_warm'] = acc(d['warm_forgot']); r['forgot_acc_cold'] = np.mean([acc(d['cold_a_forgot']), acc(d['cold_b_forgot'])])
        r['forgot_conf_warm'] = np.abs(d['warm_forgot'] - 0.5).mean() * 2; r['forgot_conf_cold'] = np.mean([np.abs(d['cold_a_forgot'] - 0.5).mean(), np.abs(d['cold_b_forgot'] - 0.5).mean()]) * 2
    rows.append(r)
df = pd.DataFrame(rows); df.to_csv('shadow_comparison.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(df.groupby(['variant', 'mode', 'iter']).mean(numeric_only=True).drop(columns='seed').round(3).to_string())
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for (var, mode), g in df.groupby(['variant', 'mode']):
    g = g.groupby('iter').mean(numeric_only=True); lbl = f'{var}/{mode}'
    axs[0].plot(g.index, g.disagree_warm_cold, marker='o', label=lbl)
    axs[1].plot(g.index, g.mcc_warm - g.mcc_cold, marker='o', label=lbl)
    if 'forgot_acc_warm' in g: axs[2].plot(g.index, (g.forgot_acc_warm - g.forgot_acc_cold), marker='o', label=lbl)
cc = df.groupby('iter').disagree_cold_cold.mean(); axs[0].plot(cc.index, cc.values, 'k--', label='cold vs cold (noise floor)')
axs[0].set_title('Label disagreement on val set'); axs[1].set_title('MCC: warm - cold (same train set)'); axs[2].set_title('Forgotten molecules: accuracy warm - cold')
for ax in axs: ax.set_xlabel('iteration'); ax.axhline(0, c='gray', lw=.5)
axs[0].legend(fontsize=7); plt.tight_layout(); plt.savefig('plot11_warm_vs_cold_shadow.png', dpi=150)
