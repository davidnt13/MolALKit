"""Which molecules does each setup retain/forget? Compare against the from-scratch baseline (same seed => same init split)."""
import ast, collections, glob, os, numpy as np, pandas as pd
R = 'results/'
def load(d):
    try: df = pd.read_csv(d + '/al_traj.csv')
    except Exception: return None
    L = lambda c: df[c].apply(ast.literal_eval)
    prior, add, forget = L('id_prior_al'), L('id_add'), L('id_forget')
    final = (set(prior.iloc[-1]) | set(add.iloc[-1])) - set(forget.iloc[-1])
    fl = [i for f in forget for i in f]; al = [i for a in add for i in a]
    return dict(final=final, forgets=fl, adds=al)
def path(v, f, s):
    if v == 'baseline': return f'{R}DMPNN_scoremethod/variance_{f}/seed{s}'
    base = 'DMPNN_restart' if v == 'restart10' else 'DMPNN_speedup_forget'
    return f'{R}{base}/{v}/variance_{f}/seed{s}'
jac = lambda a, b: len(a & b) / max(len(a | b), 1)
rows = []
for f in ['max', 'min']:
    for v in ['baseline', 'warm10', 'warm10_bs200', 'restart10']:
        for s in range(3):
            r = load(path(v, f, s)); b = load(path(f'baseline', f, s))
            if r is None or not os.path.exists(path(v, f, s) + '/al_traj.csv'): continue
            fc = collections.Counter(r['forgets']); top = fc.most_common(1)[0] if fc else (None, 0)
            row = dict(target=f, variant=v, seed=s, n_retained=len(r['final']), n_forget_events=len(r['forgets']),
                       n_unique_forgotten=len(fc), top_forgotten_id=top[0], top_forgotten_count=top[1])
            if b is not None and v != 'baseline':
                row['jaccard_retained_vs_baseline'] = jac(r['final'], b['final'])
                row['jaccard_acquired_vs_baseline'] = jac(set(r['adds']), set(b['adds']))
                row['jaccard_forgotten_vs_baseline'] = jac(set(r['forgets']), set(b['forgets']))
                row['same_top_forgotten_as_baseline'] = top[0] == (collections.Counter(b['forgets']).most_common(1) or [(None,)])[0][0]
            rows.append(row)
df = pd.DataFrame(rows); df.to_csv('molecule_comparison.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
print(df.round(3).to_string(index=False))
print(df.groupby(['target', 'variant']).mean(numeric_only=True).round(3).to_string())
