#!/usr/bin/env python
"""Compile all forgetting-strategy comparison results into one summary CSV."""
import pandas as pd
import ast
import os

base = "/work/dn123/MolALKit/rf_dmpnn_comparison/results"


def safe_eval(x):
    if pd.isna(x):
        return []
    try:
        v = ast.literal_eval(x)
        return v if isinstance(v, list) else []
    except Exception:
        return []


def analyze(path, seeds=(0,)):
    rows = []
    for seed in seeds:
        p = path.format(seed=seed)
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        forget_lists = df['id_forget'].apply(safe_eval)
        all_forgotten = [x for lst in forget_lists for x in lst]
        total = len(all_forgotten)
        unique = len(set(all_forgotten)) if total else 0
        vc = pd.Series(all_forgotten).value_counts() if total else pd.Series(dtype=int)
        rows.append(dict(
            seed=seed,
            n_iter=len(df),
            final_mcc=df['mcc_selector'].iloc[-1],
            final_auc=df['roc_auc_selector'].iloc[-1],
            last10_mcc=df['mcc_selector'].tail(10).mean(),
            last10_auc=df['roc_auc_selector'].tail(10).mean(),
            total_forget_events=total,
            unique_forgotten=unique,
            repeat_ratio=total / unique if unique else 0,
            top_repeat=vc.iloc[0] if len(vc) else 0,
        ))
    return rows


configs = [
    # model, strategy_label, score_method, forget_batch_size, forget_size, path_template, seeds
    ("RF", "none", "-", "-", "-", f"{base}/RF/none/seed{{seed}}/al_traj.csv", (0, 1, 2)),
    ("RF", "max", "oob_uncertainty", 1, 40, f"{base}/RF/max_oob_uncertainty/seed{{seed}}/al_traj.csv", (0, 1, 2)),
    ("RF", "min", "oob_uncertainty", 1, 40, f"{base}/RF/min_oob_uncertainty/seed{{seed}}/al_traj.csv", (0, 1, 2)),
    ("RF", "random", "-", 1, 40, f"{base}/RF_random/seed{{seed}}/al_traj.csv", (0,)),
    ("RF", "max", "oob_uncertainty", 1, 415, f"{base}/RF_plateaustart/max_oob_uncertainty/seed{{seed}}/al_traj.csv", (0,)),
    ("RF", "min", "oob_uncertainty", 1, 415, f"{base}/RF_plateaustart/min_oob_uncertainty/seed{{seed}}/al_traj.csv", (0,)),
    ("RF", "max (add1/forget1 @peak, uncorrected)", "oob_uncertainty", 1, 542, f"{base}/RF_addone_forgetone/max_oob_uncertainty/seed{{seed}}/al_traj.csv", (0,)),
    ("RF", "min (add1/forget1 @peak, uncorrected)", "oob_uncertainty", 1, 542, f"{base}/RF_addone_forgetone/min_oob_uncertainty/seed{{seed}}/al_traj.csv", (0,)),
    ("RF", "none (batch=1)", "-", "-", "per-seed peak", f"{base}/RF_noforget_batch1/seed{{seed}}/al_traj.csv", (0, 1, 2)),
    ("RF", "max (add1/forget1 @peak, short)", "oob_uncertainty", 1, "per-seed peak", f"{base}/RF_addone_forgetone_corrected/max_oob_uncertainty/seed{{seed}}/al_traj.csv", (0, 1, 2)),
    ("RF", "min (add1/forget1 @peak, short)", "oob_uncertainty", 1, "per-seed peak", f"{base}/RF_addone_forgetone_corrected/min_oob_uncertainty/seed{{seed}}/al_traj.csv", (0, 1, 2)),
    ("RF", "max (add1/forget1 @peak)", "oob_uncertainty", 1, "per-seed peak", f"{base}/RF_addone_forgetone_matchiter/max_oob_uncertainty/seed{{seed}}/al_traj.csv", (0, 1, 2)),
    ("RF", "min (add1/forget1 @peak)", "oob_uncertainty", 1, "per-seed peak", f"{base}/RF_addone_forgetone_matchiter/min_oob_uncertainty/seed{{seed}}/al_traj.csv", (0, 1, 2)),

    ("DMPNN", "none", "-", "-", "-", f"{base}/DMPNN/none/seed{{seed}}/al_traj.csv", (0, 1, 2)),
    ("DMPNN", "max", "bald", 1, 40, f"{base}/DMPNN_cpu/max/seed{{seed}}/al_traj.csv", (0, 1, 2)),
    ("DMPNN", "min", "bald", 1, 40, f"{base}/DMPNN_cpu/min/seed{{seed}}/al_traj.csv", (0, 1, 2)),
    ("DMPNN", "random", "-", 1, 40, f"{base}/DMPNN_random/seed{{seed}}/al_traj.csv", (0,)),
    ("DMPNN", "max", "entropy", 1, 40, f"{base}/DMPNN_scoremethod/entropy_max/seed{{seed}}/al_traj.csv", (0, 1, 2)),
    ("DMPNN", "min", "entropy", 1, 40, f"{base}/DMPNN_scoremethod/entropy_min/seed{{seed}}/al_traj.csv", (0,)),
    ("DMPNN", "max", "variance", 1, 40, f"{base}/DMPNN_scoremethod/variance_max/seed{{seed}}/al_traj.csv", (0, 1, 2)),
    ("DMPNN", "min", "variance", 1, 40, f"{base}/DMPNN_scoremethod/variance_min/seed{{seed}}/al_traj.csv", (0,)),
    ("DMPNN", "max", "variation_ratio", 1, 40, f"{base}/DMPNN_scoremethod/variation_ratio_max/seed{{seed}}/al_traj.csv", (0,)),
    ("DMPNN", "min", "variation_ratio", 1, 40, f"{base}/DMPNN_scoremethod/variation_ratio_min/seed{{seed}}/al_traj.csv", (0,)),
    ("DMPNN", "max", "bald (mc=50)", 1, 40, f"{base}/DMPNN_mcsensitivity/mc50_dp0.1_max/seed{{seed}}/al_traj.csv", (0,)),
    ("DMPNN", "min", "bald (mc=50)", 1, 40, f"{base}/DMPNN_mcsensitivity/mc50_dp0.1_min/seed{{seed}}/al_traj.csv", (0,)),
    ("DMPNN", "max", "bald (dp=0.3)", 1, 40, f"{base}/DMPNN_mcsensitivity/mc20_dp0.3_max/seed{{seed}}/al_traj.csv", (0,)),
    ("DMPNN", "min", "bald (dp=0.3)", 1, 40, f"{base}/DMPNN_mcsensitivity/mc20_dp0.3_min/seed{{seed}}/al_traj.csv", (0,)),
    ("DMPNN", "max", "bald", 10, 40, f"{base}/DMPNN_forgetbatch10/max/seed{{seed}}/al_traj.csv", (0,)),
    ("DMPNN", "min", "bald", 10, 40, f"{base}/DMPNN_forgetbatch10/min/seed{{seed}}/al_traj.csv", (0,)),
    ("DMPNN", "max", "bald", 20, 40, f"{base}/DMPNN_forgetbatch20/max/seed{{seed}}/al_traj.csv", (0,)),
    ("DMPNN", "min", "bald", 20, 40, f"{base}/DMPNN_forgetbatch20/min/seed{{seed}}/al_traj.csv", (0,)),
    ("DMPNN", "max", "bald", 1, 782, f"{base}/DMPNN_plateaustart/max/seed{{seed}}/al_traj.csv", (0,)),
    ("DMPNN", "min", "bald", 1, 782, f"{base}/DMPNN_plateaustart/min/seed{{seed}}/al_traj.csv", (0,)),
    ("DMPNN", "max", "bald", 10, 782, f"{base}/DMPNN_plateau_batch10/max/seed{{seed}}/al_traj.csv", (0,)),
    ("DMPNN", "min", "bald", 10, 782, f"{base}/DMPNN_plateau_batch10/min/seed{{seed}}/al_traj.csv", (0,)),
]

records = []
for model, strategy, score_method, fbs, fsize, path_t, seeds in configs:
    for row in analyze(path_t, seeds):
        row.update(model=model, strategy=strategy, score_method=score_method,
                   forget_batch_size=fbs, forget_size=fsize)
        records.append(row)

df = pd.DataFrame(records)
cols = ['model', 'strategy', 'score_method', 'forget_batch_size', 'forget_size', 'seed',
        'n_iter', 'final_mcc', 'final_auc', 'last10_mcc', 'last10_auc',
        'total_forget_events', 'unique_forgotten', 'repeat_ratio', 'top_repeat']
df = df[cols]
out_path = "/work/dn123/MolALKit/rf_dmpnn_comparison/summary_results.csv"
df.to_csv(out_path, index=False)
print(f"Wrote {len(df)} rows to {out_path}")
print(df.to_string(index=False))
