#!/usr/bin/env python
"""Generate summary plots from summary_results.csv."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#888888",
    "axes.labelcolor": "#333333",
    "text.color": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# Fixed categorical colors (colorblind-safe-ish, consistent assignment per entity)
C_NONE = "#6b7280"      # neutral gray - baseline
C_MAX = "#2563eb"       # blue - max direction
C_MIN = "#dc2626"       # red - min direction
C_RANDOM = "#9333ea"    # purple - random control

OUT = "/work/dn123/MolALKit/rf_dmpnn_comparison"
df = pd.read_csv(f"{OUT}/summary_results.csv")


def agg(sub):
    g = sub.groupby(["model", "strategy", "score_method", "forget_batch_size", "forget_size"], dropna=False)
    out = g.agg(mcc_mean=("final_mcc", "mean"), mcc_std=("final_mcc", "std"),
                auc_mean=("final_auc", "mean"), auc_std=("final_auc", "std"),
                n=("final_mcc", "size")).reset_index()
    out["mcc_std"] = out["mcc_std"].fillna(0)
    out["auc_std"] = out["auc_std"].fillna(0)
    return out


agg_df = agg(df)

# ---------------------------------------------------------------
# Figure 1: DMPNN scoring-method sweep, max vs min vs none (batch=1, forget_size=40)
# with Welch's t-test vs no-forget overlaid (only possible where n=3 on both sides)
# ---------------------------------------------------------------
from scipy import stats as _stats

order = [("none", "-"), ("random", "-"), ("max", "bald"), ("min", "bald"),
         ("max", "entropy"), ("min", "entropy"), ("max", "variance"), ("min", "variance"),
         ("max", "variation_ratio"), ("min", "variation_ratio")]
rows = []
raw_vals = []  # per-config list of per-seed final_mcc / final_auc, for the t-test
none_raw = df[(df.model == "DMPNN") & (df.strategy == "none")]
for strat, sm in order:
    r = agg_df[(agg_df.model == "DMPNN") & (agg_df.strategy == strat) & (agg_df.score_method == sm)
               & (agg_df.forget_batch_size.isin([1, "1", "-"]))]
    raw = df[(df.model == "DMPNN") & (df.strategy == strat) & (df.score_method == sm)
             & (df.forget_batch_size.astype(str).isin(["1", "-"]))]
    if strat == "none":
        r = agg_df[(agg_df.model == "DMPNN") & (agg_df.strategy == "none")]
        raw = none_raw
    if len(r):
        rows.append(r.iloc[0])
        raw_vals.append(raw)
d1 = pd.DataFrame(rows)
# wrap long score-method names onto a second line to save horizontal space
sm_wrap = {"variation_ratio": "variation_\nratio"}
labels = [f"{s}\n{sm_wrap.get(sm, sm)}" if sm != "-" else s for s, sm in zip(d1.strategy, d1.score_method)]
colors = [C_NONE if s == "none" else C_RANDOM if s == "random" else C_MAX if s == "max" else C_MIN for s in d1.strategy]

fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
x = np.arange(len(d1)) * 1.15  # extra spacing between bars
for ax, col, errcol, metric_col, title, ylab in [
    (axes[0], "mcc_mean", "mcc_std", "final_mcc", "MCC (primary metric)", "Final MCC"),
    (axes[1], "auc_mean", "auc_std", "final_auc", "ROC-AUC (secondary metric)", "Final ROC-AUC"),
]:
    bars = ax.bar(x, d1[col], yerr=d1[errcol], color=colors, width=0.75,
                  edgecolor="white", linewidth=0.5, capsize=3,
                  error_kw={"ecolor": "#444444", "elinewidth": 1})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=0, ha="center")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=12, color="#111111")
    ax.axhline(d1[d1.strategy == "none"][col].values[0], color=C_NONE, linestyle="--", linewidth=1, alpha=0.6)
    ymax = (d1[col] + d1[errcol]).max()
    ax.set_ylim(0, ymax * 1.35)
    for xi, n, raw, barval, errval in zip(x, d1["n"], raw_vals, d1[col], d1[errcol]):
        ax.text(xi, 0.015, f"n={n}", ha="center", va="bottom", fontsize=7.5, color="#666666")
        # Welch's t-test vs no-forget, only when both groups have n=3
        if n == 3 and len(none_raw) == 3 and not raw[metric_col].equals(none_raw[metric_col]):
            tstat, pval = _stats.ttest_ind(raw[metric_col].values, none_raw[metric_col].values, equal_var=False)
            sig = "*" if pval < 0.05 else "ns"
            ax.text(xi, barval + errval + ymax * 0.03, f"p={pval:.2f}{' ' + sig if sig == '*' else ''}",
                    ha="center", va="bottom", fontsize=7.5, color="#111111" if pval < 0.05 else "#999999")

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, color=C_NONE, label="no forgetting"),
    plt.Rectangle((0, 0), 1, 1, color=C_RANDOM, label="random"),
    plt.Rectangle((0, 0), 1, 1, color=C_MAX, label="max-direction"),
    plt.Rectangle((0, 0), 1, 1, color=C_MIN, label="min-direction"),
]
fig.legend(handles=legend_handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.06))
fig.suptitle("DMPNN: scoring method × target direction\n"
             "dashed line = no-forget baseline; error bars = 3-seed std; p-values = Welch's t-test vs no-forget (only where both sides have n=3)",
             fontsize=10.5, y=1.14, color="#111111")
fig.tight_layout()
fig.savefig(f"{OUT}/plot1_dmpnn_scoremethod.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------
# Figure 2: RF comparison (none/max/min/random/plateau/add1-forget1)
# ---------------------------------------------------------------
rf_order = [("none", "-", "-"), ("random", "-", "40"), ("max", "oob_uncertainty", "40"), ("min", "oob_uncertainty", "40"),
            ("max", "oob_uncertainty", "415"), ("min", "oob_uncertainty", "415"),
            ("max (add1/forget1 @peak)", "oob_uncertainty", "per-seed peak"), ("min (add1/forget1 @peak)", "oob_uncertainty", "per-seed peak")]
rows = []
rf_labels = []
rf_hatch = []
for strat, sm, fs in rf_order:
    sub = agg_df[(agg_df.model == "RF") & (agg_df.strategy == strat) & (agg_df.score_method == sm)]
    sub = sub[sub.forget_size.astype(str) == fs] if strat != "none" else sub
    if len(sub):
        rows.append(sub.iloc[0])
        if fs == "40":
            tag = "@40"
        elif fs == "415":
            tag = "@415\n(~35%)"
        elif fs == "per-seed peak":
            tag = "@peak\n(1-for-1, n_iter=1150)"
        else:
            tag = ""
        base_strat = "max" if strat.startswith("max") else ("min" if strat.startswith("min") else strat)
        rf_labels.append(f"{base_strat}\n{tag}".strip())
        rf_hatch.append("///" if "add1/forget1" in strat else None)
d2 = pd.DataFrame(rows)
colors2 = [C_NONE if s == "none" else C_RANDOM if s == "random" else C_MAX if s.startswith("max") else C_MIN for s in d2.strategy]

fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.5))
x = np.arange(len(d2))
for ax, col, errcol, title, ylab in [
    (axes[0], "mcc_mean", "mcc_std", "MCC (primary metric)", "Final MCC"),
    (axes[1], "auc_mean", "auc_std", "ROC-AUC (secondary metric)", "Final ROC-AUC"),
]:
    bars = ax.bar(x, d2[col], yerr=d2[errcol], color=colors2, width=0.62, edgecolor="white",
                  linewidth=0.5, capsize=3, error_kw={"ecolor": "#444444", "elinewidth": 1})
    for bar, h in zip(bars, rf_hatch):
        if h:
            bar.set_hatch(h)
            bar.set_edgecolor("white")
    ax.set_xticks(x)
    ax.set_xticklabels(rf_labels, fontsize=8.5)
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=12, color="#111111")
    ax.axhline(d2[d2.strategy == "none"][col].values[0], color=C_NONE, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_ylim(0, max(d2[col]) * 1.3)

extra_handle = plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#333333", hatch="///", label="add-1/forget-1 @ peak")
fig.legend(handles=legend_handles + [extra_handle], loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.04), fontsize=9)
fig.suptitle("Random Forest: forgetting strategy barely moves either metric (hatched = today's add-1/forget-1 test)",
             fontsize=11, y=1.12, color="#111111")
fig.tight_layout()
fig.savefig(f"{OUT}/plot2_rf_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------
# Figure 3: forget_batch_size dose-response (DMPNN, bald, forget_size=40)
# ---------------------------------------------------------------
fbs_vals = [1, 10, 20]
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, col, title, ylab in [
    (axes[0], "final_mcc", "MCC vs forget_batch_size", "Final MCC"),
    (axes[1], "final_auc", "ROC-AUC vs forget_batch_size", "Final ROC-AUC"),
]:
    for strat, color in [("max", C_MAX), ("min", C_MIN)]:
        vals = []
        for fbs in fbs_vals:
            sub = df[(df.model == "DMPNN") & (df.strategy == strat) & (df.score_method == "bald")
                     & (df.forget_batch_size.astype(str) == str(fbs)) & (df.forget_size.astype(str) == "40")]
            vals.append(sub[col].mean() if len(sub) else np.nan)
        ax.plot(fbs_vals, vals, marker="o", markersize=7, linewidth=2, color=color, label=strat)
    none_val = df[(df.model == "DMPNN") & (df.strategy == "none")][col].mean()
    ax.axhline(none_val, color=C_NONE, linestyle="--", linewidth=1, label="no forgetting")
    ax.set_xticks(fbs_vals)
    ax.set_xlabel("forget_batch_size (molecules forgotten / iteration)")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=12, color="#111111")
    ax.set_ylim(0, 1)
axes[0].legend(frameon=False, loc="lower left")
fig.suptitle("DMPNN: how much you forget per step dominates targeting choice (catastrophic collapse at batch=20)",
             fontsize=11, y=1.03, color="#111111")
fig.tight_layout()
fig.savefig(f"{OUT}/plot3_batchsize_doseresponse.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------
# Figure 4: diversity / fixation - repeat_ratio and top_repeat for DMPNN batch=1 configs
# ---------------------------------------------------------------
div_order = [("max", "bald"), ("min", "bald"), ("max", "entropy"), ("min", "entropy"),
             ("max", "variance"), ("min", "variance"), ("max", "bald (mc=50)"), ("min", "bald (mc=50)")]
rows = []
div_labels = []
for strat, sm in div_order:
    sub = df[(df.model == "DMPNN") & (df.strategy == strat) & (df.score_method == sm)
             & (df.forget_batch_size.astype(str) == "1") & (df.forget_size.astype(str) == "40") & (df.seed == 0)]
    if len(sub):
        rows.append(sub.iloc[0])
        div_labels.append(f"{strat}\n{sm}")
d4 = pd.DataFrame(rows)
colors4 = [C_MAX if s == "max" else C_MIN for s in d4.strategy]

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(d4))
ax.bar(x, d4["top_repeat"], color=colors4, width=0.6, edgecolor="white", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(div_labels, fontsize=9)
ax.set_ylabel("Times the single most-repeated\nmolecule was forgotten (of 99 events)")
ax.set_title("The 'min' fixation pathology: same molecule forgotten 40+ times, regardless of mc_samples/dropout",
             fontsize=11, color="#111111")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_MAX, label="max"),
                    plt.Rectangle((0, 0), 1, 1, color=C_MIN, label="min")], frameon=False)
fig.tight_layout()
fig.savefig(f"{OUT}/plot4_fixation_pathology.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------
# Figure 5: RF add-one/forget-one trajectory - "stop at peak" vs "keep churning"
# ---------------------------------------------------------------
import ast as _ast


def load_traj(path):
    d = pd.read_csv(path)
    d["train_size"] = (d["id_prior_al"].apply(lambda x: len(_ast.literal_eval(x)))
                        + d["id_add"].apply(lambda x: len(_ast.literal_eval(x))))
    return d


PEAK_SEED0 = 572  # RF's own no-forget curve (batch_size=1) MCC-peak train_size, seed0
none_traj = load_traj(f"{OUT}/results/RF_noforget_batch1/seed0/al_traj.csv")
max_traj = load_traj(f"{OUT}/results/RF_addone_forgetone_matchiter/max_oob_uncertainty/seed0/al_traj.csv")
min_traj = load_traj(f"{OUT}/results/RF_addone_forgetone_matchiter/min_oob_uncertainty/seed0/al_traj.csv")

# only the post-peak churning phase (train_size held at PEAK_SEED0 via
# 1-for-1 exchange) is the new information here; each run's own ramp-up
# (at batch_size=1, a separate run from the no-forget curve) is excluded
# from the right panel so it doesn't duplicate the no-forget curve's own ramp.
max_post = max_traj[max_traj["train_size"] >= PEAK_SEED0].reset_index(drop=True)
min_post = min_traj[min_traj["train_size"] >= PEAK_SEED0].reset_index(drop=True)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

axL.plot(none_traj["train_size"], none_traj["mcc_selector"], color=C_NONE, linewidth=1.8)
axL.axvline(PEAK_SEED0, color="#333333", linestyle=":", linewidth=1.3)
axL.text(PEAK_SEED0 + 10, axL.get_ylim()[0] + 0.02, f"peak\n(train_size={PEAK_SEED0})", fontsize=8.5, color="#333333", va="bottom")
axL.set_xlabel("Training set size")
axL.set_ylabel("MCC")
axL.set_title("No-forget curve (batch_size=1): where is the peak?", fontsize=11, color="#111111")

axR.plot(np.arange(len(max_post)), max_post["mcc_selector"], color=C_MAX, linewidth=1.3, alpha=0.9, label="max")
axR.plot(np.arange(len(min_post)), min_post["mcc_selector"], color=C_MIN, linewidth=1.3, alpha=0.9, label="min")
start_val = max_post["mcc_selector"].iloc[0]
axR.axhline(start_val, color="#333333", linestyle=":", linewidth=1.2, label=f"MCC at train_size={PEAK_SEED0} ({start_val:.2f})")
axR.set_xlabel(f"Exchange cycles after reaching peak\n(training set held at {PEAK_SEED0} via 1-for-1 swap, ~{len(max_post)} cycles)")
axR.set_ylabel("MCC")
axR.set_title("After the peak: churning drifts down, doesn't improve", fontsize=11, color="#111111")
axR.legend(frameon=False, loc="lower left", fontsize=9)

fig.suptitle("RF add-one/forget-one (seed0, matched to no-forget's full 1150-iteration length):\nstopping at the peak beats continuing to churn",
             fontsize=11.5, y=1.05, color="#111111")
fig.tight_layout()
fig.savefig(f"{OUT}/plot5_rf_addone_trajectory.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------
# Figure 6: full curves - no-forget vs max vs min, entire trajectory, vs iteration
# ---------------------------------------------------------------
forget_start_iter = int(max_traj[max_traj["train_size"] >= PEAK_SEED0]["n_iter"].iloc[0])
fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6))
for ax, metric_col, title, ylab in [
    (axL, "mcc_selector", "MCC", "MCC"),
    (axR, "roc_auc_selector", "ROC-AUC", "ROC-AUC"),
]:
    ax.plot(none_traj["n_iter"], none_traj[metric_col], color=C_NONE, linewidth=2,
            label="no-forget (batch_size=1)", zorder=3)
    ax.plot(max_traj["n_iter"], max_traj[metric_col], color=C_MAX, linewidth=1, alpha=0.8,
            label="max (add-1/forget-1)", zorder=2)
    ax.plot(min_traj["n_iter"], min_traj[metric_col], color=C_MIN, linewidth=1, alpha=0.8,
            label="min (add-1/forget-1)", zorder=1)
    ax.axvline(forget_start_iter, color="#333333", linestyle=":", linewidth=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=12, color="#111111")
axL.text(forget_start_iter + 10, 0.47, f"forgetting starts\n(train_size={PEAK_SEED0})", fontsize=8, color="#333333", va="top")
axL.legend(frameon=False, loc="upper left", fontsize=9)
fig.suptitle(f"Full curves vs iteration, seed0 (all batch_size=1, same max_iter=1150): no-forget vs max/min "
             f"(ramp to train_size={PEAK_SEED0} at iter {forget_start_iter}, then 1-for-1 exchange)",
             fontsize=10.5, y=1.05, color="#111111")
fig.tight_layout()
fig.savefig(f"{OUT}/plot6_full_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------
# Figure 8: full trajectories, 2 subplots (RF / DMPNN), shared y-axis, all seeds shown
# ---------------------------------------------------------------
def load_full(path):
    d = pd.read_csv(path)
    d["train_size"] = (d["id_prior_al"].apply(lambda x: len(_ast.literal_eval(x)))
                        + d["id_add"].apply(lambda x: len(_ast.literal_eval(x))))
    return d


rf_none_trajs = [load_full(f"{OUT}/results/RF_noforget_batch1/seed{s}/al_traj.csv") for s in [0, 1, 2]]
rf_max_trajs = [load_full(f"{OUT}/results/RF_addone_forgetone_matchiter/max_oob_uncertainty/seed{s}/al_traj.csv") for s in [0, 1, 2]]
rf_min_trajs = [load_full(f"{OUT}/results/RF_addone_forgetone_matchiter/min_oob_uncertainty/seed{s}/al_traj.csv") for s in [0, 1, 2]]

dmpnn_none_trajs = [load_full(f"{OUT}/results/DMPNN/none/seed{s}/al_traj.csv") for s in [0, 1, 2]]
dmpnn_varmax_trajs = [load_full(f"{OUT}/results/DMPNN_scoremethod/variance_max/seed{s}/al_traj.csv") for s in [0, 1, 2]]
dmpnn_baldmin_trajs = [load_full(f"{OUT}/results/DMPNN_cpu/min/seed{s}/al_traj.csv") for s in [0, 1, 2]]

all_mcc = (pd.concat([t["mcc_selector"] for t in rf_none_trajs + rf_max_trajs + rf_min_trajs
                       + dmpnn_none_trajs + dmpnn_varmax_trajs + dmpnn_baldmin_trajs]))
ylim = (all_mcc.min() - 0.03, all_mcc.max() + 0.03)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)

for trajs, color, label in [(rf_none_trajs, C_NONE, "no-forget"), (rf_max_trajs, C_MAX, "max (add-1/forget-1 @ peak)"),
                             (rf_min_trajs, C_MIN, "min (add-1/forget-1 @ peak)")]:
    for i, t in enumerate(trajs):
        axL.plot(t["n_iter"], t["mcc_selector"], color=color, linewidth=0.9, alpha=0.75,
                 label=label if i == 0 else None)
axL.set_xlabel("Iteration")
axL.set_ylabel("MCC")
axL.set_title("Random Forest (3 seeds each)", fontsize=12, color="#111111")
axL.legend(frameon=False, loc="lower right", fontsize=9)
axL.set_ylim(*ylim)

for trajs, color, label in [(dmpnn_none_trajs, C_NONE, "no-forget"), (dmpnn_varmax_trajs, C_MAX, "best max (variance)"),
                             (dmpnn_baldmin_trajs, C_MIN, "best min (bald)")]:
    for i, t in enumerate(trajs):
        axR.plot(t["n_iter"], t["mcc_selector"], color=color, linewidth=0.9, alpha=0.75,
                 label=label if i == 0 else None)
axR.set_xlabel("Iteration")
axR.set_title("DMPNN (3 seeds each)", fontsize=12, color="#111111")
axR.legend(frameon=False, loc="lower right", fontsize=9)
axR.set_ylim(*ylim)

fig.suptitle("Full trajectories, all 3 seeds overlaid, shared y-axis", fontsize=12, y=1.02, color="#111111")
fig.tight_layout()
fig.savefig(f"{OUT}/plot8_full_trajectories.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------
# Figure 7 (REDONE): best-of comparison with proper n=3 stats for both models
# ---------------------------------------------------------------
rf_none_raw = df[(df.model == "RF") & (df.strategy == "none (batch=1)")]
rf_max_raw = df[(df.model == "RF") & (df.strategy == "max (add1/forget1 @peak)")]
rf_min_raw = df[(df.model == "RF") & (df.strategy == "min (add1/forget1 @peak)")]
dmpnn_none_raw = df[(df.model == "DMPNN") & (df.strategy == "none")]
dmpnn_varmax_raw = df[(df.model == "DMPNN") & (df.strategy == "max") & (df.score_method == "variance")
                       & (df.forget_batch_size.astype(str) == "1") & (df.forget_size.astype(str) == "40")]
dmpnn_baldmin_raw = df[(df.model == "DMPNN") & (df.strategy == "min") & (df.score_method == "bald")
                        & (df.forget_batch_size.astype(str) == "1") & (df.forget_size.astype(str) == "40")]

groups7 = [
    ("RF\nno-forget\n(batch=1)", rf_none_raw, C_NONE, None),
    ("RF\nbest: max\n(add-1/forget-1)", rf_max_raw, C_MAX, rf_none_raw),
    ("RF\nbest: min\n(add-1/forget-1)", rf_min_raw, C_MIN, rf_none_raw),
    ("DMPNN\nno-forget", dmpnn_none_raw, C_NONE, None),
    ("DMPNN\nbest max\n(variance)", dmpnn_varmax_raw, C_MAX, dmpnn_none_raw),
    ("DMPNN\nbest min\n(bald)", dmpnn_baldmin_raw, C_MIN, dmpnn_none_raw),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
x = np.arange(len(groups7))
for ax, metric_col, title, ylab in [
    (axes[0], "final_mcc", "MCC (primary metric)", "Final MCC"),
    (axes[1], "final_auc", "ROC-AUC (secondary metric)", "Final ROC-AUC"),
]:
    means = [g[1][metric_col].mean() for g in groups7]
    stds = [g[1][metric_col].std() for g in groups7]
    colors_ = [g[2] for g in groups7]
    ax.bar(x, means, yerr=stds, color=colors_, width=0.6, edgecolor="white", linewidth=0.5,
           capsize=3, error_kw={"ecolor": "#444444", "elinewidth": 1})
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups7], fontsize=8.5)
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=12, color="#111111")
    ymax = max(m + s for m, s in zip(means, stds))
    ax.set_ylim(0, ymax * 1.3)
    for xi, (label, raw, color_, baseline) in zip(x, groups7):
        n = len(raw)
        ax.text(xi, 0.015, f"n={n}", ha="center", va="bottom", fontsize=7.5, color="#666666")
        if baseline is not None and len(raw) == 3 and len(baseline) == 3:
            tstat, pval = _stats.ttest_ind(raw[metric_col].values, baseline[metric_col].values, equal_var=False)
            sig = " *" if pval < 0.05 else ""
            m = raw[metric_col].mean()
            s = raw[metric_col].std()
            ax.text(xi, m + s + ymax * 0.03, f"p={pval:.3f}{sig}", ha="center", va="bottom",
                    fontsize=8, color="#111111" if pval < 0.05 else "#999999")

fig.suptitle("Best-of comparison (n=3 throughout): p-values = Welch's t-test vs each model's own no-forget baseline",
             fontsize=11, y=1.03, color="#111111")
fig.tight_layout()
fig.savefig(f"{OUT}/plot7_best_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("Saved: plot1_dmpnn_scoremethod.png, plot2_rf_comparison.png, plot3_batchsize_doseresponse.png, "
      "plot4_fixation_pathology.png, plot5_rf_addone_trajectory.png, plot6_full_curves.png, "
      "plot7_best_comparison.png (redone with n=3 stats), plot8_full_trajectories.png")
