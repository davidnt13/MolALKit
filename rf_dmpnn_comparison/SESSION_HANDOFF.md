# MolALKit Forgetting-Strategy Investigation — Session Handoff

**Project:** `/work/dn123/MolALKit` (active learning for molecular property prediction)
**Dataset:** MDR1_MDCK_classification2 (2217 molecules, binary classification, ~65/35 class split)
**Working dir for this investigation:** `/work/dn123/MolALKit/rf_dmpnn_comparison/`
**Cluster note:** this session runs directly inside a Slurm interactive allocation (not via `ssh dcc-agent` — that alias doesn't resolve from within an already-provisioned DCC session; running Bash commands directly is correct here). Jobs are submitted to the `common` partition via `sbatch`; `gpu-common`/`scavenger-gpu` were tried for DMPNN but scheduling was too unreliable (hours-to-days queue waits) — CPU on `common` ended up being the reliable path even for DMPNN.

**Status as of this handoff: no jobs pending.** All analysis below reflects the final, corrected data.

## Conda env
```bash
source $(conda info --base)/etc/profile.d/conda.sh && conda activate /hpc/group/rekerlab/dn123/conda/envs/molalkit
```

## Bugs found and fixed in the live repo (`/work/dn123/MolALKit`, uncommitted)

1. **MC-dropout was a no-op** — `molalkit/al/forgetter.py` `ChempropDropoutForgetter._mc_probs` manually activated dropout then called chemprop's `predict()`, which calls `model.eval()` first and silently undid it. Fixed to pass `dropout_prob=` into `predict()` instead (chemprop's built-in reactivation path).
2. **`--cp_model_epochs` silently ignored for DMPNN** — `molalkit/args.py` compared the wrong dict key (`model_config_selector_dict.get('model')`, which doesn't exist in DMPNN configs) instead of checking `data_format == 'chemprop'`. DMPNN always trained 50 epochs regardless of the flag before this fix.
3. **GPU device-mismatch crash** — `ChempropDropoutForgetter.__call__` forced the already-GPU-trained model back to `cpu` (`device` param defaulted to `"cpu"`, never tied to the model's actual device) before MC-scoring, causing `RuntimeError` on every forgetting-enabled GPU run. Fixed by removing the incorrect `.to(self.device)` call — the model is already on the right device from training.
4. **Forgetting was hardcoded to 1 molecule/iteration** — `forgetter.py`'s `TEMP_BATCH_SIZE=1` module constant (now removed) plus `args.py`'s `forgeter.batch_size = 1` (now `self.forget_batch_size`) meant `--forget_size` only controlled *when* forgetting started, never *how many* got forgotten per step. Added new CLI flag `--forget_batch_size` (default 1, preserves old behavior).
5. **`score_method`/`mc_samples`/`dropout_prob` never exposed via CLI** — always defaulted to `bald`/20/0.1. Added `--forget_score_method`, `--forget_mc_samples`, `--forget_dropout_prob`.

All new flags default to the old hardcoded values, so existing invocations are unaffected.

## Key scientific findings (final, as of this handoff)

- **RF: forgetting strategy (which molecules, which uncertainty metric) basically doesn't matter** across every targeted/random/batch-size/start-point variant tried at the *original* fixed-threshold design (`forget_size=40`, `batch_size=20`). This is a solid, repeatedly-confirmed null result.
- **The one place forgetting helps RF is "add-one/forget-one starting at the no-forget curve's own MCC peak"** (`batch_size=1`, `forget_batch_size=1`, `forget_size=<peak train_size>`) — but the strength of this effect **depends heavily on how long you let the exchange run**:
  - Short version (peak + ~110 more iterations, n=3 seeds): **both directions significant** — max p=0.023, min p=0.015.
  - **Corrected/final version** (exchange continues all the way to `max_iter=1150`, matching the no-forget run's exact length, n=3 seeds): **max is no longer significant** (p=0.383), **min remains significant but weaker** (p=0.038). Longer churning erodes most of the gain — this is the trajectory-level "stopping at the peak beats continuing to churn" finding showing up quantitatively. **This matched-iteration version is the one to trust/cite.**
- **DMPNN**: no forgetting config (bald/entropy/variance/variation_ratio, either direction) reliably beats no-forgetting with real seed-level statistics — `variance`+max direction has the best point estimate (MCC 0.516 vs no-forget's 0.493) but it's not significant (p=0.643, n=3).
- **`min`-direction forgetting for DMPNN reliably fixates on a single molecule** (40+ of 99 forget events hit the same molecule), confirmed to be a real, stable phenomenon — NOT a sampling artifact (ruled out by varying `mc_samples` 20→50 and `dropout_prob` 0.1→0.3; the fixation persisted identically).
- **`forget_batch_size` dominates everything else for DMPNN**: at `batch_size=1` only ~8% of acquisitions get discarded net; at `batch_size=20` it's 99%, and the model's net retained training set collapses to just 42 molecules — directly causing the catastrophic AUC collapse (0.49-0.58) seen at that setting. This matters far more than which scoring method or target direction is used.
- **DMPNN significantly outperforms RF** on both models' no-forget baselines (MCC 0.493 vs 0.336, p=0.043) and on each model's own best config (0.516 vs 0.397-0.429 depending on short/matched-iter RF version, p≈0.03). RF's best result still doesn't statistically close the gap with DMPNN's plain baseline.
- **Default scoring metric before any fixes was `bald`** (hardcoded, unreachable via CLI). `variance` and `entropy` scoring are real untested alternatives that turned out to be worth trying.
- **MCC and AUC disagree on several configs** — MCC is the agreed primary metric for this investigation (better suited to the class-imbalanced binary classification task and more sensitive to real failure modes, e.g. it caught the plateau-start-min fixation pathology that AUC nearly missed).
- **DMPNN is 50-700× slower per iteration than RF**, and gets worse the longer a run goes (chemprop retrains 30 backprop epochs from scratch every iteration, cost scales with current training-set size; RF just refits a tree ensemble, cheap regardless of size). Concretely: RF reaches ~1100 molecules of training data (1150 iterations at `batch_size=1`) in under 15 minutes; DMPNN reaches a similar final dataset size (100 iterations at `batch_size=20`) in up to 10+ hours. This asymmetry is why RF could be swept across seeds/batch-sizes/start-points cheaply while every DMPNN experiment was a multi-hour (or, with GPU queue waits, multi-day) commitment.

## Deliverables (all in `rf_dmpnn_comparison/`, all current/final)

- `summary_results.csv` — compiled results table
- `compile_summary.py` — regenerates the CSV from raw `al_traj.csv` files
- `make_plots.py` — generates all plots below from the CSV (run `compile_summary.py` first)
- `plot1_dmpnn_scoremethod.png` — DMPNN scoring method × direction, with Welch's t-test p-values overlaid (n=3 where available)
- `plot2_rf_comparison.png` — RF across all tested configs (fixed: was pointing at a stale `forget_size` filter, now correctly shows the add-1/forget-1 bars)
- `plot3_batchsize_doseresponse.png` — the forget_batch_size collapse (1→10→20) for DMPNN
- `plot4_fixation_pathology.png` — the min-direction single-molecule fixation across scoring methods
- `plot5_rf_addone_trajectory.png` — RF peak-vs-churn, two-panel (fixed: was comparing a `batch_size=20` no-forget curve against `batch_size=1` forgetting runs — now both sides correctly use `batch_size=1`, matched-iteration data, seed0, `PEAK_SEED0=572`)
- `plot6_full_curves.png` — full no-forget vs max vs min curves vs iteration (RF, seed0, now matched-iteration/corrected data)
- `plot7_best_comparison.png` — **the current best 4-way comparison with real n=3 stats** (RF no-forget / RF best / DMPNN no-forget / DMPNN best), using the matched-iteration RF data (note: a duplicate stale "Figure 7" block that silently got overwritten by the correct one was found and deleted from `make_plots.py` during cleanup — if you see two blocks writing `plot7_best_comparison.png` again, that's a regression)
- `SESSION_HANDOFF.md` — this file

### Result-directory map (for anything not in the CSV)
- `RF/none|max_oob_uncertainty|min_oob_uncertainty/seed{0,1,2}` — original fixed-threshold RF comparison (`forget_size=40`, `batch_size=20`)
- `RF_noforget_batch1/seed{0,1,2}` — RF no-forget at `batch_size=1`, `max_iter=1150` (the correct baseline for peak-finding)
- `RF_addone_forgetone_matchiter/{max,min}_oob_uncertainty/seed{0,1,2}` — **the final/correct** add-1/forget-1 RF runs, `max_iter=1150` matching no-forget exactly. Per-seed `forget_size` (each seed's own MCC peak from `RF_noforget_batch1`): seed0=572, seed1=498, seed2=626.
- `RF_addone_forgetone_corrected/` and `RF_addone_forgetone/` — superseded (shorter `max_iter`, or wrong/uncorrected peak point) — kept for reference, not used by current plots except as historical comparison
- `DMPNN/none/seed{0,1,2}` — DMPNN no-forget baseline (`batch_size=20`)
- `DMPNN_cpu/{max,min}/seed{0,1,2}` — DMPNN bald scoring, `forget_size=40`, `batch_size=20`, `forget_batch_size=1` (default)
- `DMPNN_scoremethod/{method}_{max,min}/seed{N}` — entropy/variance/variation_ratio sweep (variance_max and entropy_max have n=3; everything else n=1)
- `DMPNN_forgetbatch10/`, `DMPNN_forgetbatch20/` — the forget_batch_size dose-response experiment
- `DMPNN_mcsensitivity/` — mc_samples=50 and dropout_prob=0.3 variants (fixation-robustness check)
- `DMPNN_plateaustart/`, `DMPNN_plateau_batch10/` — DMPNN start-point experiments (less central to current conclusions)

## Useful one-liners

Check any job: `sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed -X`

Find a no-forget curve's peak MCC point:
```python
import pandas as pd, ast
df = pd.read_csv(path)
df['train_size'] = df['id_prior_al'].apply(lambda x: len(ast.literal_eval(x))) + df['id_add'].apply(lambda x: len(ast.literal_eval(x)))
idx = df['mcc_selector'].idxmax()
# df.loc[idx, 'train_size'] is the peak point
```

Regenerate everything: `python3 compile_summary.py && python3 make_plots.py` (from `rf_dmpnn_comparison/`)

## Natural next steps (not started, just ideas)
- Push DMPNN's `variance-max`/`bald-min` confirmations to a 4th+ seed to see if significance ever emerges (currently n=3, both non-significant).
- Try the `variance` scoring method at higher `mc_samples` for DMPNN (mirrors the RF finding that longer/more-careful runs change the answer).
- Formally test whether the DMPNN `min`-fixation molecule is chemically/structurally unusual (e.g. a scaffold outlier) — would explain *why* the model repeatedly lands on it.
- Consider committing the 5 code fixes in `molalkit/` upstream (currently uncommitted in the live repo) if this is meant to be a lasting contribution back to the codebase.
