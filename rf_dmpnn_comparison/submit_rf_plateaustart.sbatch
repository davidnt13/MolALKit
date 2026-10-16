#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --partition=common
#SBATCH -o /work/dn123/MolALKit/rf_dmpnn_comparison/logs/rf_plateau_%A_%a.out
#SBATCH -e /work/dn123/MolALKit/rf_dmpnn_comparison/logs/rf_plateau_%A_%a.err
#SBATCH --job-name=rf_dmpnn_comp_RF_plateau
#SBATCH --array=0-1

module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /hpc/group/rekerlab/dn123/conda/envs/molalkit

REPO=/work/dn123/MolALKit
BASE=/work/dn123/MolALKit/rf_dmpnn_comparison

# array index 0 = max_oob_uncertainty, 1 = min_oob_uncertainty; seed fixed at 0.
# forget_size=415: the training-set size at which RF's no-forget baseline
# (results/RF/none/seed{0,1,2}) reaches within 3% of its near-final smoothed
# ROC-AUC (averaged across 3 seeds: 482, 382, 382). Same plateau-start idea
# as the DMPNN companion run, using RF's own (much earlier) plateau point.
STRATEGIES=(max_oob_uncertainty min_oob_uncertainty)
STRATEGY=${STRATEGIES[$SLURM_ARRAY_TASK_ID]}
SEED=0

SAVE_DIR="$BASE/results/RF_plateaustart/${STRATEGY}/seed${SEED}"
mkdir -p "$SAVE_DIR"

COMMON_ARGS="--data_public MDR1_MDCK_classification2 \
  --metrics roc_auc mcc accuracy precision recall f1_score \
  --learning_type explorative \
  --model_config_selector RandomForest_Morgan_Config \
  --split_type scaffold_order \
  --split_sizes 0.5 0.5 \
  --seed $SEED \
  --batch_size 20 \
  --forget_protocol $STRATEGY \
  --forget_size 415 \
  --write_traj_stride 1 \
  --evaluate_stride 1 \
  --max_iter 100"

echo "=== RF plateau-start task $SLURM_ARRAY_TASK_ID: strategy=$STRATEGY seed=$SEED -> $SAVE_DIR ==="
START=$(date +%s)
python "$REPO/molalkit_run" $COMMON_ARGS --save_dir "$SAVE_DIR"
STATUS=$?
END=$(date +%s)
echo "RF plateau-start task $SLURM_ARRAY_TASK_ID finished status=$STATUS in $((END-START))s"
