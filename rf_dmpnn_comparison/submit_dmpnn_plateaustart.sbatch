#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=common
#SBATCH -o /work/dn123/MolALKit/rf_dmpnn_comparison/logs/dmpnn_plateau_%A_%a.out
#SBATCH -e /work/dn123/MolALKit/rf_dmpnn_comparison/logs/dmpnn_plateau_%A_%a.err
#SBATCH --job-name=rf_dmpnn_comp_DMPNN_plateau
#SBATCH --array=0-1

module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /hpc/group/rekerlab/dn123/conda/envs/molalkit

REPO=/work/dn123/MolALKit
BASE=/work/dn123/MolALKit/rf_dmpnn_comparison

# array index 0 = max, 1 = min; seed fixed at 0.
# forget_size=782: the training-set size at which DMPNN's no-forget baseline
# (results/DMPNN/none/seed{0,1,2}) reaches within 3% of its near-final
# smoothed ROC-AUC (averaged across 3 seeds: 802, 742, 802). Instead of
# forgetting from near the very start (forget_size=40, the original
# arbitrary value), forgetting only kicks in once the model has already
# reached its performance plateau. forget_batch_size left at the default
# (1) to isolate this single variable against the original comparison.
STRATEGIES=(max min)
STRATEGY=${STRATEGIES[$SLURM_ARRAY_TASK_ID]}
SEED=0

SAVE_DIR="$BASE/results/DMPNN_plateaustart/${STRATEGY}/seed${SEED}"
mkdir -p "$SAVE_DIR"

COMMON_ARGS="--data_public MDR1_MDCK_classification2 \
  --metrics roc_auc mcc accuracy precision recall f1_score \
  --learning_type explorative \
  --model_config_selector DMPNN_BinaryClassification_Config \
  --split_type scaffold_order \
  --split_sizes 0.5 0.5 \
  --seed $SEED \
  --batch_size 20 \
  --cp_model_epochs 30 \
  --forget_protocol ChempropDropoutForgetter \
  --mc_forget_version $STRATEGY \
  --forget_size 782 \
  --write_traj_stride 1 \
  --evaluate_stride 1 \
  --max_iter 100"

echo "=== DMPNN plateau-start task $SLURM_ARRAY_TASK_ID: strategy=$STRATEGY seed=$SEED -> $SAVE_DIR ==="
START=$(date +%s)
python "$REPO/molalkit_run" $COMMON_ARGS --save_dir "$SAVE_DIR"
STATUS=$?
END=$(date +%s)
echo "DMPNN plateau-start task $SLURM_ARRAY_TASK_ID finished status=$STATUS in $((END-START))s"
