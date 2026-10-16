#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=common
#SBATCH -o /work/dn123/MolALKit/rf_dmpnn_comparison/logs/dmpnn_fb10_%A_%a.out
#SBATCH -e /work/dn123/MolALKit/rf_dmpnn_comparison/logs/dmpnn_fb10_%A_%a.err
#SBATCH --job-name=rf_dmpnn_comp_DMPNN_forgetbatch10
#SBATCH --array=0-1

module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /hpc/group/rekerlab/dn123/conda/envs/molalkit

REPO=/work/dn123/MolALKit
BASE=/work/dn123/MolALKit/rf_dmpnn_comparison

# array index 0 = max, 1 = min; seed fixed at 0 for this targeted test.
# forget_batch_size=10, half of add batch_size=20, so the net training set
# still grows (+10/iteration) instead of staying flat like the =20 test did
# — isolates the forgetting-target effect from the "starved of data" confound.
STRATEGIES=(max min)
STRATEGY=${STRATEGIES[$SLURM_ARRAY_TASK_ID]}
SEED=0

SAVE_DIR="$BASE/results/DMPNN_forgetbatch10/${STRATEGY}/seed${SEED}"
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
  --forget_size 40 \
  --forget_batch_size 10 \
  --write_traj_stride 1 \
  --evaluate_stride 1 \
  --max_iter 100"

echo "=== DMPNN forget_batch_size=10 task $SLURM_ARRAY_TASK_ID: strategy=$STRATEGY seed=$SEED -> $SAVE_DIR ==="
START=$(date +%s)
python "$REPO/molalkit_run" $COMMON_ARGS --save_dir "$SAVE_DIR"
STATUS=$?
END=$(date +%s)
echo "DMPNN forget_batch_size=10 task $SLURM_ARRAY_TASK_ID finished status=$STATUS in $((END-START))s"
