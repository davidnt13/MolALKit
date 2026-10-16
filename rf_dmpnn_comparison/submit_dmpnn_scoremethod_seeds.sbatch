#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=common
#SBATCH -o /work/dn123/MolALKit/rf_dmpnn_comparison/logs/dmpnn_sm_seeds_%A_%a.out
#SBATCH -e /work/dn123/MolALKit/rf_dmpnn_comparison/logs/dmpnn_sm_seeds_%A_%a.err
#SBATCH --job-name=rf_dmpnn_comp_DMPNN_scoremethod_seeds
#SBATCH --array=0-3

module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /hpc/group/rekerlab/dn123/conda/envs/molalkit

REPO=/work/dn123/MolALKit
BASE=/work/dn123/MolALKit/rf_dmpnn_comparison
MAP="$BASE/job_array_map_scoremethod_seeds.txt"

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MAP")
SCORE_METHOD=$(echo "$LINE" | awk '{print $1}')
STRATEGY=$(echo "$LINE" | awk '{print $2}')
SEED=$(echo "$LINE" | awk '{print $3}')

# Confirming variance-max=0.835 and entropy-max=0.826 (seed0, both beating
# the no-forget baseline of 0.830) aren't seed noise, by adding seeds 1,2
# (seed0 already exists in results/DMPNN_scoremethod/{method}_{strategy}/seed0).
SAVE_DIR="$BASE/results/DMPNN_scoremethod/${SCORE_METHOD}_${STRATEGY}/seed${SEED}"
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
  --forget_score_method $SCORE_METHOD \
  --forget_size 40 \
  --write_traj_stride 1 \
  --evaluate_stride 1 \
  --max_iter 100"

echo "=== DMPNN score_method seed-confirm task $SLURM_ARRAY_TASK_ID: score_method=$SCORE_METHOD target=$STRATEGY seed=$SEED -> $SAVE_DIR ==="
START=$(date +%s)
python "$REPO/molalkit_run" $COMMON_ARGS --save_dir "$SAVE_DIR"
STATUS=$?
END=$(date +%s)
echo "DMPNN score_method seed-confirm task $SLURM_ARRAY_TASK_ID finished status=$STATUS in $((END-START))s"
