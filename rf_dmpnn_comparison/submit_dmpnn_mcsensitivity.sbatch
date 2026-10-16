#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=common
#SBATCH -o /work/dn123/MolALKit/rf_dmpnn_comparison/logs/dmpnn_mcsens_%A_%a.out
#SBATCH -e /work/dn123/MolALKit/rf_dmpnn_comparison/logs/dmpnn_mcsens_%A_%a.err
#SBATCH --job-name=rf_dmpnn_comp_DMPNN_mcsensitivity
#SBATCH --array=0-3

module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /hpc/group/rekerlab/dn123/conda/envs/molalkit

REPO=/work/dn123/MolALKit
BASE=/work/dn123/MolALKit/rf_dmpnn_comparison
MAP="$BASE/job_array_map_mcsensitivity.txt"

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MAP")
MC_SAMPLES=$(echo "$LINE" | awk '{print $1}')
DROPOUT_PROB=$(echo "$LINE" | awk '{print $2}')
STRATEGY=$(echo "$LINE" | awk '{print $3}')
SEED=0

# Tests whether the single-molecule "fixation" pathology seen with bald-min
# (one molecule forgotten 42-43/99 times) is an artifact of a noisy/unstable
# MC-dropout uncertainty estimate (default mc_samples=20, dropout_prob=0.1)
# rather than a real property of the data. Two variants: more MC samples
# (50, same dropout_prob) for a more stable estimate, and higher dropout
# (0.3, same mc_samples) for a stronger uncertainty signal. Same base config
# otherwise (batch_size=20, forget_size=40, forget_batch_size=1, bald score).
SAVE_DIR="$BASE/results/DMPNN_mcsensitivity/mc${MC_SAMPLES}_dp${DROPOUT_PROB}_${STRATEGY}/seed${SEED}"
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
  --forget_mc_samples $MC_SAMPLES \
  --forget_dropout_prob $DROPOUT_PROB \
  --forget_size 40 \
  --write_traj_stride 1 \
  --evaluate_stride 1 \
  --max_iter 100"

echo "=== DMPNN mc-sensitivity task $SLURM_ARRAY_TASK_ID: mc_samples=$MC_SAMPLES dropout_prob=$DROPOUT_PROB target=$STRATEGY seed=$SEED -> $SAVE_DIR ==="
START=$(date +%s)
python "$REPO/molalkit_run" $COMMON_ARGS --save_dir "$SAVE_DIR"
STATUS=$?
END=$(date +%s)
echo "DMPNN mc-sensitivity task $SLURM_ARRAY_TASK_ID finished status=$STATUS in $((END-START))s"
