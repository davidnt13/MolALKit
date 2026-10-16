#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=common
#SBATCH -o /work/dn123/MolALKit/rf_dmpnn_comparison/logs/random_%A_%a.out
#SBATCH -e /work/dn123/MolALKit/rf_dmpnn_comparison/logs/random_%A_%a.err
#SBATCH --job-name=rf_dmpnn_comp_random_control
#SBATCH --array=0-1

module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /hpc/group/rekerlab/dn123/conda/envs/molalkit

REPO=/work/dn123/MolALKit
BASE=/work/dn123/MolALKit/rf_dmpnn_comparison
SEED=0

# array index 0 = DMPNN, 1 = RF. forget_random isolates whether forgetting
# per se hurts performance, independent of the uncertainty-based targeting
# rule (max/min). Same forget_size=40 as the original comparison.
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  MODEL_CONFIG="DMPNN_BinaryClassification_Config"
  EXTRA_ARGS="--cp_model_epochs 30"
  SAVE_DIR="$BASE/results/DMPNN_random/seed${SEED}"
else
  MODEL_CONFIG="RandomForest_Morgan_Config"
  EXTRA_ARGS=""
  SAVE_DIR="$BASE/results/RF_random/seed${SEED}"
fi
mkdir -p "$SAVE_DIR"

COMMON_ARGS="--data_public MDR1_MDCK_classification2 \
  --metrics roc_auc mcc accuracy precision recall f1_score \
  --learning_type explorative \
  --model_config_selector $MODEL_CONFIG \
  --split_type scaffold_order \
  --split_sizes 0.5 0.5 \
  --seed $SEED \
  --batch_size 20 \
  $EXTRA_ARGS \
  --forget_protocol forget_random \
  --forget_size 40 \
  --write_traj_stride 1 \
  --evaluate_stride 1 \
  --max_iter 100"

echo "=== random-forgetting control task $SLURM_ARRAY_TASK_ID: model=$MODEL_CONFIG seed=$SEED -> $SAVE_DIR ==="
START=$(date +%s)
python "$REPO/molalkit_run" $COMMON_ARGS --save_dir "$SAVE_DIR"
STATUS=$?
END=$(date +%s)
echo "random-forgetting control task $SLURM_ARRAY_TASK_ID finished status=$STATUS in $((END-START))s"
