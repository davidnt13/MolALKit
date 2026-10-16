#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH -o /work/dn123/MolALKit/rf_dmpnn_comparison/logs/dmpnn_%A_%a.out
#SBATCH -e /work/dn123/MolALKit/rf_dmpnn_comparison/logs/dmpnn_%A_%a.err
#SBATCH --job-name=rf_dmpnn_comp_DMPNN
#SBATCH --array=0-8

module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /hpc/group/rekerlab/dn123/conda/envs/molalkit

REPO=/work/dn123/MolALKit
BASE=/work/dn123/MolALKit/rf_dmpnn_comparison
MAP="$BASE/job_array_map_dmpnn.txt"

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MAP")
STRATEGY=$(echo "$LINE" | awk '{print $1}')
SEED=$(echo "$LINE" | awk '{print $2}')

SAVE_DIR="$BASE/results/DMPNN/${STRATEGY}/seed${SEED}"
mkdir -p "$SAVE_DIR"

echo "=== GPU visible to job ==="
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

COMMON_ARGS="--data_public MDR1_MDCK_classification2 \
  --metrics roc_auc mcc accuracy precision recall f1_score \
  --learning_type explorative \
  --model_config_selector DMPNN_BinaryClassification_Config \
  --split_type scaffold_order \
  --split_sizes 0.5 0.5 \
  --seed $SEED \
  --batch_size 20 \
  --cp_model_epochs 30 \
  --write_traj_stride 1 \
  --evaluate_stride 1 \
  --max_iter 100"

echo "=== DMPNN task $SLURM_ARRAY_TASK_ID: strategy=$STRATEGY seed=$SEED -> $SAVE_DIR ==="
START=$(date +%s)

if [ "$STRATEGY" = "none" ]; then
  python "$REPO/molalkit_run" $COMMON_ARGS --save_dir "$SAVE_DIR"
else
  python "$REPO/molalkit_run" $COMMON_ARGS --forget_protocol ChempropDropoutForgetter --mc_forget_version "$STRATEGY" --forget_size 40 --save_dir "$SAVE_DIR"
fi

STATUS=$?
END=$(date +%s)
echo "DMPNN task $SLURM_ARRAY_TASK_ID finished status=$STATUS in $((END-START))s"
