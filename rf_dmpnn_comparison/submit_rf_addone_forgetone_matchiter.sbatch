#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --partition=common
#SBATCH -o /work/dn123/MolALKit/rf_dmpnn_comparison/logs/rf_addoneforgetone_matchiter_%A_%a.out
#SBATCH -e /work/dn123/MolALKit/rf_dmpnn_comparison/logs/rf_addoneforgetone_matchiter_%A_%a.err
#SBATCH --job-name=rf_dmpnn_comp_RF_addone_forgetone_matchiter
#SBATCH --array=0-5

module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /hpc/group/rekerlab/dn123/conda/envs/molalkit

REPO=/work/dn123/MolALKit
BASE=/work/dn123/MolALKit/rf_dmpnn_comparison
MAP="$BASE/job_array_map_addonefb_matchiter.txt"

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MAP")
STRATEGY=$(echo "$LINE" | awk '{print $1}')
SEED=$(echo "$LINE" | awk '{print $2}')
FORGET_SIZE=$(echo "$LINE" | awk '{print $3}')

# Same forget_size (each seed's own peak train_size) as the earlier
# RF_addone_forgetone_corrected runs, but now max_iter=1150 to match
# RF_noforget_batch1's run length exactly -- gives a much longer post-peak
# 1-for-1 exchange phase (~550-650 more iterations instead of ~110) for a
# true apples-to-apples trajectory-length comparison.
SAVE_DIR="$BASE/results/RF_addone_forgetone_matchiter/${STRATEGY}/seed${SEED}"
mkdir -p "$SAVE_DIR"

COMMON_ARGS="--data_public MDR1_MDCK_classification2 \
  --metrics roc_auc mcc accuracy precision recall f1_score \
  --learning_type explorative \
  --model_config_selector RandomForest_Morgan_Config \
  --split_type scaffold_order \
  --split_sizes 0.5 0.5 \
  --seed $SEED \
  --batch_size 1 \
  --forget_protocol $STRATEGY \
  --forget_size $FORGET_SIZE \
  --forget_batch_size 1 \
  --write_traj_stride 1 \
  --evaluate_stride 1 \
  --max_iter 1150"

echo "=== RF add-one/forget-one (match-iter) task $SLURM_ARRAY_TASK_ID: strategy=$STRATEGY seed=$SEED forget_size=$FORGET_SIZE -> $SAVE_DIR ==="
START=$(date +%s)
python "$REPO/molalkit_run" $COMMON_ARGS --save_dir "$SAVE_DIR"
STATUS=$?
END=$(date +%s)
echo "RF add-one/forget-one (match-iter) task $SLURM_ARRAY_TASK_ID finished status=$STATUS in $((END-START))s"
