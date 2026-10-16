#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --partition=common
#SBATCH -o /work/dn123/MolALKit/rf_dmpnn_comparison/logs/rf_addoneforgetone_seeds_%A_%a.out
#SBATCH -e /work/dn123/MolALKit/rf_dmpnn_comparison/logs/rf_addoneforgetone_seeds_%A_%a.err
#SBATCH --job-name=rf_dmpnn_comp_RF_addone_forgetone_seeds
#SBATCH --array=0-3

module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /hpc/group/rekerlab/dn123/conda/envs/molalkit

REPO=/work/dn123/MolALKit
BASE=/work/dn123/MolALKit/rf_dmpnn_comparison
MAP="$BASE/job_array_map_addonefb_seeds.txt"

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MAP")
STRATEGY=$(echo "$LINE" | awk '{print $1}')
SEED=$(echo "$LINE" | awk '{print $2}')
FORGET_SIZE=$(echo "$LINE" | awk '{print $3}')
MAX_ITER=$((FORGET_SIZE + 110))

# Each seed uses ITS OWN peak train_size (from RF_noforget_batch1/seed{N}):
# seed1 peak train_size=498 (MCC 0.4868 at iter=495)
# seed2 peak train_size=626 (MCC 0.4612 at iter=623)
# matching the already-completed seed0 (forget_size=572) run in
# RF_addone_forgetone_corrected/, to build n=3 statistics.
SAVE_DIR="$BASE/results/RF_addone_forgetone_corrected/${STRATEGY}/seed${SEED}"
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
  --max_iter $MAX_ITER"

echo "=== RF add-one/forget-one task $SLURM_ARRAY_TASK_ID: strategy=$STRATEGY seed=$SEED forget_size=$FORGET_SIZE -> $SAVE_DIR ==="
START=$(date +%s)
python "$REPO/molalkit_run" $COMMON_ARGS --save_dir "$SAVE_DIR"
STATUS=$?
END=$(date +%s)
echo "RF add-one/forget-one task $SLURM_ARRAY_TASK_ID finished status=$STATUS in $((END-START))s"
