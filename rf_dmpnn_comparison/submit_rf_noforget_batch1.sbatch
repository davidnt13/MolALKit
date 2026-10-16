#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --partition=common
#SBATCH -o /work/dn123/MolALKit/rf_dmpnn_comparison/logs/rf_noforget_batch1_%j.out
#SBATCH -e /work/dn123/MolALKit/rf_dmpnn_comparison/logs/rf_noforget_batch1_%j.err
#SBATCH --job-name=rf_dmpnn_comp_RF_noforget_batch1

module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /hpc/group/rekerlab/dn123/conda/envs/molalkit

REPO=/work/dn123/MolALKit
BASE=/work/dn123/MolALKit/rf_dmpnn_comparison
SEED=0

# No-forget RF at batch_size=1 (matching the add-one/forget-one runs'
# granularity), run to full pool exhaustion (~1108 molecules), so the peak
# point used to start forgetting is derived from a curve with the SAME
# batch size as the forgetting runs that will use it -- not borrowed from
# the earlier batch_size=20 no-forget curve.
SAVE_DIR="$BASE/results/RF_noforget_batch1/seed${SEED}"
mkdir -p "$SAVE_DIR"

COMMON_ARGS="--data_public MDR1_MDCK_classification2 \
  --metrics roc_auc mcc accuracy precision recall f1_score \
  --learning_type explorative \
  --model_config_selector RandomForest_Morgan_Config \
  --split_type scaffold_order \
  --split_sizes 0.5 0.5 \
  --seed $SEED \
  --batch_size 1 \
  --write_traj_stride 1 \
  --evaluate_stride 1 \
  --max_iter 1150"

echo "=== RF no-forget batch_size=1 -> $SAVE_DIR ==="
START=$(date +%s)
python "$REPO/molalkit_run" $COMMON_ARGS --save_dir "$SAVE_DIR"
STATUS=$?
END=$(date +%s)
echo "RF no-forget batch_size=1 finished status=$STATUS in $((END-START))s"
