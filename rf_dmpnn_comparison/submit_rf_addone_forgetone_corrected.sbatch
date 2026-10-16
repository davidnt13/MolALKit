#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --partition=common
#SBATCH -o /work/dn123/MolALKit/rf_dmpnn_comparison/logs/rf_addoneforgetone_corrected_%A_%a.out
#SBATCH -e /work/dn123/MolALKit/rf_dmpnn_comparison/logs/rf_addoneforgetone_corrected_%A_%a.err
#SBATCH --job-name=rf_dmpnn_comp_RF_addone_forgetone_corrected
#SBATCH --array=0-1

module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /hpc/group/rekerlab/dn123/conda/envs/molalkit

REPO=/work/dn123/MolALKit
BASE=/work/dn123/MolALKit/rf_dmpnn_comparison

# array index 0 = max_oob_uncertainty, 1 = min_oob_uncertainty; seed fixed at 0.
# CORRECTED version: forget_size=572 is the peak-MCC point of RF's no-forget
# curve run at the SAME batch_size=1 as this experiment (results/RF_noforget_batch1,
# peak MCC=0.4711 at iter=569, train_size=572) -- unlike the earlier version,
# which mistakenly used the peak from a batch_size=20 no-forget curve (542),
# a different trajectory shape than the batch_size=1 runs being compared to it.
# max_iter=682: ~572 iterations to reach train_size=572 at 1/iteration, plus
# ~110 more iterations of actual 1-for-1 exchange after that.
STRATEGIES=(max_oob_uncertainty min_oob_uncertainty)
STRATEGY=${STRATEGIES[$SLURM_ARRAY_TASK_ID]}
SEED=0

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
  --forget_size 572 \
  --forget_batch_size 1 \
  --write_traj_stride 1 \
  --evaluate_stride 1 \
  --max_iter 682"

echo "=== RF add-one/forget-one (corrected) task $SLURM_ARRAY_TASK_ID: strategy=$STRATEGY seed=$SEED -> $SAVE_DIR ==="
START=$(date +%s)
python "$REPO/molalkit_run" $COMMON_ARGS --save_dir "$SAVE_DIR"
STATUS=$?
END=$(date +%s)
echo "RF add-one/forget-one (corrected) task $SLURM_ARRAY_TASK_ID finished status=$STATUS in $((END-START))s"
