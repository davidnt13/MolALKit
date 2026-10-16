#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --partition=common
#SBATCH -o /work/dn123/MolALKit/rf_dmpnn_comparison/logs/rf_addoneforgetone_%A_%a.out
#SBATCH -e /work/dn123/MolALKit/rf_dmpnn_comparison/logs/rf_addoneforgetone_%A_%a.err
#SBATCH --job-name=rf_dmpnn_comp_RF_addone_forgetone
#SBATCH --array=0-1

module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /hpc/group/rekerlab/dn123/conda/envs/molalkit

REPO=/work/dn123/MolALKit
BASE=/work/dn123/MolALKit/rf_dmpnn_comparison

# array index 0 = max_oob_uncertainty, 1 = min_oob_uncertainty; seed fixed at 0.
# "add one/forget one": batch_size=1 (add rate) AND forget_batch_size=1
# (forget rate) -- an exact 1:1 exchange once forgetting kicks in, unlike
# every earlier RF test which used batch_size=20 (add-20/forget-1).
# forget_size=542: the training-set size at which RF's no-forget baseline
# (results/RF/none/seed{0,1,2}, batch_size=20) reaches its single MAX MCC
# point (not a plateau-tolerance band like the earlier plateaustart run) --
# averaged across 3 seeds: 522, 602, 502.
# max_iter=650: ~540 iterations just to reach train_size=542 at 1/iteration,
# plus ~110 more iterations of actual 1-for-1 forgetting after that.
STRATEGIES=(max_oob_uncertainty min_oob_uncertainty)
STRATEGY=${STRATEGIES[$SLURM_ARRAY_TASK_ID]}
SEED=0

SAVE_DIR="$BASE/results/RF_addone_forgetone/${STRATEGY}/seed${SEED}"
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
  --forget_size 542 \
  --forget_batch_size 1 \
  --write_traj_stride 1 \
  --evaluate_stride 1 \
  --max_iter 650"

echo "=== RF add-one/forget-one task $SLURM_ARRAY_TASK_ID: strategy=$STRATEGY seed=$SEED -> $SAVE_DIR ==="
START=$(date +%s)
python "$REPO/molalkit_run" $COMMON_ARGS --save_dir "$SAVE_DIR"
STATUS=$?
END=$(date +%s)
echo "RF add-one/forget-one task $SLURM_ARRAY_TASK_ID finished status=$STATUS in $((END-START))s"
