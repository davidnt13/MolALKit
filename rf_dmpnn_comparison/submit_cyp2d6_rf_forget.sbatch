#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --partition=common
#SBATCH -o /work/dn123/MolALKit/rf_dmpnn_comparison/logs/cyp_rf_forget_%A_%a.out
#SBATCH -e /work/dn123/MolALKit/rf_dmpnn_comparison/logs/cyp_rf_forget_%A_%a.err
#SBATCH --job-name=cyp2d6_rf_forget
#SBATCH --array=0-8

module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /hpc/group/rekerlab/dn123/conda/envs/molalkit
REPO=/work/dn123/MolALKit; BASE=/work/dn123/MolALKit/rf_dmpnn_comparison
STRATS=(forget_random max_oob_uncertainty min_oob_uncertainty)
STRATEGY=${STRATS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=$((SLURM_ARRAY_TASK_ID % 3))
NF="$BASE/results/CYP2D6_Veith/RF_noforget_batch1/seed${SEED}/al_traj.csv"
read FORGET_SIZE N_ITER < <(python3 -I "$BASE/peak_info.py" "$NF")
SAVE_DIR="$BASE/results/CYP2D6_Veith/RF_addone_forgetone/${STRATEGY}/seed${SEED}"
mkdir -p "$SAVE_DIR"
echo "=== $STRATEGY seed $SEED: forget_size=$FORGET_SIZE (no-forget MCC peak), max_iter=$N_ITER (no-forget run length) ==="
# Add-one/forget-one: forgetting starts once the training set reaches the no-forget MCC peak.
python "$REPO/molalkit_run" --data_public CYP2D6_Veith \
  --metrics roc_auc mcc accuracy precision recall f1_score \
  --learning_type explorative --model_config_selector RandomForest_Morgan_Config \
  --split_type scaffold_order --split_sizes 0.5 0.5 \
  --seed $SEED --batch_size 1 --forget_protocol $STRATEGY \
  --forget_size $FORGET_SIZE --forget_batch_size 1 \
  --write_traj_stride 1 --evaluate_stride 1 --max_iter $N_ITER \
  --save_dir "$SAVE_DIR"
echo "cyp2d6 rf $STRATEGY seed $SEED status=$?"
