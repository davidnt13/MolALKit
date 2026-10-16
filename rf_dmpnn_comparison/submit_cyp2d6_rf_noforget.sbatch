#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=common
#SBATCH -o /work/dn123/MolALKit/rf_dmpnn_comparison/logs/cyp_rf_none_%A_%a.out
#SBATCH -e /work/dn123/MolALKit/rf_dmpnn_comparison/logs/cyp_rf_none_%A_%a.err
#SBATCH --job-name=cyp2d6_rf_none
#SBATCH --array=0-2

module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /hpc/group/rekerlab/dn123/conda/envs/molalkit
REPO=/work/dn123/MolALKit; BASE=/work/dn123/MolALKit/rf_dmpnn_comparison
SEED=$SLURM_ARRAY_TASK_ID
SAVE_DIR="$BASE/results/CYP2D6_Veith/RF_noforget_batch1/seed${SEED}"
mkdir -p "$SAVE_DIR"
# Full no-forget run: add one molecule per iteration until the pool is exhausted (max_iter > pool size).
python "$REPO/molalkit_run" --data_public CYP2D6_Veith \
  --metrics roc_auc mcc accuracy precision recall f1_score \
  --learning_type explorative --model_config_selector RandomForest_Morgan_Config \
  --split_type scaffold_order --split_sizes 0.5 0.5 \
  --seed $SEED --batch_size 1 --write_traj_stride 1 --evaluate_stride 1 --max_iter 7000 \
  --save_dir "$SAVE_DIR"
echo "cyp2d6 rf noforget seed $SEED status=$?"
