#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=common
#SBATCH -o /work/dn123/MolALKit/rf_dmpnn_comparison/logs/dmpnn_speedup_forget_%A_%a.out
#SBATCH -e /work/dn123/MolALKit/rf_dmpnn_comparison/logs/dmpnn_speedup_forget_%A_%a.err
#SBATCH --job-name=rf_dmpnn_comp_DMPNN_speedup_forget
#SBATCH --array=0-11

module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /hpc/group/rekerlab/dn123/conda/envs/molalkit

REPO=/work/dn123/MolALKit
BASE=/work/dn123/MolALKit/rf_dmpnn_comparison
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$BASE/job_array_map_speedup_forget.txt")
VARIANT=$(echo "$LINE" | awk '{print $1}')
STRATEGY=$(echo "$LINE" | awk '{print $2}')
SEED=$(echo "$LINE" | awk '{print $3}')

case $VARIANT in
  warm10)        CFG=DMPNN_BinaryClassification_Warm_Config ;;
  warm10_bs200)  CFG=DMPNN_BinaryClassification_WarmBs200_Config ;;
esac

# variance scoring, forget_size 40, forget_batch_size 1 (same as DMPNN_scoremethod/variance_*)
SAVE_DIR="$BASE/results/DMPNN_speedup_forget/${VARIANT}/variance_${STRATEGY}/seed${SEED}"
mkdir -p "$SAVE_DIR"
START=$(date +%s)
python "$REPO/molalkit_run" --data_public MDR1_MDCK_classification2 \
  --metrics roc_auc mcc accuracy precision recall f1_score \
  --learning_type explorative \
  --model_config_selector $CFG \
  --split_type scaffold_order --split_sizes 0.5 0.5 \
  --seed $SEED --batch_size 20 --cp_model_epochs 10 \
  --forget_protocol ChempropDropoutForgetter \
  --mc_forget_version $STRATEGY --forget_score_method variance \
  --forget_size 40 \
  --write_traj_stride 1 --evaluate_stride 1 --max_iter 100 \
  --save_dir "$SAVE_DIR"
echo "speedup_forget $VARIANT variance_$STRATEGY seed $SEED status=$? in $(( $(date +%s)-START ))s"
