#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=72:00:00
#SBATCH --partition=common
#SBATCH -o /work/dn123/MolALKit/rf_dmpnn_comparison/logs/cyp_dmpnn_%A_%a.out
#SBATCH -e /work/dn123/MolALKit/rf_dmpnn_comparison/logs/cyp_dmpnn_%A_%a.err
#SBATCH --job-name=cyp2d6_dmpnn
#SBATCH --array=0-11

module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /hpc/group/rekerlab/dn123/conda/envs/molalkit
REPO=/work/dn123/MolALKit; BASE=/work/dn123/MolALKit/rf_dmpnn_comparison
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$BASE/job_array_map_cyp_dmpnn.txt")
STRATEGY=$(echo "$LINE" | awk '{print $1}'); SEED=$(echo "$LINE" | awk '{print $2}')
SAVE_DIR="$BASE/results/CYP2D6_Veith/DMPNN/${STRATEGY}/seed${SEED}"
mkdir -p "$SAVE_DIR"
case $STRATEGY in
  none)          FORGET="" ;;
  forget_random) FORGET="--forget_protocol forget_random --forget_size 40" ;;
  max|min)       FORGET="--forget_protocol ChempropDropoutForgetter --mc_forget_version $STRATEGY --forget_size 40" ;;
esac
# Plain from-scratch DMPNN (no warm start), same design as the MDR1 DMPNN baselines.
python "$REPO/molalkit_run" --data_public CYP2D6_Veith \
  --metrics roc_auc mcc accuracy precision recall f1_score \
  --learning_type explorative --model_config_selector DMPNN_BinaryClassification_Config \
  --split_type scaffold_order --split_sizes 0.5 0.5 \
  --seed $SEED --batch_size 20 --cp_model_epochs 30 $FORGET \
  --write_traj_stride 1 --evaluate_stride 1 --max_iter 100 \
  --save_dir "$SAVE_DIR"
echo "cyp2d6 dmpnn $STRATEGY seed $SEED status=$?"
