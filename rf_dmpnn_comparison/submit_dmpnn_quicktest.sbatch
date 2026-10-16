#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=common
#SBATCH -o /work/dn123/MolALKit/rf_dmpnn_comparison/logs/dmpnn_quicktest_%A_%a.out
#SBATCH -e /work/dn123/MolALKit/rf_dmpnn_comparison/logs/dmpnn_quicktest_%A_%a.err
#SBATCH --job-name=dmpnn_quicktest
#SBATCH --array=0-3

module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /hpc/group/rekerlab/dn123/conda/envs/molalkit
REPO=/work/dn123/MolALKit; BASE=/work/dn123/MolALKit/rf_dmpnn_comparison
MODES=(none max min random)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
D=$BASE/quicktest_data
SAVE_DIR="$BASE/results/DMPNN_quicktest_rfpeak/${MODE}/seed0"
mkdir -p "$SAVE_DIR"
case $MODE in
  none)   FORGET="" ;;
  random) FORGET="--forget_protocol forget_random --forget_size 572 --forget_batch_size 20" ;;
  max|min) FORGET="--forget_protocol ChempropDropoutForgetter --mc_forget_version $MODE --forget_size 572 --forget_batch_size 20" ;;
esac
# Quick test: DMPNN (from scratch, 30 epochs) starting from the RF's retained set at its MDR1 seed-0
# no-forget MCC peak (572 molecules). Forgetting (if any) starts at iteration 0: add 20 / forget 20.
python "$REPO/molalkit_run" \
  --data_path_training $D/mdr1_rfpeak_seed0_train.csv --data_path_pool $D/mdr1_rfpeak_seed0_pool.csv \
  --data_path_val $D/mdr1_rfpeak_seed0_val.csv \
  --pure_columns SMILES --target_columns Y --dataset_type classification \
  --metrics roc_auc mcc accuracy precision recall f1_score \
  --learning_type explorative --model_config_selector DMPNN_BinaryClassification_Config \
  --seed 0 --batch_size 20 --cp_model_epochs 30 $FORGET \
  --write_traj_stride 1 --evaluate_stride 1 --max_iter 15 \
  --save_dir "$SAVE_DIR"
echo "quicktest $MODE status=$?"
