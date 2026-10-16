#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=common
#SBATCH -o /work/dn123/MolALKit/rf_dmpnn_comparison/logs/dmpnn_opt3_test_%j.out
#SBATCH -e /work/dn123/MolALKit/rf_dmpnn_comparison/logs/dmpnn_opt3_test_%j.err
#SBATCH --job-name=dmpnn_opt3_test

module load Anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /hpc/group/rekerlab/dn123/conda/envs/molalkit
REPO=/work/dn123/MolALKit; BASE=/work/dn123/MolALKit/rf_dmpnn_comparison
# Option 3 test (MDR1, seed 0, max/bald): add 20 / forget 20 once the training set reaches the
# RF no-forget MCC peak (train size 572, from results/RF_noforget_batch1/seed0). Compare with
# results/DMPNN_cpu/max/seed0 (forget_size 40, forget_batch_size 1) and results/DMPNN/none/seed0.
SAVE_DIR="$BASE/results/DMPNN_option3_test/max/seed0"
mkdir -p "$SAVE_DIR"
python "$REPO/molalkit_run" --data_public MDR1_MDCK_classification2 \
  --metrics roc_auc mcc accuracy precision recall f1_score \
  --learning_type explorative --model_config_selector DMPNN_BinaryClassification_Config \
  --split_type scaffold_order --split_sizes 0.5 0.5 \
  --seed 0 --batch_size 20 --cp_model_epochs 30 \
  --forget_protocol ChempropDropoutForgetter --mc_forget_version max \
  --forget_size 572 --forget_batch_size 20 \
  --write_traj_stride 1 --evaluate_stride 1 --max_iter 100 \
  --save_dir "$SAVE_DIR"
echo "opt3 test status=$?"
