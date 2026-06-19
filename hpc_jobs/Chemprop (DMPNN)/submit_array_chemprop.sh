#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --partition=common

# Dynamic logging: saves logs into a dedicated folder
# %X is job name, %A is master job ID, %a is individual array task ID
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

# IMPORTANT: Set this equal to your total number of combinations minus 1
# e.g., if you have 48 combinations, your array range is 0-47
#SBATCH --array=1-95%24

#SBATCH --job-name=Molalkit_Grid_Chemprop

# Environment Setup
module load Anaconda3
source activate molalkit

mkdir -p logs

# Task ID to parameters mapping
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" job_array_map.txt)

# Unpacking JSON parameters into native Bash environment variables
eval $(python -c "
import json
line = '''$LINE'''.split('\t')[1]
b_size, start, interval, ep, unc, seed = json.loads(line)
print(f'ADD_B={b_size[0]}; FORGET_B={b_size[1]}; START={start}; INTERVAL={interval}; EPOCHS={ep}; UNC=\"{unc}\"; SEED={seed}')
")

# Setting folder layout
BASE_DIR="../../updated_test_data/MDR1_MDCK_classification2/DMPNN"
PARAM_PATH="b_add${ADD_B}_fg${FORGET_B}/start${START}_int${INTERVAL}/ep${EPOCHS}/${UNC}_unc"
SAVE_DIR="${BASE_DIR}/${PARAM_PATH}/seed${SEED}"

echo "=========================================================="
echo "SLURM ARRAY TASK ID: $SLURM_ARRAY_TASK_ID"
echo "Running parameters: Add Batch=$ADD_B, Forget Batch=$FORGET_B, Start=$START, Interval=$INTERVAL, Epochs=$EPOCHS, Uncertainty=$UNC, Seed=$SEED"
echo "Target Save Directory: $SAVE_DIR"
echo "=========================================================="

# Running Code
python ../../molalkit_run \
    --data_public MDR1_MDCK_classification2 \
    --metrics roc_auc mcc accuracy precision recall f1_score \
    --learning_type explorative \
    --model_config_selector DMPNN_BinaryClassification_Config \
    --split_type scaffold_order \
    --split_sizes 0.5 0.5 \
    --seed "$SEED" \
    --batch_size "$ADD_B" \
    --cp_model_epochs "$EPOCHS" \
    --save_dir "$SAVE_DIR" \
    --forget_protocol ChempropDropoutForgetter \
    --forget_size "$FORGET_B" \
    --mc_forget_version "$UNC" \
    --write_traj_stride 10 \
    --save_cpt_stride 20 \
    --evaluate_stride 5

echo "Task $SLURM_ARRAY_TASK_ID completed successfully."