#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --partition=common

#SBATCH -o %j.out
#SBATCH -e %j.err

#SBATCH --export=ALL

#SBATCH --job-name=Molalkit_dnn


module load Anaconda3
source activate molalkit

python molalkit_run --data_public bace --metrics roc_auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector RandomForest_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir bace_rf_min_oob --forget_protocol min_oob_uncertainty --forget_size 200
