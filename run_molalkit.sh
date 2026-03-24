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

python molalkit_run --data_public pgp_broccatelli --metrics roc_auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector David_MLP_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 2 --batch_size 2 --save_dir "test data/pgp/mlp/pgp_mlp_maxf3" --forget_protocol MCDropoutForgetter --forget_size 200

# python molalkit_run --data_public bbb_martins --metrics roc_auc mcc accuracy precision recall f1_score --learning_type passive --model_config_selector David_MLP_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir "test data/bbb_david_mforg_batch2_random" --forget_protocol MCDropoutForgetter --forget_size 250 --batch_size 2

# python molalkit_run --data_public bace --metrics roc_auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector MLP_Morgan_BinaryClassification_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir bace_dnn_mc_forget --forget_protocol MCDropoutForgetter --forget_size 2

# python molalkit_run --data_public pgp_broccatelli  --metrics roc_auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector RandomForest_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --batch_size 2 --save_dir "test data/pgp/rf/pgp_rf_explorative_maxf"  --forget_protocol max_oob_uncertainty --forget_size 200

# Models Used:
# Custom MLP: David_MLP_Morgan_Config
# RF: RandomForest_Morgan_Config
# Chemprop: MLP_Morgan_BinaryClassification_Config

# Datasets Used:
# pgp_broccatelli
# bbb_martins
# bace
# CYP3A4_Veith

# Forget Protocols:
# MCDropoutForgetter (MLP)
# max_oob_uncertainty
# min_oob_uncertainty
