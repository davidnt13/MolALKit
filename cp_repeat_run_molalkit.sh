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

python molalkit_run --data_public MDR1_MDCK_classification2  --metrics roc_auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector MLP_Morgan_BinaryClassification_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --batch_size 1 --save_dir "test data/mdr1/cp/mdr1_cp_minf"  --forget_protocol ChempropDropoutForgetter --forget_size 500 --mc_forget_version "min"

python molalkit_run --data_public MDR1_MDCK_classification2  --metrics roc_auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector MLP_Morgan_BinaryClassification_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 1 --batch_size 1 --save_dir "test data/mdr1/cp/mdr1_cp_minf2"  --forget_protocol ChempropDropoutForgetter --forget_size 500 --mc_forget_version "min"

python molalkit_run --data_public MDR1_MDCK_classification2  --metrics roc_auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector MLP_Morgan_BinaryClassification_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 2 --batch_size 1 --save_dir "test data/mdr1/cp/mdr1_cp_minf3"  --forget_protocol ChempropDropoutForgetter --forget_size 500 --mc_forget_version "min"

python molalkit_run --data_public MDR1_MDCK_classification2  --metrics roc_auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector MLP_Morgan_BinaryClassification_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --batch_size 1 --save_dir "test data/mdr1/cp/mdr1_cp_maxf"  --forget_protocol ChempropDropoutForgetter --forget_size 500 --mc_forget_version "max"

python molalkit_run --data_public MDR1_MDCK_classification2  --metrics roc_auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector MLP_Morgan_BinaryClassification_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 1 --batch_size 1 --save_dir "test data/mdr1/cp/mdr1_cp_maxf2"  --forget_protocol ChempropDropoutForgetter --forget_size 500 --mc_forget_version "max"

python molalkit_run --data_public MDR1_MDCK_classification2  --metrics roc_auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector MLP_Morgan_BinaryClassification_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 2 --batch_size 1 --save_dir "test data/mdr1/cp/mdr1_cp_maxf3"  --forget_protocol ChempropDropoutForgetter --forget_size 500 --mc_forget_version "max"

python molalkit_run --data_public MDR1_MDCK_classification2  --metrics roc_auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector MLP_Morgan_BinaryClassification_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --batch_size 1 --save_dir "test data/mdr1/cp/mdr1_cp_explorative_nf"  #--forget_protocol min_oob_uncertainty --forget_size 200

python molalkit_run --data_public MDR1_MDCK_classification2  --metrics roc_auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector MLP_Morgan_BinaryClassification_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 1 --batch_size 1 --save_dir "test data/mdr1/cp/mdr1_cp_explorative_nf2"  #--forget_protocol min_oob_uncertainty --forget_size 200

python molalkit_run --data_public MDR1_MDCK_classification2  --metrics roc_auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector MLP_Morgan_BinaryClassification_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 2 --batch_size 1 --save_dir "test data/mdr1/cp/mdr1_cp_explorative_nf3"  #--forget_protocol max_oob_uncertainty --forget_size 200


# Models Used:
# Custom MLP: David_MLP_Morgan_Config
# RF: RandomForest_Morgan_Config
# Chemprop: MLP_Morgan_BinaryClassification_Config

# Datasets Used:
# pgp_broccatelli
# bbb_martins
# bace
# CYP3A4_Veith
# MDR1_MDCK_classification2

# Forget Protocols:
# MCDropoutForgetter (MLP)
# max_oob_uncertainty
# min_oob_uncertainty
