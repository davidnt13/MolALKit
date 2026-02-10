#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
from functools import cached_property
from tap import Tap
import os
import shutil
import json
import math
import pandas as pd
import numpy as np
from mgktools.features_mol.features_generators import FeaturesGenerator
from mgktools.data.split import data_split_index
from mgktools.evaluators.metric import Metric
from molalkit.logging import create_logger
from molalkit.utils import get_data, get_model, get_kernel
from molalkit.data.datasets import DATA_DIR
from molalkit.al.selection_method import *
from molalkit.al.forgetter import *
CWD = os.path.dirname(os.path.abspath(__file__))


class CommonArgs(Tap):
    save_dir: str
    """the output directory."""
    n_jobs: int = 1
    """the cpu numbers used for parallel computing."""
    quiet: bool = False
    """Whether the stream handler should be quiet (i.e., print only important info)."""
    logger_name: str = 'alb_output'
    """the prefix of the output logger file: verbose.log and quite.log"""
    seed: int = 0
    """random seed."""

    def process_args(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger = create_logger(self.logger_name, save_dir=self.save_dir, quiet=self.quiet)
        np.random.seed(self.seed)


class DatasetArgs(CommonArgs):
    data_public = None
    """Use public data sets."""
    data_path: str = None
    """the Path of input data CSV file."""
    data_path_training: str = None
    """the Path of input data CSV file for training set."""
    data_path_pool: str = None
    """the Path of input data CSV file for pool set."""
    data_path_val: str = None
    """the Path of input data CSV file for validation set."""
    pure_columns: List[str] = None
    """
    For pure compounds.
    Name of the columns containing single SMILES or InChI string.
    """
    mixture_columns: List[str] = None
    """
    For mixtures.
    Name of the columns containing multiple SMILES or InChI string and 
    corresponding concentration.
    example: ['C', 0.5, 'CC', 0.3]
    """
    target_columns: List[str] = None
    """
    Name of the columns containing target values.
    """
    feature_columns: List[str] = None
    """
    Name of the columns containing additional features_mol such as temperature, 
    pressuer.
    """
    dataset_type: Literal['regression', 'classification', 'multiclass'] = None
    """
    Type of task.
    """
    split_type: Literal['random', 'scaffold_random', 'scaffold_order'] = None
    """Method of splitting the data into active learning/validation."""
    split_sizes: List[float] = None
    """Split proportions for active learning/validation sets."""
    full_val: bool = False
    """validate the performance of active learning on the full dataset."""
    error_rate: float = None
    """the percent of the training set that will be affected by error (0-1)."""

    def get_train_pool_split_index(self, df_al: pd.DataFrame) -> Tuple[List[int], List[int]]:
        if self.init_size >= len(df_al):
            self.logger.warning('molalkit warning: init_size is larger than the dataset size, '
                                'use the full dataset as training set and empty pool set.')
            train_index, pool_index = list(range(len(df_al))), []
        else:
            if self.dataset_type == 'regression':
                train_index, pool_index = data_split_index(
                    n_samples=len(df_al),
                    mols=df_al[self.pure_columns[0]] if self.pure_columns is not None else None,
                    split_type='random',
                    sizes=[self.init_size / len(df_al), 1 - self.init_size / len(df_al)],
                    seed=self.seed)
            else:
                train_index, pool_index = data_split_index(
                    n_samples=len(df_al),
                    mols=df_al[self.pure_columns[0]] if self.pure_columns is not None else None,
                    targets=df_al[self.target_columns[0]],
                    split_type='init_al',
                    n_samples_per_class=1,
                    seed=self.seed)
                # randomly select self.init_size - 2 samples from the pool set to be the training set
                if self.init_size > 2:
                    train_index.extend(np.random.choice(pool_index, self.init_size - 2, replace=False))
                    _ = []
                    for i in pool_index:
                        if i not in train_index:
                            _.append(i)
                    pool_index = _
        return train_index, pool_index

    def process_args(self) -> None:
        super().process_args()
        if self.data_public == 'freesolv' or self.data_public == 'test_regression':
            self.data_path = os.path.join(DATA_DIR, '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['freesolv']
            self.dataset_type = 'regression'
        elif self.data_public == 'delaney':
            self.data_path = os.path.join(DATA_DIR, '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['logSolubility']
            self.dataset_type = 'regression'
        elif self.data_public == 'lipo':
            self.data_path = os.path.join(DATA_DIR, '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['lipo']
            self.dataset_type = 'regression'
        elif self.data_public == 'pdbbind_refined':
            self.data_path = os.path.join(DATA_DIR, '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['-logKd/Ki']
            self.dataset_type = 'regression'
        elif self.data_public == 'pdbbind_full':
            self.data_path = os.path.join(DATA_DIR, '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['-logKd/Ki']
            self.dataset_type = 'regression'
        elif self.data_public in ['ld50_zhu', 'caco2_wang', 'solubility_aqsoldb', 'ppbr_az', 'vdss_lombardo',
                                  'Half_Life_Obach', 'Clearance_Hepatocyte_AZ']:
            self.data_path = os.path.join(DATA_DIR, '%s.csv' % self.data_public)
            self.pure_columns = ['Drug']
            self.target_columns = ['Y']
            self.dataset_type = 'regression'
        elif self.data_public == 'bbbp':
            self.data_path = os.path.join(DATA_DIR, '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['p_np']
            self.dataset_type = 'classification'
        elif self.data_public == 'bace':
            self.data_path = os.path.join(DATA_DIR, '%s.csv' % self.data_public)
            self.pure_columns = ['mol']
            self.target_columns = ['Class']
            self.dataset_type = 'classification'
        elif self.data_public == 'hiv':
            self.data_path = os.path.join(DATA_DIR, '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['HIV_active']
            self.dataset_type = 'classification'
        elif self.data_public in ['ames', 'carcinogens_lagunin', 'dili', 'herg', 'skin', 'hia_hou', 'pgp_broccatelli',
                                  'bioavailability_ma', 'clintox', 'bbb_martins', 'CYP1A2_Veith',
                                  'CYP2C9_Substrate_CarbonMangels', 'CYP2C9_Veith', 'CYP2C19_Veith',
                                  'CYP2D6_Substrate_CarbonMangels', 'CYP2D6_Veith', 'CYP3A4_Veith',
                                  'CYP3A4_Substrate_CarbonMangels']:
            self.data_path = os.path.join(DATA_DIR, '%s.csv' % self.data_public)
            self.pure_columns = ['Drug']
            self.target_columns = ['Y']
            self.dataset_type = 'classification'
        elif self.data_public == 'human_liver_microsome_stability':
            self.data_path = os.path.join(DATA_DIR, '%s.csv' % self.data_public)
            self.pure_columns = ['SMILES']
            self.target_columns = ['LOG HLM_CLint (mL/min/kg)']
            self.dataset_type = 'regression'
        elif self.data_public == 'rat_liver_microsome_stability':
            self.data_path = os.path.join(DATA_DIR, '%s.csv' % self.data_public)
            self.pure_columns = ['SMILES']
            self.target_columns = ['LOG RLM_CLint (mL/min/kg)']
            self.dataset_type = 'regression'
        elif self.data_public == 'MDRR1-MDCK_efflux_ratio':
            self.data_path = os.path.join(DATA_DIR, '%s.csv' % self.data_public)
            self.pure_columns = ['SMILES']
            self.target_columns = ['LOG MDR1-MDCK ER (B-A/A-B)']
            self.dataset_type = 'regression'
        elif self.data_public == 'aqueous_solubility':
            self.data_path = os.path.join(DATA_DIR, '%s.csv' % self.data_public)
            self.pure_columns = ['SMILES']
            self.target_columns = ['LOG SOLUBILITY PH 6.8 (ug/mL)']
            self.dataset_type = 'regression'
        elif self.data_public == 'human_plasma_protein_binding':
            self.data_path = os.path.join(DATA_DIR, '%s.csv' % self.data_public)
            self.pure_columns = ['SMILES']
            self.target_columns = ['LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)']
        elif self.data_public == 'rat_plasma_protein_binding':
            self.data_path = os.path.join(DATA_DIR, '%s.csv' % self.data_public)
            self.pure_columns = ['SMILES']
            self.target_columns = ['LOG PLASMA PROTEIN BINDING (RAT) (% unbound)']

        if self.split_type is not None and self.split_type.startswith('scaffold'):
            assert len(self.pure_columns) == 1
            assert self.mixture_columns is None

        assert len(self.target_columns) == 1

        if self.data_path is not None:
            # All data comes from the same file.
            assert self.data_path_val is None and self.data_path_training is None and self.data_path_pool is None
            df = pd.read_csv(self.data_path)
            if 'id' not in df:
                df['id'] = range(len(df))
            if self.full_val:
                # use the full dataset as validation set
                assert self.split_type is None
                assert self.split_sizes is None
                assert self.error_rate is None
                df.to_csv('%s/val.csv' % self.save_dir, index=False)
                df_al = df
            else:
                # split the dataset into active learning and validation sets
                al_index, val_index = data_split_index(
                    n_samples=len(df),
                    mols=df[self.pure_columns[0]] if self.pure_columns is not None else None,
                    # targets=df[self.target_columns[0]],
                    split_type=self.split_type,
                    sizes=self.split_sizes,
                    seed=self.seed,
                    logger=self.logger)
                df[df.index.isin(val_index)].to_csv('%s/val.csv' % self.save_dir, index=False)
                df_al = df[df.index.isin(al_index)].copy()
                if self.error_rate is not None:
                    # randomly select a portion of the training set to be affected by error
                    error_index = np.random.choice(al_index, int(self.error_rate * len(al_index)), replace=False)
                    df_al['flip_label'] = False
                    df_al.loc[error_index, self.target_columns[0]] ^= 1
                    df_al.loc[error_index, 'flip_label'] = True
            # split the active learning set into training and pool sets
            train_index, pool_index = self.get_train_pool_split_index(df_al)
            df_al.iloc[train_index].to_csv('%s/train_init.csv' % self.save_dir, index=False)
            df_al.iloc[pool_index].to_csv('%s/pool_init.csv' % self.save_dir, index=False)
        else:
            # data comes from 3 different files.
            assert self.data_path_training is not None, 'please provide input data'
            if self.data_path_pool is None:
                df_al = pd.read_csv(self.data_path_training)
                train_index, pool_index = self.get_train_pool_split_index(df_al)
                df_al.iloc[train_index].to_csv('%s/train_init.csv' % self.save_dir, index=False)
                df_al.iloc[pool_index].to_csv('%s/pool_init.csv' % self.save_dir, index=False)
            else:
                shutil.copyfile(self.data_path_training, '%s/train_init.csv' % self.save_dir)
                shutil.copyfile(self.data_path_pool, '%s/pool_init.csv' % self.save_dir)
            if self.data_path_val is None:
                pd.read_csv(self.data_path_training).sample(0).to_csv('%s/val.csv' % self.save_dir, index=False)
                df = pd.concat([pd.read_csv(f) for f in ['%s/train_init.csv' % self.save_dir,
                                                         '%s/pool_init.csv' % self.save_dir]])
            else:
                shutil.copyfile(self.data_path_val, '%s/val.csv' % self.save_dir)
                df = pd.concat([pd.read_csv(f) for f in ['%s/train_init.csv' % self.save_dir,
                                                         '%s/pool_init.csv' % self.save_dir,
                                                         '%s/val.csv' % self.save_dir]])
            if 'id' not in df:
                df['id'] = range(len(df))
                df_train = pd.read_csv('%s/train_init.csv' % self.save_dir)
                df_train['id'] = range(len(df_train))
                df_train.to_csv('%s/train_init.csv' % self.save_dir, index=False)
                df_pool = pd.read_csv('%s/pool_init.csv' % self.save_dir)
                df_pool['id'] = range(len(df_pool))
                df_pool['id'] += len(df_train)
                df_pool.to_csv('%s/pool_init.csv' % self.save_dir, index=False)
                df_val = pd.read_csv('%s/val.csv' % self.save_dir)
                df_val['id'] = range(len(df_val))
                df_val['id'] += len(df_train) + len(df_pool)
                df_val.to_csv('%s/val.csv' % self.save_dir, index=False)
            df.to_csv('%s/full.csv' % self.save_dir, index=False)
            self.data_path = '%s/full.csv' % self.save_dir


class ModelArgs(Tap):
    model_config_selector: str
    """config file contain all information of the machine learning model."""
    model_config_evaluators: List[str] = None
    """A list of config files contain all information of the machine learning model for performance evaluation."""

    @property
    def model_config_selector_dict(self) -> Dict:
        config_name = self.find_stored_model_config(self.model_config_selector)
        return json.loads(open(config_name).read())

    @property
    def model_config_evaluators_dict(self) -> List[Dict]:
        if self.model_config_evaluators is None:
            return []
        else:
            return [json.loads(open(self.find_stored_model_config(m)).read()) for m in self.model_config_evaluators]

    @staticmethod
    def find_stored_model_config(config_name):
        if os.path.exists(config_name):
            return config_name
        else:
            stored_config_name = f'{CWD}/models/configs/{config_name}'
            assert os.path.exists(stored_config_name), f'{config_name} is not a valid model configuration'
            return stored_config_name


class ActiveLearningArgs(DatasetArgs, ModelArgs):
    save_dir: str
    """the output directory."""
    n_jobs: int = 1
    """the cpu numbers used for parallel computing."""
    data_path: str = None
    """the Path of input data CSV file."""
    metrics: List[Metric] = None
    """the metrics to evaluate model performance."""
    evaluate_stride: int = None
    """evaluate model performance on the validation set when the size of the training set is an integer multiple of the 
    evaluation stride."""
    output_details: bool = False
    """output the details of each active learning iteration."""
    top_k: float = None
    """the ratio of top molecules are considered."""
    yoked_learning_only: bool = False
    """Only perform yoked learning."""
    learning_type: Literal['passive', 'explorative', 'exploitive']
    """the learning type to be performed."""
    exploitive_target: str = None
    """the target value for exploitive active learning."""
    init_size: int = 2
    """number of samples as the initial."""
    batch_size: int = 1
    """number of samples added in each active learning iteration."""
    batch_mode: Literal['naive', 'cluster'] = 'naive'
    """the method that add a batch of samples."""
    cluster_size: int = None
    """number of samples in each cluster. (default = 20 * batch_size)"""
    n_query: int = None
    """number of samples to query in each active learning iteration. (default=None means query all samples in the 
    pool set)"""
    n_pool: int = None
    """number of pool sets to be split for active learning iteration."""
    stop_ratio: float = None
    """Stop active learning when the selected molecules reach the ratio."""
    stop_size: int = None
    """Stop active learning when the selected molecules reach the number."""
    stop_cutoff: float = None
    """Stop active learning when the acquisition function reach the cutoff."""
    confidence_cutoff: float = None
    """Stop active learning when the confidence of the model reach the cutoff."""
    max_iter: int = None
    """the maximum number of iterations."""
    save_cpt_stride: int = None
    """save checkpoint file every no. steps of active learning iteration."""
    write_traj_stride: int = None
    """write trajectory file every no. steps of active learning iteration."""
    load_checkpoint: bool = False
    """load checkpoint file and continue the active learning."""
    # Arguments for forgetting active learning.
    forget_protocol: Literal['forget_first', 'forget_random', 'min_oob_uncertainty', 'max_oob_uncertainty',
    'min_oob_error', 'min_loo_error', 'MCDropoutForgetter'] = None
    """protocol to use (forget_first, forget_random, min_oob_uncertain (RF only), max_oob_uncertain (RF only)
    , min_loo_error, MCDropoutForgetter (DNN only))."""
    forget_cutoff: float = None
    """The error cutoff for forgetting ."""
    forget_size: int = None
    """the number of molecules in the training set to start forgetting data at."""
    forget_ratio: float = None
    """the percent of the full training set to start forgetting data."""

    @property
    def model_selector(self):
        if not hasattr(self, '_model_selector'):
            self._model_selector = get_model(
                data_format=self.model_config_selector_dict['data_format'],
                dataset_type=self.dataset_type,
                model=self.model_config_selector_dict.get('model'),
                save_dir='%s/selector' % self.save_dir,
                data_path=self.data_path,
                smiles_columns=self.pure_columns,
                target_columns=self.target_columns,
                loss_function=self.model_config_selector_dict.get('loss_function'),
                multiclass_num_classes=self.model_config_selector_dict.get('loss_function') or 3,
                features_generator=self.features_generator_selector,
                no_features_scaling=self.model_config_selector_dict.get('no_features_scaling') or False,
                features_only=self.model_config_selector_dict.get('features_only') or False,
                features_size=self.data_train_selector.features_size(),
                epochs=self.model_config_selector_dict.get('epochs') or 30,
                depth=self.model_config_selector_dict.get('depth') or 3,
                hidden_size=self.model_config_selector_dict.get('hidden_size') or 300,
                ffn_num_layers=self.model_config_selector_dict.get('ffn_num_layers') or 2,
                ffn_hidden_size=self.model_config_selector_dict.get('ffn_hidden_size'),
                dropout=self.model_config_selector_dict.get('dropout') or 0.0,
                batch_size=self.model_config_selector_dict.get('batch_size') or 50,
                ensemble_size=self.model_config_selector_dict.get('ensemble_size') or 1,
                number_of_molecules=self.model_config_selector_dict.get('number_of_molecules') or len(self.pure_columns),
                mpn_shared=self.model_config_selector_dict.get('mpn_shared') or False,
                atom_messages=self.model_config_selector_dict.get('atom_messages') or False,
                undirected=self.model_config_selector_dict.get('undirected') or False,
                class_balance=self.model_config_selector_dict.get('class_balance') or False,
                checkpoint_dir=self.model_config_selector_dict.get('checkpoint_dir'),
                checkpoint_frzn=self.model_config_selector_dict.get('checkpoint_frzn'),
                frzn_ffn_layers=self.model_config_selector_dict.get('frzn_ffn_layers') or 0,
                freeze_first_only=self.model_config_selector_dict.get('freeze_first_only') or False,
                mpn_path=self.model_config_selector_dict.get('mpn_path'),
                freeze_mpn=self.model_config_selector_dict.get('freeze_mpn') or False,
                continuous_fit=self.model_config_selector_dict.get('continuous_fit') or False,
                kernel=self.kernel_selector,
                uncertainty_type=self.model_config_selector_dict.get('uncertainty_type'),
                alpha=self.model_config_selector_dict.get('alpha'),
                C=self.model_config_selector_dict.get('C'),
                booster=self.model_config_selector_dict.get('booster'),
                n_estimators=self.model_config_selector_dict.get('n_estimators') or 100,
                max_depth=self.model_config_selector_dict.get('max_depth'),
                learning_rate=self.model_config_selector_dict.get('learning_rate') or 0.1,
                n_jobs=self.n_jobs,
                seed=self.seed,
                logger=self.logger)
        return self._model_selector

    @property
    def model_evaluators(self):
        if not hasattr(self, '_model_evaluators'):
            self._model_evaluators = [get_model(
                data_format=model_config['data_format'],
                dataset_type=self.dataset_type,
                model=model_config.get('model'),
                save_dir='%s/evaluator_%d' % (self.save_dir, i),
                data_path=self.data_path,
                smiles_columns=self.pure_columns,
                target_columns=self.target_columns,
                loss_function=model_config.get('loss_function'),
                multiclass_num_classes=model_config.get('loss_function') or 3,
                features_generator=self.features_generator_evaluators[i],
                no_features_scaling=model_config.get('no_features_scaling') or False,
                features_only=model_config.get('features_only') or False,
                features_size=self.data_train_evaluators[i].features_size(),
                epochs=model_config.get('epochs') or 30,
                depth=model_config.get('depth') or 3,
                hidden_size=model_config.get('hidden_size') or 300,
                ffn_num_layers=model_config.get('ffn_num_layers') or 2,
                ffn_hidden_size=model_config.get('ffn_hidden_size'),
                dropout=model_config.get('dropout') or 0.0,
                batch_size=model_config.get('batch_size') or 50,
                ensemble_size=model_config.get('ensemble_size') or 1,
                number_of_molecules=model_config.get('number_of_molecules') or len(self.pure_columns),
                mpn_shared=model_config.get('mpn_shared') or False,
                atom_messages=model_config.get('atom_messages') or False,
                undirected=model_config.get('undirected') or False,
                class_balance=model_config.get('class_balance') or False,
                checkpoint_dir=model_config.get('checkpoint_dir'),
                checkpoint_frzn=model_config.get('checkpoint_frzn'),
                frzn_ffn_layers=model_config.get('frzn_ffn_layers') or 0,
                freeze_first_only=model_config.get('freeze_first_only') or False,
                mpn_path=model_config.get('mpn_path'),
                freeze_mpn=model_config.get('freeze_mpn') or False,
                continuous_fit=model_config.get('continuous_fit') or False,
                kernel=self.kernel_evaluators[i],
                uncertainty_type=model_config.get('uncertainty_type'),
                alpha=model_config.get('alpha'),
                C=self.model_config_selector_dict.get('C'),
                booster=self.model_config_selector_dict.get('booster'),
                n_estimators=self.model_config_selector_dict.get('n_estimators') or 100,
                max_depth=self.model_config_selector_dict.get('max_depth'),
                learning_rate=self.model_config_selector_dict.get('learning_rate') or 0.1,
                n_jobs=self.n_jobs,
                seed=self.seed,
                logger=self.logger
            ) for i, model_config in enumerate(self.model_config_evaluators_dict)]
        return self._model_evaluators

    @property
    def data_train_selector(self):
        if not hasattr(self, '_data_train_selector'):
            df = pd.read_csv('%s/train_init.csv' % self.save_dir)
            self._data_train_selector = get_subset(self.data_full_selector, 
                                                   idx=df['id'].tolist(),
                                                   unique_data_idx=True)
            if self.error_rate is not None:
                for i, data in enumerate(self._data_train_selector.data):
                    data.targets = df[self.target_columns].to_numpy()[i:i+1]
        return self._data_train_selector

    @property
    def data_pool_selector(self):
        if not hasattr(self, '_data_pool_selector'):
            df = pd.read_csv('%s/pool_init.csv' % self.save_dir)
            self._data_pool_selector = get_subset(self.data_full_selector,
                                                  idx=df['id'].tolist(),
                                                  unique_data_idx=True)
            if self.error_rate is not None:
                for i, data in enumerate(self._data_pool_selector.data):
                    data.targets = df[self.target_columns].to_numpy()[i:i+1]
        return self._data_pool_selector

    @property
    def data_val_selector(self):
        if not hasattr(self, '_data_val_selector'):
            df = pd.read_csv('%s/val.csv' % self.save_dir)
            if len(df) == 0:
                return None
            self._data_val_selector = get_subset(self.data_full_selector,
                                                 idx=df['id'].tolist(),
                                                 unique_data_idx=True)
        return self._data_val_selector

    @property
    def data_train_evaluators(self):
        if not hasattr(self, '_data_train_evaluators'):
            self._data_train_evaluators = [get_subset(data,
                                                      idx=pd.read_csv('%s/train_init.csv' % self.save_dir)['id'].tolist(),
                                                      unique_data_idx=True) for data in self.data_full_evaluators]
        return self._data_train_evaluators

    @property
    def data_pool_evaluators(self):
        if not hasattr(self, '_data_pool_evaluators'):
            self._data_pool_evaluators = [get_subset(data,
                                                     idx=pd.read_csv('%s/pool_init.csv' % self.save_dir)['id'].tolist(),
                                                     unique_data_idx=True) for data in self.data_full_evaluators]
        return self._data_pool_evaluators

    @property
    def data_val_evaluators(self):
        if not hasattr(self, '_data_val_evaluators'):
            self._data_val_evaluators = [get_subset(data,
                                                    idx=pd.read_csv('%s/val.csv' % self.save_dir)['id'].tolist(),
                                                    unique_data_idx=True) for data in self.data_full_evaluators]
        return self._data_val_evaluators

    @property
    def data_full_selector(self):
        if not hasattr(self, '_data_full_selector'):
            self._data_full_selector = get_data(
                data_format=self.model_config_selector_dict['data_format'],
                path=self.data_path,
                pure_columns=self.pure_columns,
                mixture_columns=self.mixture_columns,
                target_columns=self.target_columns,
                feature_columns=self.feature_columns,
                features_generator=self.features_generator_selector,
                features_combination=self.model_config_selector_dict.get('features_combination'),
                graph_kernel_type=self.model_config_selector_dict.get('graph_kernel_type'),
                n_jobs=self.n_jobs)
        return self._data_full_selector

    @property
    def data_full_evaluators(self) -> List:
        if not hasattr(self, '_data_full_evaluators'):
            self._data_full_evaluators = [get_data(
                data_format=model_config['data_format'],
                path=self.data_path,
                pure_columns=self.pure_columns,
                mixture_columns=self.mixture_columns,
                target_columns=self.target_columns,
                feature_columns=self.feature_columns,
                features_generator=self.features_generator_evaluators[i],
                features_combination=model_config.get('features_combination'),
                graph_kernel_type=model_config.get('graph_kernel_type'),
                n_jobs=self.n_jobs) for i, model_config in enumerate(self.model_config_evaluators_dict)]
        return self._data_full_evaluators

    @property
    def features_generator_selector(self) -> Optional[List[FeaturesGenerator]]:
        fingerprints_class = self.model_config_selector_dict.get('fingerprints_class')
        radius = self.model_config_selector_dict.get('radius')
        num_bits = self.model_config_selector_dict.get('num_bits')
        atomInvariantsGenerator = self.model_config_selector_dict.get('atomInvariantsGenerator')
        if fingerprints_class is None:
            return None
        else:
            return [FeaturesGenerator(features_generator_name=fc,
                                      radius=radius,
                                      num_bits=num_bits,
                                      atomInvariantsGenerator=atomInvariantsGenerator) for fc in fingerprints_class]

    @property
    def features_generator_evaluators(self) -> Optional[List[List[FeaturesGenerator]]]:
        results = []
        for model_config in self.model_config_evaluators_dict:
            fingerprints_class = model_config.get('fingerprints_class')
            radius = model_config.get('radius')
            num_bits = model_config.get('num_bits')
            atomInvariantsGenerator = model_config.get('atomInvariantsGenerator')
            if fingerprints_class is None:
                results.append(None)
            else:
                results.append([FeaturesGenerator(features_generator_name=fc,
                                                  radius=radius,
                                                  num_bits=num_bits, 
                                                  atomInvariantsGenerator=atomInvariantsGenerator) for fc in fingerprints_class])
        return results

    @property
    def kernel_selector(self):
        return get_kernel(
            graph_kernel_type=self.model_config_selector_dict.get('graph_kernel_type'),
            mgk_files=self.model_config_selector_dict.get('mgk_files'),
            features_kernel_type=self.model_config_selector_dict.get('features_kernel_type'),
            features_hyperparameters=self.model_config_selector_dict.get('features_hyperparameters'),
            features_hyperparameters_file=self.model_config_selector_dict.get('features_hyperparameters_file'),
            dataset=self.data_full_selector,
            kernel_pkl_path='%s/kernel_selector.pkl' % self.save_dir,
        )

    @property
    def kernel_evaluators(self) -> List:
        if not hasattr(self, '_kernel_evaluators'):
            self._kernel_evaluators = [get_kernel(
                graph_kernel_type=model_config.get('graph_kernel_type'),
                mgk_files=model_config.get('mgk_files'),
                features_kernel_type=model_config.get('features_kernel_type'),
                features_hyperparameters=model_config.get('features_hyperparameters'),
                features_hyperparameters_file=model_config.get('features_hyperparameters_file'),
                dataset=self.data_full_evaluators[i],
                kernel_pkl_path='%s/kernel_evaluator_%d.pkl' % (self.save_dir, i),
            ) for i, model_config in enumerate(self.model_config_evaluators_dict)]
        return self._kernel_evaluators

    @property
    def selection_method(self):
        if not hasattr(self, '_selection_method'):
            if self.exploitive_target is not None and self.exploitive_target not in ['min', 'max']:
                self.exploitive_target = float(self.exploitive_target)
            if self.learning_type == 'passive':
                if self.batch_mode == 'cluster':
                    self._selection_method = ClusterRandomSelectionMethod(batch_size=self.batch_size,
                                                                          cluster_size=self.cluster_size,
                                                                          seed=self.seed)
                else:
                    self._selection_method = RandomSelectionMethod(seed=self.seed)
            elif self.learning_type == 'explorative':
                if self.batch_mode == 'cluster':
                    if self.n_query is not None:
                        self._selection_method = ClusterExplorativeParitialQuerySelectionMethod(
                            batch_size=self.batch_size,
                            cluster_size=self.cluster_size,
                            n_query=self.n_query,
                            seed=self.seed)
                    else:
                        self._selection_method = ClusterExplorativeSelectionMethod(batch_size=self.batch_size,
                                                                                   cluster_size=self.cluster_size,
                                                                                   seed=self.seed)
                else:
                    if self.n_query is not None:
                        self._selection_method = ExplorativeParitialQuerySelectionMethod(n_query=self.n_query,
                                                                                         seed=self.seed)
                    else:
                        self._selection_method = ExplorativeSelectionMethod(seed=self.seed)
            elif self.learning_type == 'exploitive':
                if self.batch_mode == 'cluster':
                    self._selection_method = ClusterExploitiveSelectionMethod(batch_size=self.batch_size,
                                                                              cluster_size=self.cluster_size,
                                                                              target=self.exploitive_target,
                                                                              seed=self.seed)
                else:
                    self._selection_method = ExploitiveSelectionMethod(target=self.exploitive_target, seed=self.seed)
            else:
                raise ValueError('Unknown learning type %s' % self.learning_type)
            self._selection_method.batch_size = self.batch_size
        return self._selection_method

    @property
    def forgetter(self):
        if not hasattr(self, '_forgetter'):
            if self.forget_protocol == 'forget_first':
                forgeter = FirstForgetter()
            elif self.forget_protocol == 'forget_random':
                forgeter = RandomForgetter(seed=0)
            elif self.forget_protocol == 'min_oob_uncertainty':
                forgeter = MinOOBUncertaintyForgetter(seed=0)
            elif self.forget_protocol == 'max_oob_uncertainty':
                forgeter = MaxOOBUncertaintyForgetter(seed=0)
            elif self.forget_protocol == 'min_oob_error':
                forgeter = MinOOBErrorForgetter(seed=0)
            elif self.forget_protocol == 'min_loo_error':
                forgeter = MinLOOErrorForgetter(seed=0)
            elif self.forget_protocol == 'MCDropoutForgetter':
                forgeter = MCDropoutForgetter()
            else:
                return None
            # set forget_size and forget_cutoff in forgetter.
            # get forget_size from forget_ratio
            if self.forget_ratio is not None:
                assert self.forget_size is None
                forgeter.forget_size = math.ceil(self.forget_ratio *
                                                 (len(self.data_train_selector) + len(self.data_pool_selector)))
            else:
                forgeter.forget_size = self.forget_size
            if self.forget_cutoff is not None:
                assert self.forget_size is None
                assert self.forget_protocol in ['max_oob_uncertainty', 'min_oob_uncertainty',
                                                'min_oob_error', 'min_loo_error']
            forgeter.forget_cutoff = self.forget_cutoff
            forgeter.batch_size = 1
            self._forgetter = forgeter
        return self._forgetter

    @property
    def top_k_id(self) -> Optional[List[int]]:
        if self.top_k is not None:
            assert 0. < self.top_k < 1., 'top_k must be in (0, 1).'
            assert len(self.target_columns) == 1
            n_top_k = math.ceil(self.top_k * (len(self.data_train_evaluators) + len(self.data_pool_selector)))
            y_AL = self.data_train_selector.y.ravel().tolist() + self.data_pool_selector.y.ravel().tolist()
            top_k_index = get_topn_idx(y_AL, n_top_k, target=self.exploitive_target)
            top_k_id = []
            for i, data in enumerate(self.data_train_selector.data + self.data_pool_selector.data):
                if i in top_k_index:
                    top_k_id.append(data.id)
            return top_k_id
        else:
            return None

    def process_args(self) -> None:
        super().process_args()
        if self.model_config_selector_dict.get('graph_kernel_type') == 'graph':
            if self.data_pool_selector is not None and len(self.data_pool_selector) != 0:
                self.data_train_selector.unify_datatype(self.data_pool_selector.X_graph)
            if self.data_val_selector is not None and len(self.data_val_selector) != 0:
                self.data_train_selector.unify_datatype(self.data_val_selector.X_graph)
        for i, model_config_evaluator_dict in enumerate(self.model_config_evaluators_dict):
            if model_config_evaluator_dict.get('graph_kernel_type') == 'graph':
                if self.data_pool_evaluators[i] is not None and len(self.data_pool_evaluators[i]) != 0:
                    self.data_train_evaluators[i].unify_datatype(self.data_pool_evaluators[i].X_graph)
                if self.data_val_evaluators[i] is not None and len(self.data_val_evaluators[i]) != 0:
                    self.data_train_evaluators[i].unify_datatype(self.data_val_evaluators[i].X_graph)
        # get stop_size from stop_ratio
        if self.stop_ratio is not None:
            if self.stop_size is None:
                self.stop_size = math.ceil(
                    self.stop_ratio * (len(self.data_train_selector) + len(self.data_pool_selector)))
            else:
                self.stop_size = min(
                    self.stop_size,
                    math.ceil(self.stop_ratio * (len(self.data_train_selector) + len(self.data_pool_selector))))
            assert self.stop_size >= 2
        # set the default maximum number of iterations of active learning
        if self.max_iter is None:
            self.max_iter = 1 if self.data_pool_selector is None else 1 + len(self.data_pool_selector)
        # check the input for exploitive active learning
        if self.learning_type == 'exploitive':
            # assert self.dataset_type == 'regression', 'exploitive active learning only support regression task.'
            # assert self.top_k is not None, 'top_k must be set for exploitive active learning.'
            assert self.exploitive_target is not None, 'exploitive_target must be set for exploitive active learning.'
        # check the input for forgetting active learning
        if self.forget_protocol is not None:
            assert (self.forgetter.forget_size is None) ^ (self.forgetter.forget_cutoff is None)
        # introduce error (Randomly flip the labels of a portion of the data) to training set
        if self.error_rate is not None:
            assert self.dataset_type == 'classification'
            assert 0. < self.error_rate <= 1.
            df = pd.read_csv('%s/train_init.csv' % self.save_dir)
            for i, data in enumerate(self.data_train_selector.data):
                if df.loc[i, 'flip_label'] is True:
                    data.flip_label = True
                else:
                    data.flip_label = False
            if self.data_pool_selector is not None:
                df = pd.read_csv('%s/pool_init.csv' % self.save_dir)
                for i, data in enumerate(self.data_pool_selector.data):
                    if df.loc[i, 'flip_label'] is True:
                        data.flip_label = True
                    else:
                        data.flip_label = False
        # check unique ID for the data sets.
        unique_id = []
        for data in self.data_train_selector.data:
            assert data.id not in unique_id
            unique_id.append(data.id)
        if self.data_pool_selector is not None:
            for data in self.data_pool_selector.data:
                assert data.id not in unique_id, f'{data.id}, {unique_id}'
                unique_id.append(data.id)
        if self.data_val_selector is not None:
            for data in self.data_val_selector:
                if self.full_val:
                    assert data.id in unique_id
                else:
                    assert data.id not in unique_id
                    unique_id.append(data.id)
        else:
            assert self.metrics is None
            assert self.evaluate_stride is None

        if self.stop_cutoff is not None:
            if self.confidence_cutoff is None:
                self.confidence_cutoff = self.stop_cutoff
            else:
                assert self.confidence_cutoff <= self.stop_cutoff

    def flip_labels(self, datasets, error_index):
        i = 0
        for dataset in datasets:
            for data in dataset:
                if i in error_index:
                    data.targets ^= 1
                    data.flip_label = True
                else:
                    data.flip_label = False
                i += 1


class ActiveLearningContinueArgs(CommonArgs):
    stop_ratio: float = None
    """the ratio of molecules to stop the active learning."""
    stop_size: int = None
    """the number of molecules to stop the active learning."""
    stop_cutoff: float = None
    """Stop active learning when the acquisition function reach the cutoff."""
    max_iter: int = 10000
    """the maximum number of iterations."""

    def process_args(self) -> None:
        super().process_args()
        # get stop_size from stop_ratio
        if self.stop_ratio is not None:
            if self.stop_size is None:
                self.stop_size = math.ceil(
                    self.stop_ratio * (len(self.data_train_selector) + len(self.data_pool_selector)))
            else:
                self.stop_size = min(
                    self.stop_size,
                    math.ceil(self.stop_ratio * (len(self.data_train_selector) + len(self.data_pool_selector))))
            assert self.stop_size >= 2


class ReEvaluateArgs(CommonArgs):
    model_config_evaluator: str
    """config file contain all information of the machine learning model for performance evaluation."""
    evaluator_id: int
    """the output id of the evaluator"""
    evaluate_stride: int = 100
    """evaluate model performance on the validation set when the size of the training set is an integer multiple of the 
    evaluation stride."""
    metrics: List[Metric]
    """the metrics to evaluate model performance."""
    data_public = None
    """Use public data sets."""
    data_path: str = None
    """the Path of input data CSV file."""
    pure_columns: List[str] = None
    """
    For pure compounds.
    Name of the columns containing single SMILES or InChI string.
    """
    mixture_columns: List[str] = None
    """
    For mixtures.
    Name of the columns containing multiple SMILES or InChI string and 
    corresponding concentration.
    example: ['C', 0.5, 'CC', 0.3]
    """
    target_columns: List[str] = None
    """
    Name of the columns containing target values.
    """
    feature_columns: List[str] = None
    """
    Name of the columns containing additional features_mol such as temperature, 
    pressuer.
    """
    dataset_type: Literal['regression', 'classification', 'multiclass'] = None
    """
    Type of task.
    """
    full_val: bool = False
    """validate the performance of active learning on the full dataset."""

    @property
    def model_config_evaluator_dict(self) -> Dict:
        return json.loads(open(self.model_config_evaluator).read())

    @property
    def model_evaluator(self):
        if not hasattr(self, '_model_evaluator'):
            self._model_evaluator = get_model(
                data_format=self.model_config_evaluator_dict['data_format'],
                dataset_type=self.dataset_type,
                model=self.model_config_evaluator_dict.get('model'),
                save_dir='%s/evaluator_%d' % (self.save_dir, self.evaluator_id),
                loss_function=self.model_config_evaluator_dict.get('loss_function'),
                num_tasks=len(self.target_columns),
                multiclass_num_classes=self.model_config_evaluator_dict.get('loss_function') or 3,
                features_generator=self.features_generator_evaluator,
                no_features_scaling=self.model_config_evaluator_dict.get('no_features_scaling') or False,
                features_only=self.model_config_evaluator_dict.get('features_only') or False,
                features_size=self.data_al_evaluator.features_size(),
                epochs=self.model_config_evaluator_dict.get('epochs') or 30,
                depth=self.model_config_evaluator_dict.get('depth') or 3,
                hidden_size=self.model_config_evaluator_dict.get('hidden_size') or 300,
                ffn_num_layers=self.model_config_evaluator_dict.get('ffn_num_layers') or 2,
                ffn_hidden_size=self.model_config_evaluator_dict.get('ffn_hidden_size'),
                dropout=self.model_config_evaluator_dict.get('dropout') or 0.0,
                batch_size=self.model_config_evaluator_dict.get('batch_size') or 50,
                ensemble_size=self.model_config_evaluator_dict.get('ensemble_size') or 1,
                number_of_molecules=self.model_config_evaluator_dict.get('number_of_molecules') or len(self.pure_columns),
                mpn_shared=self.model_config_evaluator_dict.get('mpn_shared') or False,
                atom_messages=self.model_config_evaluator_dict.get('atom_messages') or False,
                undirected=self.model_config_evaluator_dict.get('undirected') or False,
                class_balance=self.model_config_evaluator_dict.get('class_balance') or False,
                checkpoint_dir=self.model_config_evaluator_dict.get('checkpoint_dir'),
                checkpoint_frzn=self.model_config_evaluator_dict.get('checkpoint_frzn'),
                frzn_ffn_layers=self.model_config_evaluator_dict.get('frzn_ffn_layers') or 0,
                freeze_first_only=self.model_config_evaluator_dict.get('freeze_first_only') or False,
                kernel=self.kernel_evaluator,
                uncertainty_type=self.model_config_evaluator_dict.get('uncertainty_type'),
                alpha=self.model_config_evaluator_dict.get('alpha'),
                C=self.model_config_selector_dict.get('C'),
                n_jobs=self.n_jobs,
                seed=self.seed,
                logger=self.logger
            )
        return self._model_evaluator

    @property
    def data_al_evaluator(self):
        if not hasattr(self, '_data_al_evaluator'):
            self._data_al_evaluator = get_data(
                data_format=self.model_config_evaluator_dict['data_format'],
                path='%s/train_al.csv' % self.save_dir,
                pure_columns=self.pure_columns,
                mixture_columns=self.mixture_columns,
                target_columns=self.target_columns,
                feature_columns=self.feature_columns,
                features_generator=self.features_generator_evaluator,
                features_combination=self.model_config_evaluator_dict.get('features_combination'),
                graph_kernel_type=self.model_config_evaluator_dict.get('graph_kernel_type'),
                n_jobs=self.n_jobs)
        return self._data_al_evaluator

    @property
    def data_val_evaluator(self):
        if not hasattr(self, '_data_val_evaluator'):
            self._data_val_evaluator = get_data(
                data_format=self.model_config_evaluator_dict['data_format'],
                path='%s/val.csv' % self.save_dir,
                pure_columns=self.pure_columns,
                mixture_columns=self.mixture_columns,
                target_columns=self.target_columns,
                feature_columns=self.feature_columns,
                features_generator=self.features_generator_evaluator,
                features_combination=self.model_config_evaluator_dict.get('features_combination'),
                graph_kernel_type=self.model_config_evaluator_dict.get('graph_kernel_type'),
                n_jobs=self.n_jobs)
        return self._data_val_evaluator

    @property
    def data_full_evaluator(self):
        if not hasattr(self, '_data_full_evaluator'):
            self._data_full_evaluator = get_data(
                data_format=self.model_config_evaluator_dict['data_format'],
                path=self.data_path,
                pure_columns=self.pure_columns,
                mixture_columns=self.mixture_columns,
                target_columns=self.target_columns,
                feature_columns=self.feature_columns,
                features_generator=self.features_generator_evaluator,
                features_combination=self.model_config_evaluator_dict.get('features_combination'),
                graph_kernel_type=self.model_config_evaluator_dict.get('graph_kernel_type'),
                n_jobs=self.n_jobs)
        return self._data_full_evaluator

    @property
    def features_generator_evaluator(self) -> Optional[List[FeaturesGenerator]]:
        fingerprints_class = self.model_config_evaluator_dict.get('fingerprints_class')
        radius = self.model_config_evaluator_dict.get('radius')
        num_bits = self.model_config_evaluator_dict.get('num_bits')
        atomInvariantsGenerator = self.model_config_evaluator_dict.get('atomInvariantsGenerator')
        if fingerprints_class is None:
            return None
        else:
            return [FeaturesGenerator(features_generator_name=fc,
                                      radius=radius,
                                      num_bits=num_bits,
                                      atomInvariantsGenerator=atomInvariantsGenerator) for fc in fingerprints_class]

    @property
    def kernel_evaluator(self):
        return get_kernel(
            graph_kernel_type=self.model_config_evaluator_dict.get('graph_kernel_type'),
            mgk_files=self.model_config_evaluator_dict.get('mgk_files'),
            features_kernel_type=self.model_config_evaluator_dict.get('features_kernel_type'),
            features_hyperparameters=self.model_config_evaluator_dict.get('features_hyperparameters'),
            features_hyperparameters_file=self.model_config_evaluator_dict.get('features_hyperparameters_file'),
            dataset=self.data_full_evaluator,
            kernel_pkl_path='%s/kernel_evaluator_%d.pkl' % (self.save_dir, self.evaluator_id),
        )

    def process_args(self) -> None:
        super().process_args()
        if self.data_public == 'freesolv':
            self.data_path = os.path.join(DATA_DIR, '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['freesolv']
            self.dataset_type = 'regression'
        elif self.data_public == 'delaney':
            self.data_path = os.path.join(DATA_DIR, '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['logSolubility']
            self.dataset_type = 'regression'
        elif self.data_public == 'lipo':
            self.data_path = os.path.join(DATA_DIR, '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['lipo']
            self.dataset_type = 'regression'
        elif self.data_public == 'pdbbind_refined':
            self.data_path = os.path.join(DATA_DIR, '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['-logKd/Ki']
            self.dataset_type = 'regression'
        elif self.data_public == 'pdbbind_full':
            self.data_path = os.path.join(DATA_DIR, '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['-logKd/Ki']
            self.dataset_type = 'regression'
        elif self.data_public in ['ld50_zhu', 'caco2_wang', 'solubility_aqsoldb', 'ppbr_az', 'vdss_lombardo',
                                  'Half_Life_Obach', 'Clearance_Hepatocyte_AZ']:
            self.data_path = os.path.join(DATA_DIR, '%s.csv' % self.data_public)
            self.pure_columns = ['Drug']
            self.target_columns = ['Y']
            self.dataset_type = 'regression'
        elif self.data_public == 'bbbp':
            self.data_path = os.path.join(DATA_DIR, '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['p_np']
            self.dataset_type = 'classification'
        elif self.data_public == 'bace':
            self.data_path = os.path.join(DATA_DIR, '%s.csv' % self.data_public)
            self.pure_columns = ['mol']
            self.target_columns = ['Class']
            self.dataset_type = 'classification'
        elif self.data_public == 'hiv':
            self.data_path = os.path.join(DATA_DIR, '%s.csv' % self.data_public)
            self.pure_columns = ['smiles']
            self.target_columns = ['HIV_active']
            self.dataset_type = 'classification'
        elif self.data_public in ['ames', 'carcinogens_lagunin', 'dili', 'herg', 'skin', 'hia_hou', 'pgp_broccatelli',
                                  'bioavailability_ma', 'clintox', 'bbb_martins', 'CYP1A2_Veith',
                                  'CYP2C9_Substrate_CarbonMangels', 'CYP2C9_Veith', 'CYP2C19_Veith',
                                  'CYP2D6_Substrate_CarbonMangels', 'CYP2D6_Veith', 'CYP3A4_Veith',
                                  'CYP3A4_Substrate_CarbonMangels']:
            self.data_path = os.path.join(DATA_DIR, '%s.csv' % self.data_public)
            self.pure_columns = ['Drug']
            self.target_columns = ['Y']
            self.dataset_type = 'classification'
