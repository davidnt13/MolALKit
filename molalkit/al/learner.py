#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple, Callable
import os
import json
import pickle
import shutil
import copy
import pandas as pd
import numpy as np
from logging import Logger
from sklearn.metrics import *
import scipy
from functools import cached_property
from molalkit.models.mpnn.mpnn import MPNN, TrainArgs
from ..args import Metric
from .selection_method import BaseSelectionMethod, RandomSelectionMethod, get_subset
from .forgetter import BaseForgetter, RandomForgetter, FirstForgetter


def eval_metric_func(y, y_pred, metric: str) -> float:
    if metric == 'roc_auc':
        return roc_auc_score(y, y_pred)
    elif metric == 'accuracy':
        y_pred = [1 if i >= 0.5 else 0 for i in y_pred]
        return accuracy_score(y, y_pred)
    elif metric == 'precision':
        y_pred = [1 if i >= 0.5 else 0 for i in y_pred]
        return precision_score(y, y_pred, average='macro')
    elif metric == 'recall':
        y_pred = [1 if i >= 0.5 else 0 for i in y_pred]
        return recall_score(y, y_pred, average='macro')
    elif metric == 'f1_score':
        y_pred = [1 if i >= 0.5 else 0 for i in y_pred]
        return f1_score(y, y_pred, average='macro')
    elif metric == 'mcc':
        y_pred = [1 if i >= 0.5 else 0 for i in y_pred]
        return matthews_corrcoef(y, y_pred)
    elif metric == 'r2':
        return r2_score(y, y_pred)
    elif metric == 'mae':
        return mean_absolute_error(y, y_pred)
    elif metric == 'mse':
        return mean_squared_error(y, y_pred)
    elif metric == 'rmse':
        return mean_squared_error(y, y_pred, squared=False)
    elif metric == 'max':
        return np.max(abs(y - y_pred))
    elif metric == 'spearman':
        return scipy.stats.spearmanr(y, y_pred)[0]
    elif metric == 'kendall':
        return scipy.stats.kendalltau(y, y_pred)[0]
    elif metric == 'pearson':
        return scipy.stats.pearsonr(y, y_pred)[0]
    else:
        raise RuntimeError(f'Unsupported metrics {metric}')


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class ActiveLearningResult:
    def __init__(self, n_iter: int):
        self.n_iter = n_iter
        self.results = dict()


class ActiveLearningTrajectory:
    def __init__(self, metrics: List[str] = None, top_k_score: bool = False):
        self.metrics = metrics
        self.top_k_score = top_k_score
        self.results = []

    def is_valid_column(self, column_name: str) -> Tuple[bool, List]:
        results = [alr.results.get(column_name) for alr in self.results]
        for result in results:
            if isinstance(result, float):
                return True, results
        return False, results

    def get_results(self) -> Dict:
        results_dict = dict()
        results_dict['n_iter'] = [alr.n_iter for alr in self.results]
        if self.metrics is not None:
            for metric in self.metrics:
                is_valid, results = self.is_valid_column(f'{metric}_selector')
                if is_valid:
                    results_dict[f'{metric}_selector'] = results
                for i in range(100):
                    is_valid, results = self.is_valid_column(f'{metric}_evaluator_{i}')
                    if is_valid:
                        results_dict[f'{metric}_evaluator_{i}'] = results
                    else:
                        break
        if self.top_k_score:
            is_valid, results = self.is_valid_column('top_k_score')
            if is_valid:
                results_dict['top_k_score'] = results
        results_dict['id_add'] = [json.dumps(alr.id_add, cls=NpEncoder) for alr in self.results]
        results_dict['acquisition_add'] = [json.dumps(alr.acquisition_add, cls=NpEncoder) for alr in self.results]
        results_dict['id_forget'] = [json.dumps(alr.id_forget, cls=NpEncoder) for alr in self.results]
        results_dict['acquisition_forget'] = [json.dumps(alr.acquisition_forget,
                                                         cls=NpEncoder) for alr in self.results]
        results_dict['id_prior_al'] = [json.dumps(alr.id_prior_al, cls=NpEncoder) for alr in self.results]
        return results_dict


class ActiveLearner:
    def __init__(self, save_dir: str, selection_method: BaseSelectionMethod, forgetter: BaseForgetter,
                 model_selector, dataset_train_selector, dataset_pool_selector, dataset_val_selector,
                 metrics: List[Metric], top_k_id: List[int] = None,
                 model_evaluators=None, dataset_train_evaluators=None,
                 dataset_pool_evaluators=None, dataset_val_evaluators=None,
                 yoked_learning_only: bool = False,
                 stop_size: int = None, stop_cutoff: float = None, confidence_cutoff: float = None,
                 n_query: int = None, n_pool: int = None,
                 evaluate_stride: int = None, output_details: bool = False, kernel: Callable = None,
                 save_cpt_stride: int = None, write_traj_stride: int = None,
                 seed: int = 0,
                 logger: Logger = None):
        self.save_dir = save_dir
        self.selection_method = selection_method
        self.forgetter = forgetter
        self.model_selector = model_selector
        self.dataset_train_selector = dataset_train_selector
        self.dataset_pool_selector = dataset_pool_selector
        self.dataset_val_selector = dataset_val_selector
        self.metrics = metrics
        self.top_k_id = top_k_id
        self.model_evaluators = model_evaluators or []
        self.dataset_train_evaluators = dataset_train_evaluators or []
        self.dataset_pool_evaluators = dataset_pool_evaluators or []
        self.dataset_val_evaluators = dataset_val_evaluators or []
        self.yoked_learning_only = yoked_learning_only
        self.stop_size = stop_size
        self.stop_cutoff = stop_cutoff
        self.confidence_cutoff = confidence_cutoff
        self.n_query = n_query
        self.pools_uid = [data.id for data in self.dataset_pool_selector] 
        if n_pool is not None and n_pool > 1:
            # splits the large pool set into small pool sets to accelerate learning process.
            np.random.seed(seed)
            shuffled_uid = np.random.permutation(self.pools_uid)
            self.pools_uid = np.array_split(shuffled_uid, n_pool)
        else:
            self.pools_uid = [self.pools_uid]
        self.evaluate_stride = evaluate_stride
        self.output_details = output_details
        self.kernel = kernel  # used for cluster selection method
        self.save_cpt_stride = save_cpt_stride
        self.write_traj_stride = write_traj_stride

        self.seed = seed
        if logger is not None:
            self.info = logger.info
        else:
            self.info = print
        self.current_iter = 0
        self.active_learning_traj = ActiveLearningTrajectory(metrics=self.metrics,
                                                             top_k_score=self.top_k_id is not None)

    @property
    def train_size(self) -> int:
        return len(self.dataset_train_selector)

    @property
    def val_size(self) -> int:
        return 0 if self.dataset_val_selector is None else len(self.dataset_val_selector)

    @cached_property
    def uid2data_selector(self) -> Dict:
        d_train = {data.id: data for data in self.dataset_train_selector}
        d_pool = {data.id: data for data in self.dataset_pool_selector}
        return {**d_train, **d_pool}
    
    @cached_property
    def uid2data_evaluators(self) -> List[Dict]:
        dict_list = []
        for i in range(len(self.dataset_train_evaluators)):
            d_train = {data.id: data for data in self.dataset_train_evaluators[i]}
            d_pool = {data.id: data for data in self.dataset_pool_evaluators[i]}
            dict_list.append({**d_train, **d_pool})
        return dict_list

    def termination(self) -> bool:
        """ Termination condition for the entire active learning. """
        if (self.stop_size is not None and self.train_size >= self.stop_size) or len(self.dataset_pool_selector) == 0:
            return True
        else:
            return False

    def termination_pool(self, dataset_pool) -> bool:
        """ Termination condition for active learning for a specific pool set.

        Parameters
        ----------
        dataset_pool: pool set to run active learning.

        Returns
        ----------
        bool: True if the termination condition is satisfied else False.
        """
        if len(dataset_pool) == 0:
            # self.info('Terminating active learning: pool size = 0')
            return True
        elif self.stop_cutoff is not None and len(self.active_learning_traj.results) > 0:
            acquisition = self.active_learning_traj.results[-1].acquisition_add
            if not hasattr(self.selection_method, 'target') or self.selection_method.target == 'max':
                if self.n_query is None and (len(acquisition) == 0 or np.min(acquisition) <= self.stop_cutoff):
                    self.info(f'Terminating active learning: acquisition < stop_cutoff {self.stop_cutoff}')
                    return True
                else:
                    return False
            elif self.selection_method.target == 'min':
                if  self.n_query is None and (len(acquisition) == 0 or np.max(acquisition) >= self.stop_cutoff):
                    self.info(f'Terminating active learning: acquisition > stop_cutoff {self.stop_cutoff}')
                    return True
                else:
                    return False
            else:
                if self.n_query is None and (len(acquisition) == 0 or np.max(np.abs(np.array(acquisition) - self.selection_method.target)) >= self.stop_cutoff):
                    self.info(f'Terminating active learning: abs(acquisition - target) > stop_cutoff {self.stop_cutoff}')
                    return True
        else:
            return False

    def run(self, max_iter: int = None):
        # active learning overview information
        info = f'Start active learning: \n' \
               f'Training set size = {self.train_size}\n' \
               f'Pool set size = {len(self.dataset_pool_selector)}\n' \
               f'Validation set size = {self.val_size}\n' \
               f'selection method: {self.selection_method.info}\n'
        if self.forgetter is not None:
            info += f'Forgetter: {self.forgetter.info}'
        self.info(info)
        # start active learning
        if not hasattr(self, "current_pool_idx"):
            self.current_pool_idx = 0
        for i, pool_uid in enumerate(self.pools_uid):
            if i < self.current_pool_idx:
                continue
            if not hasattr(self, "dataset_pool_selector_"):
                self.dataset_pool_selector_ = get_subset(self.dataset_pool_selector, idx=pool_uid, unique_data_idx=True)
            terminated_tag, alr = self.run_active_learning(max_iter=max_iter)
            del self.dataset_pool_selector_
            self.current_pool_idx += 1
            if terminated_tag in [1, 3]:
                break
        if terminated_tag in [1, 2]:
            # save the results of the last frame of AL
            alr.id_add = alr.acquisition_add = alr.id_forget = alr.acquisition_forget = []
            self.active_learning_traj.results.append(alr)
        # write output files
        self.write_traj()
        if self.save_cpt_stride:
            self.save(path=self.save_dir, overwrite=True)

    def run_active_learning(self, max_iter: int) -> bool:
        is_first_step = True
        self.model_fitted = False
        for n_iter in range(self.current_iter, max_iter):
            alr = ActiveLearningResult(n_iter)
            alr.id_prior_al = [data.id for data in self.dataset_train_selector]
            self.info('Start an new iteration of active learning: %d.' % self.current_iter)
            # evaluate
            if self.evaluate_stride is not None and n_iter % self.evaluate_stride == 0:
                self.evaluate(alr)
            # check termination condition
            if self.termination():
                return 1, alr
            if not is_first_step and self.termination_pool(self.dataset_pool_selector_):
                return 2, alr
            # add sample
            self.add_samples(alr)
            # forget sample
            self.forget_samples(alr)

            self.current_iter += 1
            self.info('Training set size = %i' % self.train_size)
            self.info('Pool set size = %i' % len(self.dataset_pool_selector_))
            self.active_learning_traj.results.append(alr)
            if self.write_traj_stride is not None and n_iter % self.write_traj_stride == 0:
                self.write_traj()
            # save checkpoint file
            if self.save_cpt_stride is not None and n_iter % self.save_cpt_stride == 0:
                self.save(path=self.save_dir, filename='al_temp.pkl', overwrite=True)
                shutil.move(os.path.join(self.save_dir, 'al_temp.pkl'), os.path.join(self.save_dir, 'al.pkl'))
                self.info('save checkpoint file %s/al.pkl' % self.save_dir)
            is_first_step = False
        return 3, alr
    """
    def run_prospective(self):
        self.model_fitted = False
        n_iter = self.current_iter
        alr = ActiveLearningResult(n_iter)
        alr.id_prior_al = [data.id for data in self.dataset_train_selector]
        if self.termination():
            return None
        # add sample
        self.add_samples(alr)
        # forget sample
        self.forget_samples(alr)
        self.current_iter += 1
        self.active_learning_traj.results.append(alr)
        return alr
    """
    def evaluate(self, alr: ActiveLearningResult):
        self.info('evaluating model performance.')
        # conventional active learning.
        if not self.yoked_learning_only:
            # evaluate the prediction performance of ML model on the validation set
            if self.metrics is not None:
                if not self.model_fitted:
                    self.model_selector.fit_molalkit(self.dataset_train_selector)
                    self.model_fitted = True
                y_pred = self.model_selector.predict_value(self.dataset_val_selector)
                if self.output_details:
                    pd.DataFrame({'true': self.dataset_val_selector.y.ravel(), 'pred': y_pred}).to_csv(
                        os.path.join(self.save_dir, f'selector_{self.current_iter}.csv'), index=False)
                for metric in self.metrics:
                    metric_value = eval_metric_func(self.dataset_val_selector.y.ravel(), y_pred, metric=metric)
                    alr.results[f'{metric}_selector'] = metric_value
            # evaluate the percentage of top-k data selected in the training set
            if self.top_k_id is not None:
                alr.results['top_k_score'] = self.get_top_k_score(self.dataset_train_selector, self.top_k_id)
        # yoked learning
        for i, model in enumerate(self.model_evaluators):
            if self.metrics is not None:
                model.fit_molalkit(self.dataset_train_evaluators[i])
                y_pred = model.predict_value(self.dataset_val_evaluators[i])
                if self.output_details:
                    pd.DataFrame({'true': self.dataset_val_selector.y.ravel(), 'pred': y_pred}).to_csv(
                        os.path.join(self.save_dir, f'evaluator_{i}_{self.current_iter}.csv'), index=False)
                for metric in self.metrics:
                    metric_value = eval_metric_func(self.dataset_val_evaluators[i].y.ravel(), y_pred, metric=metric)
                    alr.results[f'{metric}_evaluator_{i}'] = metric_value
        self.info('evaluating model performance finished.')

    def add_samples(self, alr: ActiveLearningResult):
        # train the model if it is not trained in the evaluation step, and the selection method is not random.
        if not self.model_fitted and not isinstance(self.selection_method, RandomSelectionMethod):
            self.model_selector.fit_molalkit(self.dataset_train_selector)
        selected_idx, acquisition, remain_idx = self.selection_method(model=self.model_selector,
                                                                      data_train=self.dataset_train_selector,
                                                                      data_pool=self.dataset_pool_selector_,
                                                                      kernel=self.kernel,
                                                                      stop_cutoff=self.stop_cutoff,
                                                                      confidence_cutoff=self.confidence_cutoff)
        alr.id_add = [self.dataset_pool_selector_.data[i].id for i in selected_idx]
        alr.acquisition_add = acquisition
        # transfer data from pool to train.
        for uid in alr.id_add:
            self.dataset_train_selector.data.append(self.uid2data_selector[uid])
            for j in range(len(self.model_evaluators)):
                self.dataset_train_evaluators[j].data.append(self.uid2data_evaluators[j][uid])
        self.dataset_pool_selector_ = get_subset(self.dataset_pool_selector_, remain_idx)
        # set the model unfitted because new data is added.
        self.model_fitted = False

    def forget_samples(self, alr: ActiveLearningResult):
        # or forget algorithm is applied when self.forgetter.forget_size <= self.train_size.
        if self.forgetter is None or (self.forgetter.forget_size is not None and
                                      self.forgetter.forget_size > self.train_size):
            alr.id_forget = []
            alr.acquisition_forget = []
            return
        # train the model if the forgetter is not random or first.
        if not self.model_fitted and not self.forgetter.__class__ in [RandomForgetter, FirstForgetter]:
            self.model_selector.fit_molalkit(self.dataset_train_selector)
        # forget algorithm is applied.
        forget_idx, acquisition = self.forgetter(model=self.model_selector,
                                                 data=self.dataset_train_selector,
                                                 batch_size=self.forgetter.batch_size,
                                                 cutoff=self.forgetter.forget_cutoff)
        if forget_idx:
            alr.id_forget = [self.dataset_train_selector.data[i].id for i in forget_idx]
            alr.acquisition_forget = acquisition
            # transfer data from train to pool.
            for i in sorted(forget_idx, reverse=True):
                self.dataset_pool_selector_.data.append(self.dataset_train_selector.data.pop(i))
        else:
            alr.id_forget = []
            alr.acquisition_forget = []

    def write_traj(self):
        df_traj = pd.DataFrame(self.active_learning_traj.get_results())
        df_traj.to_csv(os.path.join(self.save_dir, 'al_traj.csv'), index=False)

    @staticmethod
    def get_top_k_score(dataset, top_k_id) -> float:
        N_top_k = 0
        for data in dataset:
            if data.id in top_k_id:
                N_top_k += 1
        return N_top_k / len(top_k_id)

    def save(self, path, filename='al.pkl', overwrite=False):
        f_al = os.path.join(path, filename)
        if os.path.isfile(f_al) and not overwrite:
            raise RuntimeError(
                f'Path {f_al} already exists. To overwrite, set '
                '`overwrite=True`.'
            )
        store = self.__dict__.copy()
        # TrainArgs is unpicklable, transform into dict.
        for model in ([store['model_selector']] + store['model_evaluators']):
            if isinstance(model, MPNN):
                model.args = model.args.as_dict()
        pickle.dump(store, open(f_al, 'wb'), protocol=4)
        # transform back to TrainArgs
        for model in ([store['model_selector']] + store['model_evaluators']):
            if isinstance(model, MPNN):
                model.args = TrainArgs().from_dict(model.args, skip_unsettable=True)

    @classmethod
    def load(cls, path, filename='al.pkl'):
        f_al = os.path.join(path, filename)
        store = pickle.load(open(f_al, 'rb'))
        for model in ([store['model_selector']] + store['model_evaluators']):
            if isinstance(model, MPNN):
                model.args = TrainArgs().from_dict(model.args, skip_unsettable=True)
        input = {}
        for key in ['save_dir', 'selection_method', 'forgetter', 'model_selector',
                    'dataset_train_selector', 'dataset_pool_selector', 'dataset_val_selector', 'metrics']:
            input[key] = store[key]
        learner = cls(**input)
        learner.__dict__.update(**store)
        return learner
