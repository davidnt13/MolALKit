#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path
import time
from molalkit.args import ActiveLearningArgs, ActiveLearningContinueArgs, ReEvaluateArgs
from molalkit.al.learner import ActiveLearner


def molalkit_run(arguments=None):
    # read args.
    args = ActiveLearningArgs().parse_args(arguments)
    # active learning
    args.logger.info('Start a new active learning run.')
    start = time.time()
    if args.load_checkpoint and os.path.exists('%s/al.pkl' % args.save_dir):
        args.logger.info('continue active learning using checkpoint file %s/al.pkl' % args.save_dir)
        active_learner = ActiveLearner.load(path=args.save_dir)
    else:
        active_learner = ActiveLearner(save_dir=args.save_dir,
                                       selection_method=args.selection_method,
                                       forgetter=args.forgetter,
                                       model_selector=args.model_selector,
                                       dataset_train_selector=args.data_train_selector,
                                       dataset_pool_selector=args.data_pool_selector,
                                       dataset_val_selector=args.data_val_selector,
                                       metrics=args.metrics,
                                       top_k_id=args.top_k_id,
                                       model_evaluators=args.model_evaluators,
                                       dataset_train_evaluators=args.data_train_evaluators,
                                       dataset_pool_evaluators=args.data_pool_evaluators,
                                       dataset_val_evaluators=args.data_val_evaluators,
                                       yoked_learning_only=args.yoked_learning_only,
                                       stop_size=args.stop_size,
                                       stop_cutoff=args.stop_cutoff,
                                       confidence_cutoff=args.confidence_cutoff,
                                       n_query=args.n_query, n_pool=args.n_pool,
                                       evaluate_stride=args.evaluate_stride,
                                       write_traj_stride=args.write_traj_stride,
                                       output_details=args.output_details,
                                       kernel=args.kernel_selector,
                                       save_cpt_stride=args.save_cpt_stride,
                                       seed=args.seed,
                                       logger=args.logger)
    active_learner.shadow_cold_every = args.shadow_cold_every
    active_learner.shadow_cold_epochs = args.shadow_cold_epochs
    active_learner.shadow_cold_batch_size = args.shadow_cold_batch_size
    active_learner.run(max_iter=args.max_iter)
    end = time.time()
    args.logger.info('total time: %d s' % (end - start))
    return active_learner


def molalkit_run_from_cpt(arguments=None):
    # read args.
    args = ActiveLearningContinueArgs().parse_args(arguments)
    # active learning
    args.logger.info('Start a new active learning run.')
    start = time.time()
    args.logger.info('continue active learning using checkpoint file %s/al.pkl' % args.save_dir)
    active_learner = ActiveLearner.load(path=args.save_dir)
    assert len(active_learner.pools_uid) == 1
    active_learner.current_pool_idx = 0
    # change stop_size.
    active_learner.stop_size = args.stop_size
    active_learner.stop_cutoff = args.stop_cutoff
    active_learner.run(max_iter=args.max_iter)
    end = time.time()
    args.logger.info('total time: %d s' % (end - start))
    return active_learner


def run_eval():
    pass
    # read args.
    """
    args = ReEvaluateArgs().parse_args()
    args.logger.info('Start a new active learning run.')
    data_train = args.data_al_evaluator.copy()
    active_learning_traj_dict = {'training_size': []}
    for metric in args.metrics:
        active_learning_traj_dict[metric] = []
    n_list = list(range(args.evaluate_stride, len(args.data_al_evaluator), args.evaluate_stride))
    if n_list[-1] != len(data_train):
        n_list.append(len(data_train))
    for n in tqdm(n_list):
        active_learning_traj_dict['training_size'].append(n)
        data_train.data = args.data_al_evaluator.data[:n]
        args.model_evaluator.fit(data_train)
        y_pred = args.model_evaluator.predict_value(args.data_val_evaluator)
        for metric in args.metrics:
            metric_value = eval_metric_func(args.data_val_evaluator.y, y_pred, metric=metric)
            active_learning_traj_dict[metric].append(metric_value)
    pd.DataFrame(active_learning_traj_dict).to_csv(
        '%s/active_learning_extra_%d.traj' % (args.save_dir, args.evaluator_id), index=False)
    """
