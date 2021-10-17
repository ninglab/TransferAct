
from collections import defaultdict
import csv
from logging import Logger
import os
import sys
from typing import Callable, Dict, List, Tuple

import random
import numpy as np
import pandas as pd

# import necessary functions to run this file
from aada.train.run_training import run_training
from aada.utils.args import TrainArgs
from aada.utils.constants import TEST_SCORES_FILE_NAME, TRAIN_LOGGER_NAME
from aada.data import get_data, get_task_names, MoleculeDataset, filter_common_smiles
from aada.utils.utils import create_logger, makedirs, timeit
from aada.features import set_extra_atom_fdim

@timeit(logger_name=TRAIN_LOGGER_NAME)
def cross_validate(args):
	""" 
	"""
	logger = create_logger(name='temp', save_dir=args.save_dir)
	if logger is not None:
		debug, info = logger.debug, logger.info
	else:
		debug = info = print

	# Print command line
	debug('Command line')
	debug(f'python {" ".join(sys.argv)}')

	# Print args
	debug('Args')
	debug(args)

	# for reproducibility
	random.seed(args.seed)
	np.random.seed(args.seed)

	# Initialize relevant variables
	init_seed = args.seed
	save_dir  = args.save_dir
	args.task_names = get_task_names(path=args.target_data_path, smiles_columns=args.smiles_columns,
				target_columns=args.target_columns, ignore_columns=args.ignore_columns)

	# Save args
	makedirs(args.save_dir)
	args.save(os.path.join(args.save_dir, 'args.json'))

	# load data from files
	source_data = get_data(path=args.source_data_path, args=args,
				logger=logger, skip_none_targets=True)

	target_data = get_data(path=args.target_data_path, args=args,
                                logger=logger, skip_none_targets=True)


	# filter out common SMILES across dataset
	# source_data, target_data = filter_common_smiles(source_data, target_data)

	# Run training for each fold, if #fold=1 => normal training
	all_scores = defaultdict(list)

	for fold_num in range(args.num_folds):
		info(f'Fold {fold_num}')
		#args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
		#makedirs(args.save_dir)
		model_scores = run_training(args, source_data, target_data, logger)

		for metric, scores in model_scores.items():
			all_scores[metric].append(scores)


	# convert scores to numpy arrays
	for metric, scores in all_scores.items():
		all_scores[metric] = np.array(scores)

	# Report results
	info(f'{args.num_folds}-fold cross validation')

	# Report scores for each fold
	for fold_num in range(args.num_folds):
		for metric, scores in all_scores.items():
			info(f'\tSeed {init_seed + fold_num} ==> test {metric}\
				= {np.nanmean(scores[fold_num]):.6f}')

			if args.show_individual_scores:
				for task_name, score in zip(args.task_names, scores[fold_num]):
					info(f'\t\tSeed {init_seed + fold_num} \
	  					==> test {task_name} {metric} = {score:.6f}')

	# Report scores across folds
	for metric, scores in all_scores.items():
		avg_scores = np.nanmean(scores, axis=1)  # average score for each model across tasks
		mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
		info(f'Overall test {metric} = {mean_score:.6f} +/- {std_score:.6f}')

		if args.show_individual_scores:
			for task_num, task_name in enumerate(args.task_names):
				info(f'\tOverall test {task_name} {metric} = '
				f'{np.nanmean(scores[:, task_num]):.6f} +/- {np.nanstd(scores[:, task_num]):.6f}')

	# Save scores
	with open(os.path.join(save_dir, TEST_SCORES_FILE_NAME), 'w') as f:
		writer = csv.writer(f)

		header = ['Task']
		for metric in args.metrics:
			header += [f'Mean {metric}', f'Standard deviation {metric}'] + \
				[f'Fold {i} {metric}' for i in range(args.num_folds)]
		writer.writerow(header)

		for task_num, task_name in enumerate(args.task_names):
			row = [task_name]
			for metric, scores in all_scores.items():
				task_scores = scores[:, task_num]
				mean, std = np.nanmean(task_scores), np.nanstd(task_scores)
				row += [mean, std] + task_scores.tolist()
			writer.writerow(row)

	# Determine mean and std score of main metric
	avg_scores = np.nanmean(all_scores[args.metric], axis=1)
	mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)

	# Optionally merge and save test preds
	if args.save_preds:
		all_preds = pd.concat([pd.read_csv(os.path.join(save_dir, f'fold_{fold_num}', 'test_preds.csv'))
			for fold_num in range(args.num_folds)])
		all_preds.to_csv(os.path.join(save_dir, 'test_preds.csv'), index=False)

	return mean_score, std_score



if __name__ == '__main__':
	""" Parses aada training arguments and trains/cross-validate
	"""
	cross_validate(args=TrainArgs().parse_args())

