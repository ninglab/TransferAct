import os
import sys
import csv
import json
import pickle
import numpy as np

from aada.train import evaluate_predictions
from aada.data import get_data, split_data
from aada.utils.args import PredictArgs, TrainArgs
from aada.utils.utils import load_args, load_checkpoint, load_scalers, makedirs, timeit

def evaluate(args, test_preds, test_targets, metrics):
	test_scores = evaluate_predictions(
		preds=test_preds,
		targets=test_targets,
		num_tasks=1,
		metrics=metrics,
		dataset_type=args.dataset_type,
		logger=None
	)

	with open(os.path.join(os.path.dirname(args.save_dir), '_test_scores.csv'),'w') as f:

		f.write('Task')
		for metric in metrics:
			f.write(',Mean ' + metric)
			f.write(',Standard deviation ' + metric)
			f.write(',Fold 0 ' + metric)
		f.write('\n')

		f.write('target')
		for metric in metrics:
			f.write(',')
			f.write(','.join(map(str,[np.nanmean(test_scores[metric]), np.std(test_scores[metric]),\
				np.nanmean(test_scores[metric])])))
		f.write('\n')


def run_evaluate(args):

	test_preds = {}
	train_args = load_args(args.checkpoint_paths[0])

	# Update predict args with training arguments to create a merged args object
	for key, value in vars(train_args).items():
		if not hasattr(args, key):
			setattr(args, key, value)

	with open(args.crossval_index_file, 'rb') as rf:
		args.crossval_index_sets = pickle.load(rf)

	args: Union[PredictArgs, TrainArgs]

	full_data = get_data(path=args.test_path, ignore_columns=[], skip_invalid_smiles=False,
				args=args, store_row=True)

	_, _, test_data = split_data(data=full_data, split_type=args.split_type, sizes=args.split_sizes, seed=args.seed, args=args)

	with open(os.path.join(os.path.dirname(args.save_dir), 'test_preds.csv'), 'r') as f:
		data = list(csv.DictReader(f))# delimiter=',', quotechar='"', escapechar='\\'))
		for i in data:
			#test_preds[i['smiles'].strip("[]").replace("'","").split()[0]] = float(i['target'])
			test_preds[i['smiles']] = float(i['target'])

	test_smiles, test_targets = test_data.smiles(), test_data.targets()

	a, b = list(), list()
	metrics = train_args.metrics + ["precision", "recall", "specificity", "accuracy", "f1_score"]

	for i in range(len(test_smiles)):
		a.append([test_preds[test_smiles[i][0]]])

	evaluate(args, a, test_targets, metrics)



if __name__ == '__main__':
	run_evaluate(PredictArgs().parse_args())
