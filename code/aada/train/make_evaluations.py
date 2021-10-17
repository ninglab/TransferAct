import csv
import pickle
from typing import List, Optional, Union

import numpy as np
from tqdm import tqdm

from .predict import predict
from aada.utils.args import PredictArgs, TrainArgs
from aada.data import get_data, split_data, get_data_from_smiles, MoleculeDataLoader, MoleculeDataset
from aada.utils.utils import load_args, load_checkpoint, load_scalers, makedirs, timeit


@timeit()
def make_predictions(args: PredictArgs, smiles: List[List[str]] = None) -> List[List[Optional[float]]]:
	"""
	Loads data and a trained model and uses the model to make predictions on the data.

	If SMILES are provided, then makes predictions on smiles.
	Otherwise makes predictions on :code:`args.test_data`.

	:param args: A :class:`~aada.args.PredictArgs` object containing arguments for
				 loading data and a model and making predictions.
	:param smiles: List of list of SMILES to make predictions on.
	:return: A list of lists of target predictions.
	"""
	print('Loading training args')
	train_args = load_args(args.checkpoint_paths[0])
	num_tasks, task_names = train_args.num_tasks, train_args.task_names

	# If features were used during training, they must be used when predicting
	if ((train_args.features_path is not None or train_args.features_generator is not None)
			and args.features_path is None
			and args.features_generator is None):
		raise ValueError('Features were used during training so they must be specified again during prediction '
					'using the same type of features as before (with either --features_generator or '
					'--features_path and using --no_features_scaling if applicable).')

	# Update predict args with training arguments to create a merged args object
	for key, value in vars(train_args).items():
		if not hasattr(args, key):
			setattr(args, key, value)

	with open(args.crossval_index_file, 'rb') as rf:
		args.crossval_index_sets = pickle.load(rf)

	args: Union[PredictArgs, TrainArgs]

	print('Loading data')
	if smiles is not None:
		full_data = get_data_from_smiles(
			smiles=smiles,
			skip_invalid_smiles=False,
			features_generator=args.features_generator
		)
	else:
		full_data = get_data(path=args.test_path, target_columns=[], ignore_columns=[], skip_invalid_smiles=False,
					args=args, store_row=True)

	_, _, full_data = split_data(data=full_data, split_type=args.split_type, sizes=args.split_sizes, seed=args.seed, args=args)

	print('Validating SMILES')
	full_to_valid_indices = {}
	valid_index = 0
	for full_index in range(len(full_data)):
		if all(mol is not None for mol in full_data[full_index].mol):
			full_to_valid_indices[full_index] = valid_index
			valid_index += 1

	test_data = MoleculeDataset([full_data[i] for i in sorted(full_to_valid_indices.keys())])

	# Edge case if empty list of smiles is provided
	if len(test_data) == 0:
		return [None] * len(full_data)

	print(f'Test size = {len(test_data):,}')

	# Predict with each model individually and sum predictions
	if args.dataset_type == 'multiclass':
		sum_preds = np.zeros((len(test_data), num_tasks, args.multiclass_num_classes))
	else:
		sum_preds = np.zeros((len(test_data), num_tasks))

	# Create data loader
	test_data_loader = MoleculeDataLoader(
		dataset=test_data,
		batch_size=args.batch_size,
		drop_last=False
	)

	print(f'Predicting with an ensemble of {len(args.checkpoint_paths)} models')
	for checkpoint_path in tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths)):
		# Load model and scalers
		model = load_checkpoint(checkpoint_path, device=args.device)
		scaler, features_scaler = load_scalers(checkpoint_path)

		# Normalize features
		if args.features_scaling:
			test_data.reset_features_and_targets()
			test_data.normalize_features(features_scaler)

		# Make predictions
		model_preds = predict(
			model=model,
			data_loader=test_data_loader,
			scaler=scaler
		)
		sum_preds += np.array(model_preds)

	# Ensemble predictions
	avg_preds = sum_preds / len(args.checkpoint_paths)
	avg_preds = avg_preds.tolist()

	test_scores = evaluate_predictions(preds=test_preds,
			targets=test_targets,
			num_tasks=args.num_tasks,
			metrics=args.metrics,
			dataset_type=args.dataset_type,
			logger=logger
		)

	if len(test_preds) != 0:
		sum_test_preds += np.array(test_preds)

	# Average test score
	for metric, scores in test_scores.items():
		avg_test_score = np.nanmean(scores)
		info(f'Model {model_idx} test {metric} = {avg_test_score:.6f}')
		writer.add_scalar(f'test_{metric}', avg_test_score, 0)

		if args.show_individual_scores:
			# Individual test scores
			for task_name, test_score in zip(args.task_names, scores):
				info(f'Model {model_idx} test {task_name} {metric} = {test_score:.6f}')
				writer.add_scalar(f'test_{task_name}_{metric}', test_score, n_iter)
		writer.close()

	return avg_preds


def aada_predict_and_eval() -> None:
	"""Parses Chemprop predicting arguments and runs prediction using a trained Chemprop model.

	This is the entry point for the command line command :code:`aada_predict`.
	"""
	make_predictions(args=PredictArgs().parse_args())
