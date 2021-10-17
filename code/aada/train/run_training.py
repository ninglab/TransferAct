from logging import Logger
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
import torch
from tqdm import trange
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from aada.utils.args import TrainArgs
from aada.utils.constants import MODEL_FILE_NAME
from aada.data import get_class_sizes, get_data, MoleculeDataLoader, MoleculeDataset, set_cache_graph, \
	ForeverDataIterator, split_data, filter_common_smiles
from aada.modules import MoleculeModel, AADALoss, DANNLoss, CADALoss, ADDALoss
from aada.utils.nn_utils import param_count
from aada.utils.utils import build_optimizer, build_lr_scheduler, get_loss_func, load_checkpoint,makedirs, \
	save_checkpoint, save_smiles_splits


def run_training(args: TrainArgs,
		source_data: MoleculeDataset,
		target_data: MoleculeDataset,
		logger: Logger = None) -> Dict[str, List[float]]:
	"""
	Loads data, trains a Chemprop model, and returns test scores for the model checkpoint with the highest validation score.

	:param args: A :class:`~aada.args.TrainArgs` object containing arguments for
		 loading data and training the Chemprop model.
	:param data: A :class:`~aada.data.MoleculeDataset` containing the data.
	:param logger: A logger to record output.
	:return: A dictionary mapping each metric in :code:`args.metrics` to a list of values for each task.

	"""
	if logger is not None:
		debug, info = logger.debug, logger.info
	else:
		debug = info = print

	# Set pytorch seed for random initial weights
	torch.manual_seed(args.pytorch_seed)
	torch.cuda.manual_seed(args.pytorch_seed)
	torch.cuda.manual_seed_all(args.pytorch_seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True


	# Split data
	debug(f'Splitting data with seed {args.seed}')

	if args.separate_test_path:
		test_data = get_data(path=args.separate_test_path, 
		       			args=args, 
		       			features_path=args.separate_test_features_path,
					logger=logger)
	if args.separate_val_path:
		val_data = get_data(path=args.separate_val_path, 
		      			args=args, 
		      			features_path=args.separate_val_features_path,
					logger=logger)

	if args.separate_val_path and args.separate_test_path:
		train_data = target_data

	elif args.separate_val_path:
		train_data, _, test_data = split_data(data=target_data, 
							split_type=args.split_type, 
							sizes=(0.8, 0.0, 0.2), 
							seed=args.seed, 
							num_folds=args.num_folds, 
							args=args, 
							logger=logger)

	elif args.separate_test_path:
		train_data, val_data, _ = split_data(data=target_data, 
				       			split_type=args.split_type, 
				       			sizes=(0.8, 0.2, 0.0), 
				       			seed=args.seed, 
				       			num_folds=args.num_folds, 
				       			args=args, 
				       			logger=logger)

	else:
		train_data, val_data, test_data = split_data(data=target_data, 
					       			split_type=args.split_type, 
					       			sizes=args.split_sizes, 
					       			seed=args.seed, 
					       			num_folds=args.num_folds, 
					       			args=args, 
					       			logger=logger)

	"""
	Creating pairs of samples first
	"""
	"""
	training_pairs = list()
	for x_s in source_data:
		for x_t in train_data:
			training_pairs.append((x_s, x_t))
	np.random.shuffle(training_pairs)

	source_data = MoleculeDataset([_[0] for _ in training_pairs])
	train_data = MoleculeDataset([_[1] for _ in training_pairs])

	#train_data, val_data, test_data = split_data(data=target_data, split_type=args.split_type, \
	#				sizes=args.split_sizes, seed=args.seed, num_folds=args.num_folds,\
	#				args=args, logger=logger)
	"""

	if args.dataset_type == 'classification':
		class_sizes = get_class_sizes(source_data)
		debug('Class sizes')
		for i, task_class_sizes in enumerate(class_sizes):
			debug(f'{args.task_names[i]} '
				f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')


	# Below code block NOT in use
	if args.features_scaling:
		source_features_scaler = source_data.normalize_features(replace_nan_token=0)
		source_data.normalize_features(source_features_scaler)
		target_features_scaler = target_data.normalize_features(replace_nan_token=0)
		target_data.normalize_features(target_features_scaler)
	else:
		source_features_scaler = None
		target_features_scaler = None

	args.train_data_size = len(source_data) + len(train_data)

	debug(f'Total size = {len(source_data)+len(target_data):,} | '
		f'train size = {len(source_data)+len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

	# Get loss function
	# loss_func = get_loss_func(args)

	# Set up test set evaluation
	test_smiles, test_targets = test_data.smiles(), test_data.targets()
	if args.dataset_type == 'multiclass':
		sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
	else:
		sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

	# Automatically determine whether to cache
	if args.train_data_size <= args.cache_cutoff:
		set_cache_graph(True)
		num_workers = 0
	else:
		set_cache_graph(False)
		num_workers = args.num_workers

	# Create data loaders
	train_source_loader = MoleculeDataLoader(dataset=source_data,
						batch_size=args.batch_size,
						num_workers=num_workers,
						class_balance=args.class_balance,
						shuffle=True,
						seed=args.seed, drop_last=True)

	train_target_loader = MoleculeDataLoader(dataset=train_data,
						batch_size=args.batch_size,
						num_workers=num_workers,
						class_balance=args.class_balance,
						shuffle=True,
						seed=args.seed, drop_last=True)

	val_loader = MoleculeDataLoader(dataset=val_data,
					batch_size=len(val_data),
					num_workers=num_workers,
					drop_last=False)

	test_loader = MoleculeDataLoader(dataset=test_data,
					batch_size=len(test_data),
					num_workers=num_workers,
					drop_last=False)

	if args.class_balance:
		debug(f'With class_balance, effective train size = {train_source_loader.iter_size+train_target_loader.iter_size:,}')

	scaler = None
	args.iters_per_epoch = min(args.iters_per_epoch, len(train_source_loader))

	# Train ensemble of models
	for model_idx in range(args.ensemble_size):
		# Tensorboard writer
		#save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
		#makedirs(save_dir)

		save_dir = args.save_dir
		"""
		try:
			writer = SummaryWriter(log_dir=save_dir)
		except:
			writer = SummaryWriter(logdir=save_dir)
		"""
		writer = None

		# Load/build model
		if args.checkpoint_paths is not None:
			debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
			model = load_checkpoint(args.checkpoint_paths[model_idx], logger=logger)
		elif args.model == "aada":
			debug(f'Building model {model_idx}')
			model = AADALoss(args)
		elif args.model == "dann":
			debug(f'Building model {model_idx}')
			model = DANNLoss(args)
		elif args.model == "cada":
			debug(f'Building model {model_idx}')
			model = CADALoss(args)
		elif args.model == "adda":
			debug(f'Building model {model_idx}')
			model = ADDALoss(args)

			params = [{'params': model.discriminator_local.parameters(), 'lr': args.init_lr, 'weight_decay': 0},
				{'params': model.discriminator_global.parameters(), 'lr': args.init_lr, 'weight_decay': 0}]
			optimizer = Adam(params)

			params = [{'params': model.feature_generator.parameters(), 'lr': args.init_lr, 'weight_decay': 0},
				{'params': model.classifier.parameters(), 'lr': args.init_lr, 'weight_decay': 0}]
			tgt_optimizer = Adam(params)

		debug(model)
		debug(f'Number of parameters = {param_count(model):,}')
		if args.cuda:
			debug('Moving model to cuda')
		model = model.to(args.device)

		# Ensure that model is saved in correct location for evaluation if 0 epochs
		save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model, scaler, target_features_scaler, args)

		# Optimizers
		optimizer = build_optimizer(model, args)

		# Learning rate schedulers
		scheduler = build_lr_scheduler(optimizer, args)
		#scheduler = ExponentialLR(optimizer, 0.995)
		#tgt_scheduler = ExponentialLR(tgt_optimizer, 0.995)

		# Run training
		best_score = float('inf') if args.minimize_score else -float('inf')
		best_epoch, n_iter = 0, 0

		source_train_iter = ForeverDataIterator(train_source_loader)
		target_train_iter = ForeverDataIterator(train_target_loader)

		for epoch in trange(args.epochs):
			debug(f'Epoch {epoch}')

			# train the model with pairs of batches
			n_iter = train(model, source_train_iter, target_train_iter, \
				optimizer=optimizer, scheduler=scheduler, \
				args=args, logger=logger, writer=writer) # tgt_optimizer=tgt_optimizer, tgt_scheduler=tgt_scheduler)

			if isinstance(scheduler, ExponentialLR):
				scheduler.step()
			"""
			if isinstance(tgt_scheduler, ExponentialLR):
				tgt_scheduler.step()
			"""

			# validate the model on target validation data
			val_scores = evaluate(
				model=model,
				data_loader=val_loader,
				num_tasks=args.num_tasks,
				metrics=args.metrics,
				dataset_type=args.dataset_type,
				scaler=target_features_scaler,
				logger=logger
				)

			for metric, scores in val_scores.items():
				# Average validation score
				avg_val_score = np.nanmean(scores)
				debug(f'Validation {metric} = {avg_val_score:.6f}')
				#writer.add_scalar(f'validation_{metric}', avg_val_score, epoch)

				if args.show_individual_scores:
					# Individual validation scores
					for task_name, val_score in zip(args.task_names, scores):
						debug(f'Validation {task_name} {metric} = {val_score:.6f}')
				#		writer.add_scalar(f'validation_{task_name}_{metric}', val_score, epoch)

			# evaluate the model on training data = source + target_train
			if args.train_eval:

				train_eval_loader = MoleculeDataLoader(dataset=source_data, batch_size=len(source_data),
						num_workers=num_workers, drop_last=False)

				train_scores = evaluate(model=model,
						data_loader=train_eval_loader,
						num_tasks=args.num_tasks,
						metrics=args.metrics,
						dataset_type=args.dataset_type,
						scaler=target_features_scaler,
						logger=logger
						)
				for metric, scores in train_scores.items():
					# Average validation score
					avg_train_score = np.nanmean(scores)
					debug(f'Training {metric} = {avg_train_score:.6f}')
				#	writer.add_scalar(f'training_{metric}', avg_train_score, epoch)

					if args.show_individual_scores:
						# Individual validation scores
						for task_name, train_score in zip(args.task_names, scores):
							debug(f'Training {task_name} {metric} = {train_score:.6f}')
				#			writer.add_scalar(f'training_{task_name}_{metric}', train_score, epoch)

			# Save model checkpoint if improved validation score
			avg_val_score = np.nanmean(val_scores[args.metric])

			if args.minimize_score and avg_val_score < best_score or \
					not args.minimize_score and avg_val_score > best_score:
				best_score, best_epoch = avg_val_score, epoch
				save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model, scaler, target_features_scaler, args)

		# Evaluate on test set using model with best validation score
		info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
		model = load_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), device=args.device, logger=logger)

		test_preds = predict(model=model, data_loader=test_loader, scaler=scaler)

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
			#writer.add_scalar(f'test_{metric}', avg_test_score, 0)

			if args.show_individual_scores:
				# Individual test scores
				for task_name, test_score in zip(args.task_names, scores):
					info(f'Model {model_idx} test {task_name} {metric} = {test_score:.6f}')
			#		writer.add_scalar(f'test_{task_name}_{metric}', test_score, n_iter)
			#writer.close()

		# Evaluate ensemble on test set
		avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()

		ensemble_scores = evaluate_predictions(
		preds=avg_test_preds,
		targets=test_targets,
		num_tasks=args.num_tasks,
		metrics=args.metrics,
		dataset_type=args.dataset_type,
		logger=logger
		)

		for metric, scores in ensemble_scores.items():
			# Average ensemble score
			avg_ensemble_test_score = np.nanmean(scores)
			info(f'Ensemble test {metric} = {avg_ensemble_test_score:.6f}')

			# Individual ensemble scores
			if args.show_individual_scores:
				for task_name, ensemble_score in zip(args.task_names, scores):
					info(f'Ensemble test {task_name} {metric} = {ensemble_score:.6f}')

		# Optionally save test preds
		if args.save_preds:
			test_preds_dataframe = pd.DataFrame(data={'smiles': test_data.smiles()})

			for i, task_name in enumerate(args.task_names):
				test_preds_dataframe[task_name] = [pred[i] for pred in avg_test_preds]

			test_preds_dataframe.to_csv(os.path.join(args.save_dir, 'test_preds.csv'), index=False)

		return ensemble_scores
