import sys
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import torch.optim as opt

from sklearn.model_selection import KFold
import logging
from torch.utils.data import DataLoader


from utils.args import parse_args
from utils.common import *
from utils.nn_utils import compute_pnorm, compute_gnorm, param_count
from models.compranknet import CompRankNet
from models.baseline import Baseline
from models.loss import myLoss
from dataloader.loader import MoleculePoint, MoleculeDatasetTrain, MoleculeDatasetTest
from dataloader.utils import get_data

from train.training import train_step
from train.eval import evaluate

# for reproducibility
SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)


def log_metrics(metric, mode, epoch, fold=None):
	print(mode, epoch, fold, sep=',', end=',')
	for k, v in metric.items():
		if fold:
			logging.info('%s (Epoch %d) : %s for fold %d = %.4f' %(mode, epoch, k, fold, v))
		else:
			logging.info('%s (Epoch %d) : Avg %s = %.4f' %(mode, epoch, k, v))
		print('%.4f' %v, end=',')
	print()


def run(model, dataset, train_index, test_index, criterion, optimizer, scheduler, fold, args):
	test_metrics = {}
	train_metrics = {}

	train_dataset = MoleculeDatasetTrain([MoleculePoint(comp, ic) for comp, ic \
						in dataset[train_index]])
	test_dataset  = MoleculeDatasetTest([MoleculePoint(comp, ic) for comp, ic \
						in dataset[test_index]])
	traineval_dataset = MoleculeDatasetTest([MoleculePoint(comp, ic) for comp, ic \
						in dataset[train_index]])

	train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=4,\
					batch_size = args.batch_size, collate_fn = train_dataset.collate_fn)

	test_dataloader  = DataLoader(test_dataset, shuffle=True, num_workers=4,\
					batch_size = args.batch_size, collate_fn = test_dataset.collate_fn)

	traineval_dataloader  = DataLoader(traineval_dataset, shuffle=True, num_workers=4,\
					batch_size = args.batch_size, collate_fn = traineval_dataset.collate_fn)

	pred_scores_test = []
	true_scores_test = []
	pred_scores_train = []
	true_scores_train = []

	for epoch in range(1, args.max_iter+1):
		loss = train_step(model, train_dataloader, criterion, optimizer, args)

		logging.info("Loss at epoch = %d : %.4f" %(epoch, loss))
		logging.info("Pnorm at epoch = %d : %.4f" %(epoch, compute_pnorm(model)))
		logging.info("GNorm at epoch = %d : %.4f" %(epoch, compute_gnorm(model)))

		if (epoch) and (epoch % args.log_steps == 0):
			if args.do_test:
				pred_scores, true_ic, metric = evaluate(model, test_dataloader)
				log_metrics(metric, 'TEST', epoch, fold)
				pred_scores_test.append(pred_scores)
				true_scores_test.append(true_ic)
				test_metrics[epoch] = metric

			if args.do_train_eval:
				pred_scores, true_ic, metric = evaluate(model, traineval_dataloader)
				log_metrics(metric, 'TRAIN', epoch, fold)
				pred_scores_train.append(pred_scores)
				true_scores_train.append(true_ic)
				train_metrics[epoch] = metric

		if args.lr_scheduler:
			scheduler.step()

	write_to_file(pred_scores_test, true_scores_test, args, 'test.log', fold)
	write_to_file(pred_scores_train, true_scores_train, args, 'train.log', fold)

	return test_metrics, train_metrics


def cross_validate(dataset, args, kfold=5):
	device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

	cv = KFold(n_splits = kfold, random_state = SEED, shuffle = True)

	fold = 1
	train_metrics_folds = []
	test_metrics_folds = []

	for train_index, test_index in cv.split(dataset):
		# initialize the model
		if args.model == 'model1':
			model = CompRankNet(args, device)
		elif args.model == 'baseline':
			model = Baseline(args)

		model = model.to(device)

		if args.optim == 'sgd':
			optimizer = opt.SGD(model.parameters(), lr=args.learning_rate)
		elif args.optim == 'adam':
			optimizer = opt.Adam(model.parameters(), lr=args.learning_rate)

		if args.lr_scheduler:
			scheduler = opt.lr_scheduler.ExponentialLR(optimizer, 0.95)
		else:
			scheduler = None

		criterion = myLoss()

		logging.info('Cross validation: Fold %d/%d' %(fold, kfold))


		if args.do_train:
			logging.info('Start training...')
			m1, m2 = run(model, dataset, train_index, test_index, criterion, optimizer, scheduler, fold, args)
			test_metrics_folds.append(m1)
			train_metrics_folds.append(m2)

		'''
		if args.do_test:


			metric = evaluate(model, test_dataloader)
			test_metrics.append(metric)
			log_metrics(metric, 'TEST', fold)

		if args.do_train_eval:
			test_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=0,\
						batch_size = args.batch_size, collate_fn = train_dataset.collate_fn_test)

			logging.info('Evaluating on training set...')
			metric = evaluate(model, test_dataloader)
			train_metrics.append(metric)
			log_metrics(metric, 'TRAIN', fold)
		'''
		fold += 1


	if args.do_test:
		for ep in range(args.log_steps, args.max_iter+1, args.log_steps):
			avg_metrics = []
			for fold in range(kfold):
				for e, metric in test_metrics_folds[fold].items():
					if e == ep:
						avg_metrics.append(metric)
			log_metrics(calc_avg_perf(avg_metrics), 'TEST', ep)

	if args.do_train_eval:
		for ep in range(args.log_steps, args.max_iter+1, args.log_steps):
			avg_metrics = []
			for fold in range(kfold):
				for e, metric in train_metrics_folds[fold].items():
					if e == ep:
						avg_metrics.append(metric)
			log_metrics(calc_avg_perf(avg_metrics), 'TRAIN', ep)


def main(args):
	set_logger(args)

	data = get_data(args.data_path, args.smiles_path)

	cross_validate(np.asarray(data), args)

if __name__ == '__main__':
	main(parse_args())
