import logging
from typing import Callable

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from aada.utils.args import TrainArgs
from aada.data import MoleculeDataLoader, MoleculeDataset, ForeverDataIterator
from aada.modules import MoleculeModel, AADALoss
from aada.utils.nn_utils import compute_gnorm, compute_pnorm, NoamLR


def train(model: AADALoss,
	source_train_iter: ForeverDataIterator,
	target_train_iter: ForeverDataIterator,
	optimizer: Optimizer,
	scheduler: _LRScheduler,
	args: TrainArgs,
	logger: logging.Logger = None,
	writer: SummaryWriter = None):

	"""
	Trains a model for an epoch.

	:param model: A :class:`~aada.models.model.MoleculeModel`.
	:param data_loader: A :class:`~aada.data.data.MoleculeDataLoader`.
	:param loss_func: Loss function.
	:param optimizer: An optimizer.
	:param scheduler: A learning rate scheduler.
	:param args: A :class:`~aada.args.TrainArgs` object containing arguments for training the model.
	:param logger: A logger for recording output.
	:param writer: A tensorboardX SummaryWriter.
	:return: The total number of iterations (training examples) trained on so far.
	"""
	debug = logger.debug if logger is not None else print

	model.train()
	loss_sum = count = 0
	source_loss = target_loss = s_local_loss = t_local_loss = s_global_loss = t_global_loss = 0

	"""
	for each epoch, run the model with `args.iters_per_epoch` pair of source and target batches,
	implemented with infinite data iterator so that even for smaller dataset, model is trained enough.
	hence it is implemented as a for loop with specified #iterations per epoch
	"""
	for iter_count in range(1, args.iters_per_epoch+1):

		# get source and target batch
		x_s = next(source_train_iter)
		x_t = next(target_train_iter)

		model.zero_grad()
		# forward-pass through the model
		ret = model(x_s, x_t, training=True)
		loss = ret[0]

		try:
			source_loss += ret[1].item()
			target_loss += ret[2].item()

			if args.lamda != 0 and not args.exclude_local:
				s_local_loss += ret[3].item()
				t_local_loss += ret[4].item()
			if args.lamda != 0 and not args.exclude_global:
				s_global_loss += ret[5].item()
				t_global_loss += ret[6].item()

		except:
			pass

		loss_sum += loss.item()
		count += 1

		# backward pass
		loss.backward()

		if args.grad_clip:
			nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
		optimizer.step()

		if isinstance(scheduler, NoamLR):
			scheduler.step()

		# Log and/or add to tensorboard
		if iter_count % args.log_frequency == 0:
			lrs = scheduler.get_lr()
			pnorm = compute_pnorm(model)
			gnorm = compute_gnorm(model)
			loss_avg = loss_sum / count

			lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
			if (args.lamda != 0):
				debug(f'Source Classification Loss = {source_loss/count:.4f}, Target Classification Loss = {target_loss/count:.4f}, '
					f'Source Local Disc Loss = {s_local_loss/count:.4f}, Target Local Disc Loss = {t_local_loss/count:.4f}, '
					f'Source Global Disc Loss = {s_global_loss/count:.4f}, Target Global Disc Loss = {t_global_loss/count:.4f}')
			else:
				debug(f'Source Classification Loss = {source_loss/count:.4f}, Target Classification Loss = {target_loss/count:.4f}')
			debug(f'Loss = {loss_avg:.4e}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, {lrs_str}')

			loss_sum = count = 0
			source_loss = target_loss = s_local_loss = t_local_loss = s_global_loss = t_global_loss = 0

			if writer is not None:
				writer.add_scalar('train_loss', loss_avg, iter_count)
				writer.add_scalar('param_norm', pnorm, iter_count)
				writer.add_scalar('gradient_norm', gnorm, iter_count)

				for i, lr in enumerate(lrs):
					writer.add_scalar(f'learning_rate_{i}', lr, iter_count)

