import os
import argparse

def override_args(args):
	if args.model == 'baseline':
		args.message_steps = None
		args.pooling = None
		if args.use_features is None:
			raise Exception('use_features must be provided for baseline models')

	return args

def parse_args(args = None):
	parser = argparse.ArgumentParser(
		description = 'Training and Testing Compound Rank Net',
		usage = 'run.py [<args>] [-h | --help]'
		)

	parser.add_argument('--cuda', action = 'store_true', help = 'use GPU')

	parser.add_argument('--model', type=str, default='model1', choices=['model1', 'baseline'])
	parser.add_argument('--use_features', type=str, choices=['morgan', 'morgan_count', \
						'morgan_tanimoto_bioassay', 'rdkit_2d', 'rdkit_2d_normalized'])

	parser.add_argument('--do_train', action = 'store_true')
	parser.add_argument('--do_valid', action = 'store_true')
	parser.add_argument('--do_test', action = 'store_true')
	parser.add_argument('--do_train_eval', action = 'store_true', help='Evaluating on training data')

	parser.add_argument('--data_path', type=str, required=True)
	parser.add_argument('--smiles_path', type=str, required=True)
	parser.add_argument('-d', '--hidden_dim', default=50, type=int)
	parser.add_argument('-attn_d', '--attn_dim', default=20, type=int)
	parser.add_argument('-T', '--message_steps', default=2, type=int)
	parser.add_argument('-pool', '--pooling', default='mean', type=str, \
				choices=['sum', 'mean', 'max', 'attention', 'self-attention', 'all-concat', 'hier-attention'])
	parser.add_argument('--atom_messages', action = 'store_true')

	parser.add_argument('--optim', default='adam', choices=['sgd', 'adam'], type=str)
	parser.add_argument('-loss', '--loss_fn', default='pair', choices=['pair', 'pair2', 'pairc', 'npair'], type=str)
	parser.add_argument('-reg', '--regularization', default=None, type=float)
	parser.add_argument('-drop', '--dropout', default=0, type=float)
	parser.add_argument('-del', '--delta', default=None, type=float, help='Provide threshold for creating ranked pairs')
	parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float)
	parser.add_argument('--max_iter', default=100, type=int)
	parser.add_argument('-b', '--batch_size', default=64, type=int)
	parser.add_argument('-clip', '--grad_clip', default=None, type=int, help='Allow gradient clipping')
	parser.add_argument('-lrsch', '--lr_scheduler', action='store_true', help='Allow learning rate scheduler')

	parser.add_argument('--log_steps', default=5, type=int, help='log evaluation results every 5 epochs')
	parser.add_argument('-save', '--save_path', required=True, type=str)
	parser.add_argument('-viz_dir', '--viz_dir', type=str)

	temp = parser.parse_args(args)
	return override_args(temp)
