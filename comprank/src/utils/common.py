import sys
import os
import logging
import json

from collections import defaultdict

import torch
import numpy as np
from sklearn.metrics import ndcg_score
from features.features_generators import *

from rdkit import Chem, DataStructs
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import SimilarityMaps
import matplotlib
import matplotlib.pyplot as plt

# for reproducibility
def set_seed(seed=123):
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.manual_seed(seed)
	torch.backends.cudnn.deterministic=True
	import random
	random.seed(seed)

def set_logger(args=None):
	'''
	Write logs to checkpoint and console
	'''
	'''
	if args.do_train:
		log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
	else:
		log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')
	'''
	log_file = os.path.join(args.save_path)

	logging.basicConfig(
		format	 = '%(asctime)s %(levelname)-8s %(message)s',
		level	 = logging.INFO,
		datefmt  = '%Y-%m-%d %H:%M:%S',
		filename = log_file,
		filemode = 'w'
	)
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)


def save_model(model, optimizer, args):
	'''
	Save the parameters of the model and the optimizer,
	as well as some other variables such as step and learning_rate
	'''

	argparse_dict = vars(args)
	with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
		json.dump(argparse_dict, fjson)

	torch.save({
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict()},
		os.path.join(args.save_path, 'checkpoint')
	)


def load_model(model, path):
	logging.info('Loading checkpoint %s...' % path)
	checkpoint = torch.load(os.path.join(path, 'checkpoint'))
	model.load_state_dict(checkpoint['model_state_dict'])

def create_pairs(dataloader):
	x_train = []
	for data1 in dataloader:
		for data2 in dataloader:
			if data1[2] < data2[2]:
				x_train.append(list(zip(*[data1, data2])))
	if x_train:
		return np.array(x_train)
	else:
		return None

def calc_avg_perf(metrics):
	avg_metric = defaultdict(list)
	for metric in metrics:
		for k, v in metric.items():
			if k in avg_metric:
				avg_metric[k].append(v)
			else:
				avg_metric[k] = [v]

	return {k: np.mean(v) for k, v in avg_metric.items()}


def write_to_file(pred_scores_list, true_scores_list, args, name, fold):
	logfile = args.save_path.split('.log')[0] + "." + name

	assert(len(pred_scores_list) == len(true_scores_list))
	mode = 'w'
	if fold > 1:
		mode = 'a'

	with open(logfile, mode) as fp:
		for i in range(len(pred_scores_list)):
			print(' '.join(map(str, pred_scores_list[i])), file=fp)
			print(' '.join(map(str, true_scores_list[i])), file=fp)


def precompute_features(dataset, train_index, test_index, use_features):
	features_generator = get_features_generator(use_features)

	train_all_mols = [Chem.MolFromSmiles(mol) for mol,_ in dataset[train_index]]
	train_features, test_features = [], []

	for mol, ic in dataset[train_index]:
		mol = Chem.MolFromSmiles(mol)
		if mol is not None and mol.GetNumHeavyAtoms() > 0:
			if use_features == 'morgan_tanimoto_bioassay':
				train_features.append(features_generator(mol, train_all_mols))
			else:
				train_features.append(features_generator(mol))

	for mol, ic in dataset[test_index]:
		mol = Chem.MolFromSmiles(mol)
		if mol is not None and mol.GetNumHeavyAtoms() > 0:
			if use_features == 'morgan_tanimoto_bioassay' or use_features == 'mc_tanimoto_bioassay':
				test_features.append(features_generator(mol, train_all_mols))
			else:
				test_features.append(features_generator(mol))

	return train_features, test_features


def viz_atom_attention(viz_dir, smiles, ic, attn_scores):
	"""
	smiles_viz_dir = os.path.join(viz_dir, f'{smiles}')
	os.makedirs(smiles_viz_dir, exist_ok=True)
	"""
	attn_scores = attn_scores.cpu().data.numpy().flatten()
	mol = Chem.MolFromSmiles(smiles)
	"""
	for atom in mol.GetAtoms():
		atom.SetAtomMapNum(atom.GetIdx())
	#	lbl = '%s:%f'%(atom.GetSymbol(), attn_scores[atom.GetIdx()])
	#	atom.SetProp('atomLabel', lbl)

	#d = rdMolDraw2D.MolDraw2DSVG(400, 400)
	#d.DrawMolecule(mol)
	#colorMap=matplotlib.cm.bwr
	d = None
	fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, colorMap=matplotlib.cm.bwr, weights=attn_scores, draw2d=d)
	#d.FinishDrawing()
	"""
	fp = open(os.path.join(viz_dir, 'attn_scores.txt'), 'a')
	txt = ','.join(map(str, ['{:.2e}'.format(_) for _ in attn_scores]))
	print(smiles + "," + str(ic) + "," + txt, file=fp)
	fp.close()
	"""
	txt = ''
	tmp = []
	for i in range(len(attn_scores)):
		tmp.append(attn_scores[i])
		if (i+1) % 10 == 0 or i == len(attn_scores)-1:
			txt = txt + ','.join(map(str, ['{:.2e}'.format(_) for _ in tmp])) + '\n,'
			tmp = []

	plt.figtext(0.5, 0.2, txt, fontsize=8, wrap=True)
	plt.figtext(0.5, 1.99, 'ic50 = ' + f'{ic}')
	fig.savefig(os.path.join(smiles_viz_dir, 'atom_att.png'), bbox_inches='tight')
	plt.close(fig)
	"""
