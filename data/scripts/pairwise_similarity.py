#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : pairwise_similarity_V2.py
# Author            : Vishal Dey <dey.78@osu.edu>
# Date              : Fri 01 Oct 2021 18:51:46
# Last Modified Date: Sat 16 Oct 2021 15:43:23
# Last Modified By  : Vishal Dey <dey.78@osu.edu>
from itertools import product
import math
import os
import sys
import csv
from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
from rdkit.Chem import AllChem


def morgan_similarity(mols_1, mols_2, use_topk):
	#input smiles here are already converted to fingerprints
	similarities = []

	ind = []
	for i in range(len(mols_1)):
		tmp = []
		for j in range(len(mols_2)):
			if mols_1[i] != mols_2[j]:
				similarity = DataStructs.TanimotoSimilarity(mols_1[i], mols_2[j])
				tmp.append(similarity)

		if use_topk == -1:
			similarities.append(np.mean(tmp))
		else:
			tmp = np.array(tmp)
			#most_similar = tmp[np.argpartition(tmp, -use_topk)[-use_topk:]]
			most_similar = tmp[np.argsort(-tmp)[:use_topk]]
			similarities.append(np.mean(most_similar))

	print(f'{np.mean(similarities):.4f}', end=',')


def get_smiles(file):
	active_smiles, inactive_smiles = [], []

	with open(file, 'r') as fp:
		data = csv.DictReader(fp)
		for line in data:
			if line['target'] == '1':
				active_smiles.append(line['smiles'])
			else:
				inactive_smiles.append(line['smiles'])

	return active_smiles, inactive_smiles


def get_similarity(mols_1, mols_2, radius, similarity_measure, use_topk):
	if similarity_measure == 'morgan':
		morgan_similarity(mols_1, mols_2, use_topk)
	else:
		raise(f"Not yet implemented: {similarity_measure}")


def compute_similarity(data_path_1, data_path_2, similarity_measure, sampling, bits, radius, seed, use_topk):
	print(os.path.basename(data_path_1) + "," + os.path.basename(data_path_2), end=',')

	np.random.seed(seed)

	active_smiles_1, inactive_smiles_1 = get_smiles(data_path_1)
	active_smiles_2, inactive_smiles_2 = get_smiles(data_path_2)

#	sample_size = min(ceil(sample_rate*max(len(active_smiles_1)+len(inactive_smiles_1),\
#				len(active_smiles_2)+len(inactive_smiles_2))), 1000)

	# Sample to improve speedup
	if sampling:
		sample_size = 5000
		np.random.seed(seed)
		sample_active_smiles_1 = np.random.choice(active_smiles_1, size=min(len(active_smiles_1), sample_size), replace=False)
		np.random.seed(seed)
		sample_inactive_smiles_1 = np.random.choice(inactive_smiles_1, size=min(sample_size, len(inactive_smiles_1)), replace=False)
		np.random.seed(seed)
		sample_active_smiles_2 = np.random.choice(active_smiles_2, size=min(len(active_smiles_2), sample_size), replace=False)
		np.random.seed(seed)
		sample_inactive_smiles_2 = np.random.choice(inactive_smiles_2, min(sample_size, len(inactive_smiles_2)), replace=False)
	else:
		sample_active_smiles_1, sample_inactive_smiles_1 = active_smiles_1, inactive_smiles_1
		sample_active_smiles_2, sample_inactive_smiles_2 = active_smiles_2, inactive_smiles_2

	smiles_1 = np.concatenate((sample_active_smiles_1, sample_inactive_smiles_1))
	smiles_2 = np.concatenate((sample_active_smiles_2, sample_inactive_smiles_2))

	sample_active_mols_1 = [AllChem.GetMorganFingerprint(Chem.MolFromSmiles(_), radius=radius) for _ in sample_active_smiles_1]
	sample_inactive_mols_1 = [AllChem.GetMorganFingerprint(Chem.MolFromSmiles(_), radius=radius) for _ in sample_inactive_smiles_1]
	sample_active_mols_2 = [AllChem.GetMorganFingerprint(Chem.MolFromSmiles(_), radius=radius) for _ in sample_active_smiles_2]
	sample_inactive_mols_2 = [AllChem.GetMorganFingerprint(Chem.MolFromSmiles(_), radius=radius) for _ in sample_inactive_smiles_2]
	mols_1 = sample_active_mols_1 + sample_inactive_mols_1
	mols_2 = sample_active_mols_2 + sample_inactive_mols_2

	# computes similarity among the compounds in data_path_1
	get_similarity(mols_1, mols_1, radius, similarity_measure, use_topk)

	# computes similarity among the compounds in data_path_2
	get_similarity(mols_2, mols_2, radius, similarity_measure, use_topk)
	# computes similarity b/w all the compounds in data_path_1 and data_path_2
	get_similarity(mols_1, mols_2, radius, similarity_measure, use_topk)

	# computes similarity among the active compounds in data_path_1
	get_similarity(sample_active_mols_1, sample_active_mols_1, radius, similarity_measure, use_topk)
	# computes similarity among the inactive compounds in data_path_1 
	get_similarity(sample_inactive_mols_1, sample_inactive_mols_1, radius, similarity_measure, use_topk)
	# computes similarity among the active & inactive compounds in data_path_1
	get_similarity(sample_active_mols_1, sample_inactive_mols_1, radius, similarity_measure, use_topk)

	# computes similarity among the active compounds in data_path_2
	get_similarity(sample_active_mols_2, sample_active_mols_2, radius, similarity_measure, use_topk)
	# computes similarity among the inactive compounds in data_path_2 
	get_similarity(sample_inactive_mols_2, sample_inactive_mols_2, radius, similarity_measure, use_topk)
	# computes similarity among the active & inactive compounds in data_path_2 
	get_similarity(sample_active_mols_2, sample_inactive_mols_2, radius, similarity_measure, use_topk)

	# computes similarity b/w the active compounds in data_path_1 and data_path_2
	get_similarity(sample_active_mols_1, sample_active_mols_2, radius, similarity_measure, use_topk)
	# computes similarity b/w the inactive compounds in data_path_1 and data_path_2
	get_similarity(sample_inactive_mols_1, sample_inactive_mols_2, radius, similarity_measure, use_topk)

	# computes similarity b/w the active in data_path_1 and inactive in data_path_2
	get_similarity(sample_active_mols_1, sample_inactive_mols_2, radius, similarity_measure, use_topk)
	# computes similarity b/w the inactive in data_path_1 and active in data_path_2
	get_similarity(sample_inactive_mols_1, sample_active_mols_2, radius, similarity_measure, use_topk)
	#print()


if __name__ == '__main__':

	parser = ArgumentParser()
	parser.add_argument('--data_path_1', type=str, required=True,
			help='Path to .csv containing data')
	parser.add_argument('--data_path_2', type=str, required=True,
			help='Path to .csv containing data')
	parser.add_argument('--sampling', action='store_true',
			help='Random sampling to speedup Tanimoto computation')
	parser.add_argument('--seed', type=int, default=0,
			help='Random seed')
	parser.add_argument('--bits', type=int, default=2048,
			help='nBits for computing binary morgan fingerprints')
	parser.add_argument('--radius', type=int, default=3,
			help='Radius for computing morgan fingerprints')
	parser.add_argument('--similarity_measure', type=str, default='morgan',
			choices=['morgan','scaffold'])
	parser.add_argument('--use_topk', type=int, default=-1,
			help='use only top-k nearest neighbors to compute similarities; -1 denotes use all compounds')

	args = parser.parse_args()

	compute_similarity(**vars(args))
