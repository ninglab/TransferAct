'''
Synopsis: Create a dataloader class that creates training instances with pairs (c_i, c_j)
	such that c_i is ranked above c_j
'''
import numpy as np
from typing import List
import torch
from torch.utils.data import DataLoader, Dataset
from rdkit import Chem

from features.features_generators import *
from features.featurization import MolGraph, BatchMolGraph


class MoleculePoint:
	'''
		initialize molecular compound pairs with features
		1st compound ranked above 2nd compound
	'''
	def __init__(self, smiles, ic, features=None, label=None, use_features=None):
		self.smiles = smiles
#		self.cid = cid
		self.ic = float(ic)
		self.label = label
		self._mol = 'None'
		self.features_dim = None
		self.features = None
		if features is not None:
			self.features_dim = len(features)
			self.features = features
	@property
	def mol(self):
		if self._mol == 'None':
			self._mol = Chem.MolFromSmiles(self.smiles)
		return self._mol

# create ordered pairs only if the difference is greater than threshold percentile delta
def get_pairs(molecules, delta=0):
	if delta:
		delta = np.percentile([_.ic for _ in molecules], delta, interpolation='nearest')
	pairs = []
	for comp1 in molecules:
		for comp2 in molecules:
			if (comp1.ic - comp2.ic) < delta:
				pairs.append((comp1, comp2))
	return np.asarray(pairs)


class MoleculeDatasetTrain(Dataset):
	def __init__(self, data, delta):
		self.molgraphs = None
		self.data = np.array(data)
		self.pairs = get_pairs(self.data, delta)
		self.scaler = None

	'''
	def get_smiles(self):
		return np.array([each.smiles for each in self.data])

	def get_true_ranks(self):
		return np.array([each.ic for each in self.data])

	def to_graph(self):
		if self.molgraphs is None:
			self.molgraphs = []
			for each in self.data:
				self.molgraphs.append(MolGraph(each.mol))

		self.molgraphs = np.array(self.molgraphs)
	'''
	def __len__(self):
		return len(self.pairs)

	def __getitem__(self, idx):
		return self.pairs[idx]

	@staticmethod
	def collate_fn(data):
		return data


class MoleculeDatasetTest(Dataset):
	def __init__(self, data):
		self.molgraphs = None
		self.data = np.array(data)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]

	@staticmethod
	def collate_fn(data):
		return data
