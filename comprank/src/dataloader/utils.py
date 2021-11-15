import csv
import os

import numpy as np
from scipy import stats

def get_data(filename, smiles_file):
	smiles_dict = get_data_from_smiles(smiles_file)

	data = []
	with open(filename, 'r') as fp:
		reader = csv.reader(fp)
		data = list(reader)

	for index in range(len(data)):
		data[index][0] = smiles_dict[data[index][0]]
		data[index][1] = float(data[index][1])
	return data

def get_pair(filename, smiles_file):
	"""
		returns the SMILES strings for ranked pairs in a CSV file
	"""
	smiles_dict = {}
	with open(smiles_file, 'r') as fp:
		for line in fp.readlines():
			cid, smiles = line.strip().split('\t')
			smiles_dict[cid] = smiles

	data = []
	with open(filename, 'r') as fp:
		reader = csv.reader(fp)
		data = list(reader)

	for index in range(len(data)):
		data[index][0] = smiles_dict[data[index][0]]
		data[index][1] = smiles_dict[data[index][1]]
		data[index][2] = float(data[index][2])
		data[index][3] = float(data[index][3])

	return data

def get_data_from_smiles(smiles_file):
	smiles_dict = {}
	with open(smiles_file, 'r') as fp:
		for line in fp.readlines():
			cid, smiles = line.strip().split('\t')
			smiles_dict[cid] = smiles

	return smiles_dict


def get_data_labels(data, bins=100):
	values = [float(_[1]) for _ in data]
	percentile = np.array([stats.percentileofscore(values, _) for _ in values])
	bins_percentile = list(range(0, 101, int(100/bins)))
	labels = np.digitize(percentile, bins_percentile, right=True)

	return labels
