import os
import random
import sys
import numpy as np
from typing import Tuple
from argparse import ArgumentParser

"""
Validates bioassay data checking overlapping between the smiles,target pairs
and accordingly split the active and inactive overlapping equally among two bioassays

e.g: python scripts/process_assay_pairs.py --data_path_1 data/bioassays/processed_assays/hepc/AID_2173_datatable_all.csv
	--data_path_2 data/bioassays/processed_assays/hiv/AID_1053136_datatable_all.csv
	--balance --save_dir data/bioassays/transfer_2/2173_1053136/
"""

def read_pubchem_smiles(smiles_file):
	count = 0
	all_smiles = set()
	with open(smiles_file, 'r') as f:
		for line in f.readlines():
			cid, smile = line.strip().split('\t')

		#	if (count%100000 == 0):
		#		print(f'Reading in {count} PubChem SMILES')

			count += 1
			all_smiles.add(smile)

	return all_smiles


# assume unique smiles in the file - still debug using assertion
def read_bioassay(file):
	"""
	param: file: input csv file with smiles,target rows
	"""
	active_smiles, inactive_smiles = list(), list()
	smiles_act = dict()

	with open(file, 'r') as fp:
		for line in fp.readlines()[1:]:
			smiles, act = line.strip().split(',')
			smiles_act[smiles] = act
	return smiles_act


def write_file(args, bioassay_path, active_smiles, inactive_smiles,
		common_active, common_inactive, common_smiles,
		common_smiles_diff_labels, smiles_to_sample_from):

	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	basename = os.path.basename(bioassay_path)
	fname = os.path.join(args.save_dir, basename.split('.csv')[0] + \
				'-' + str(args.include_ratio) + '.csv')
	fout = open(fname, 'w')

	count = 0
	count_active = 0
	count_inactive = 0

	print("smiles,target", file=fout)

	# add active compounds first
	for _ in common_active:
		print(_ + ",1", file=fout)
		count_active += 1

	# discard common SMILES so that there are no duplicate within the bioassay
	#active_smiles = [_ for _ in active_smiles if _ not in common_smiles]
	#inactive_smiles = [_ for _ in inactive_smiles if _ not in common_smiles]
	active_smiles = active_smiles - set(common_smiles)

	for _ in active_smiles:
		print(_ + ",1", file=fout)
		count_active += 1

	# add inactive compounds with same number of active compounds
	# perform random sampling if required to reach same #active
	for _ in common_inactive:
		if count_inactive < count_active:
			print(_ + ",0", file=fout)
			count_inactive += 1

	# here sort this for reproducibility
	inactive_smiles = sorted(inactive_smiles - set(common_smiles))
	if count_inactive < count_active:
		for _ in inactive_smiles:
			if count_inactive < count_active:
				print(_ + ",0", file=fout)
				count_inactive += 1

	tmp_smiles = list()
	# return tmp_smiles so that next time we can consider (smiles_to_sample_from - tmp_smiles) to sample from
	if args.balance:
		# perform sampling from PubChem set of compounds to create balanced set 
		tmp_smiles = random.sample(smiles_to_sample_from, count_active - count_inactive)
		# choosen smile should not already be in the list of smiles for the pair
		# to avoid checking duplicates across pair again
		for tmp_smile in tmp_smiles:
			print(tmp_smile + ",0", file=fout)
			count_inactive += 1

	if args.include_ratio:
		# include smiles with different labels in two assays -- this is not used now
		num  = int(args.include_ratio * len(common_smiles_diff_labels))
		for i in range(num):
			sm = common_smiles_diff_labels[i]
			print(sm + "," + smiles_act[sm], file=fout)

	fout.close()
	return tmp_smiles


def address_overlap(args):
	"""
		Splits overlapping data (smiles,target) across two bioassays in half
		overlapping smiles but different target are either included or discarded
		given the 'include_ratio', If include_ratio = 0, discard all of them
	"""
	random.seed(args.seed)
	np.random.seed(args.seed)

	smiles_act_1 = read_bioassay(args.data_path_1)
	smiles_act_2 = read_bioassay(args.data_path_2)
	common_smiles = []
	common_active = []
	common_inactive = []
	common_smiles_diff_labels = []

	for sm, act in smiles_act_1.items():
		if sm in smiles_act_2:
			if smiles_act_2[sm] == act:
				if act == "1":
					common_active.append(sm)
				else:
					common_inactive.append(sm)
			else:
				common_smiles_diff_labels.append(sm)
			common_smiles.append(sm)

	#active_smiles_1, inactive_smiles_1 = read_bioassay(args.data_path_1)
	#active_smiles_2, inactive_smiles_2 = read_bioassay(args.data_path_2)
	"""
	for sm in active_smiles_1:
		if sm in active_smiles_2:
			common_active.append(sm)
			common_smiles.append(sm)
		elif sm in inactive_smiles_2:
			common_smiles_diff_labels.append(sm)
			common_smiles.append(sm)

	for sm in inactive_smiles_1:
		if sm in active_smiles_2:
			common_smiles_diff_labels.append(sm)
			common_smiles.append(sm)
		elif sm in inactive_smiles_2:
			common_inactive.append(sm)
			common_smiles.append(sm)

	"""
	"""
	print("Common active SMILES = ", len(common_active))
	print("Common inactive SMILES = ", len(common_inactive))
	print("Common SMILES = ", len(common_smiles))
	print("Common SMILES BUT different labels = ", len(common_smiles_diff_labels))
	print("% Common SMILES = ", f'{100*len(common_smiles)/(len(list(smiles_act1.keys())) + len(list(smiles_act2.keys()))):.2f}')
	print("% Common SMILES with same labels = ", f'{100*(len(common_active) + len(common_inactive))/(len(list(smiles_act1.keys())) + len(list(smiles_act2.keys()))):.2f}')
	print("% Common SMILES BUT different labels = ", f'{100*len(common_smiles_diff_labels)/(len(list(smiles_act1.keys())) + len(list(smiles_act2.keys()))):.2f}')
	"""

	active_smiles_1 = [k for k,v in smiles_act_1.items() if v == "1"]
	inactive_smiles_1 = [k for k,v in smiles_act_1.items() if v == "0"]
	active_smiles_2 = [k for k,v in smiles_act_2.items() if v == "1"]
	inactive_smiles_2 = [k for k,v in smiles_act_2.items() if v == "0"]

	all_smiles_pair = set(active_smiles_1 + inactive_smiles_1 + active_smiles_2 + inactive_smiles_2)

	# debugging - check no duplicate within bioassay
	assert(len(active_smiles_1) == len(set(active_smiles_1)))
	assert(len(inactive_smiles_1) == len(set(inactive_smiles_1)))
	assert(len(active_smiles_2) == len(set(active_smiles_2)))
	assert(len(inactive_smiles_2) == len(set(inactive_smiles_2)))

	active_smiles_1, inactive_smiles_1, active_smiles_2, inactive_smiles_2 = \
		set(active_smiles_1), set(inactive_smiles_1), set(active_smiles_2), set(inactive_smiles_2)

	total_smiles_1 = len(active_smiles_1) + len(inactive_smiles_1)
	total_smiles_2 = len(active_smiles_2) + len(inactive_smiles_2)
	total_actives = len(active_smiles_1) + len(active_smiles_2)
	total_inactives = len(inactive_smiles_1) + len(inactive_smiles_2)

	print(f'{len(active_smiles_1)},{len(inactive_smiles_1)},{len(active_smiles_2)},{len(inactive_smiles_2)}', end=',')
	print(f'{len(common_active)},{len(common_inactive)},{len(common_active)+len(common_inactive)},{len(common_smiles_diff_labels)},{len(common_smiles)}', end=',')
	# % active = #common_activ
	print(f'{100*len(common_active)/total_actives:.2f}', end=',')
	print(f'{100*len(common_inactive)/total_inactives:.2f}', end=',')
	print(f'{100*(len(common_active)+len(common_inactive))/(total_smiles_1 + total_smiles_2):.2f}', end=',')
	print(f'{100*len(common_smiles_diff_labels)/(total_smiles_1 + total_smiles_2):.2f}', end=',')
	print(f'{100*len(common_smiles)/(total_smiles_1 + total_smiles_2):.2f}')

	random.shuffle(common_active)
	random.shuffle(common_inactive)

	# remove smiles already used in 2 assays
	smiles_to_sample_from = read_pubchem_smiles(args.smiles_path)
	smiles_to_sample_from = sorted(smiles_to_sample_from - all_smiles_pair)
	#smiles_to_sample_from = [_ for _ in smiles_to_sample_from if _ not in all_smiles_pair]

	tmp_smiles = write_file(args, args.data_path_1, active_smiles_1, inactive_smiles_1, common_active[::2], \
			common_inactive[::2], common_smiles, common_smiles_diff_labels, smiles_to_sample_from)

	smiles_to_sample_from = [_ for _ in smiles_to_sample_from if(_ not in tmp_smiles)]
	write_file(args, args.data_path_2, active_smiles_2, inactive_smiles_2, common_active[1::2], \
			common_inactive[1::2], common_smiles, common_smiles_diff_labels, smiles_to_sample_from)


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--smiles_path', type=str, required=True,
			help='Path to tsv containing all CID<tab>Smiles')
	parser.add_argument('--data_path_1', type=str, required=True,
			help='Path to .csv containing data')
	parser.add_argument('--data_path_2', type=str, required=True,
			help='Path to .csv containing data')
	parser.add_argument('--include_ratio', type=float, default=0.0,
			help='Ratio of overlapping smiles but different target pairs to include')
	parser.add_argument('--seed', type=int, default=123,
			help='Random seed')
	parser.add_argument('--balance', dest='balance', action='store_true',
			help='To oversample inactive compounds in highly imbalanced dataset')
	parser.add_argument('--split_type', type=str, choices=['random', 'scaffold_balanced'], default='random',
			help='Method of splitting the data')
	parser.add_argument('--save_dir', type=str, required=True,
			help='Path to directory where modified bioassay data will be saved')
	args = parser.parse_args()

	# checks the overlap b/w two bioassays and split accordingly the active and inactive ones
	address_overlap(args)
