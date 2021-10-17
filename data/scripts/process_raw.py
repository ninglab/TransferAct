#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : process_raw.py
# Author            : Vishal Dey <dey.78@osu.edu>
# Date              : Sat 16 Oct 2021 19:16:24
# Last Modified Date: Sat 16 Oct 2021 19:16:24
# Last Modified By  : Vishal Dey <dey.78@osu.edu>
import os
import sys
import codecs
import csv
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict
from rdkit import Chem

"""
Usage: python scripts/process_raw.py -h

Combines assays if they correspond to the same target and
does the following:
1. Ensure SMILES are canonical (using RDKit) before anything.
2. Only choose compounds which are specified as 'Inhibitors' in Phenotype column:
	a. This check is valid only if 'Phenotype*' column exists.
	b. If assay entry contains replicate experiments with multiple phenotypes in one row,
	use majority voting to determine phenotype.
3. If there are duplicate entries SMILES with same labels, keep one of such entries.
4. If there are duplicate entries SMILES with different labels, discard all entries of that compound from both the bioassay.
"""

# global dict() to store mappings of CID and SMILES
cid_smiles = {}
log_aid_wise = 'assaywise.log'
log_target_wise = 'targetwise.log'


# read cid_smiles list into memory
def read_cid_smiles(cid_smiles_path):
	with open(cid_smiles_path, 'r') as fp:
		count = 0

		for line in fp.readlines():
			if count % 100000 == 0:
				print(f'Reading line: {count}')
			count += 1
			tmp = line.strip().split('\t')
			cid_smiles[tmp[0]] = tmp[1]


# read list of targets, AIDs and pfams
def read_target_aids_pfam(target_aids_list_path, choosen_pfam_path):
	target_aids = defaultdict(list)
	pfam_targets = defaultdict(list)

	choosen_pfams = list()
	with open(choosen_pfam_path, 'r') as f:
		for line in f.readlines():
			choosen_pfams.append(line.strip())

	with open(target_aids_list_path, 'r') as fp:
		for line in fp.readlines():
			tmp = line.strip().split('\t')
			# only add if pfam is in choosen_pfamss
			if tmp[-1] in choosen_pfams:
				target_aids[tmp[0]] = tmp[1].split()
				pfam_targets[tmp[-1]].append(tmp[0])

	return target_aids, pfam_targets


# processing individual assay
def read_file(filename, aid):
	""" reads raw assay file and process each assay file differently based on header info
		finally prints PUBCHEM_CID, SMILES, PUBCHEM_ACTIVITY_OUTCOME, PubChem Standard Value/Potency/Average IC50/IC50
		if the header contains 
	"""
	smiles, activity = list(), list()

	with open(filename, 'r', errors='ignore') as f:
		#f = (x.replace('\0', '') for x in csvfile)
		data = list(csv.reader(f, delimiter=',', quotechar='"', \
				quoting=csv.QUOTE_ALL, skipinitialspace=True))

		header = data[0]

		for i in range(len(data)):
			if data[i][0].isnumeric():
				start = i 	# store the start of data table
				break

		cols = [] # columns PUBCHEM_CID, PUBCHEM_ACTIVITY_OUTCOME
		cols.append(header.index("PUBCHEM_CID"))
		cols.append(header.index("PUBCHEM_ACTIVITY_OUTCOME"))
		phenotype_flag = False
		phenotype_counts = {k:0 for k in ['Inhibitor', 'Activator', 'Fluorescent', 'Cytotoxic', 'Inactive', 'Inconclusive', 'Antagonist', 'Quencher']}
		count_undetermined_phenotype = 0 # store counts of CIDs whose phenotype could not be determined using majority voting
		count_other_labels = 0 	# store counts of CIDs labelled other than 'Active' or 'Inactive' in PUBCHEM_ACTIVITY_OUTCOME
		count_missing_cid = 0 	# store counts of entries where PUBCHEM_CID is NULL
		count_missing_smiles = 0 	# store counts of CIDs whose mapping to SMILES are not present
		activity_counts = {k:0 for k in ['Active', 'Inactive', 'Inconclusive', 'Unspecified', 'Probe']}

		if "Phenotype" in header:
			phenotype_flag = 1
			cols.append(header.index("Phenotype"))
		# also need to handle some assays which has replicates in a row
		elif (any("Phenotype" in _ for _ in header)):
			phenotype_flag = 2
			ind = [header.index(_) for _ in header if "Phenotype" in _]
			cols.extend(ind)


		for row in data[start:]:
			# counts for debugging
			if row[cols[0]] == "":
				count_missing_cid += 1
			elif row[cols[0]] not in cid_smiles:
				count_missing_smiles += 1

			activity_counts[row[cols[1]]] += 1

			phenotype = ""
			# only consider Inhibitor compounds if column 'Phenotype' exists
			if phenotype_flag == 1:
				# capitalize first letter to make uniform
				phenotype = row[cols[2]].capitalize()
				phenotype = phenotype.replace("Inhitibor", "Inhibitor")

				if phenotype == "":
					count_undetermined_phenotype += 1
					continue

				phenotype_counts[phenotype] += 1
				if (phenotype != "Inhibitor" and phenotype != "Inactive"):
					continue

			# handle replicate assays case
			elif phenotype_flag == 2:
				# perform majority voting to determine phenotype but handle missing cols
				# some of these cols don't have any value, exclude those
				pheno_cols = np.array(row)[cols[2:]]
				pheno_cols = pheno_cols[pheno_cols != ""]
				uniq_pheno_values, uniq_pheno_counts = np.unique(pheno_cols, return_counts=True)

				# check for ties
				if len(uniq_pheno_values) > 1:
					# there is a tie, ignore the row
					if len(uniq_pheno_counts) != len(set(uniq_pheno_counts)):
						count_undetermined_phenotype += 1
						continue
					else:
						phenotype = uniq_pheno_values[np.argmax(uniq_pheno_counts)]
				else:
					phenotype = uniq_pheno_values[0]

				phenotype = phenotype.capitalize()
				phenotype = phenotype.replace("Inhitibor", "Inhibitor")

				phenotype_counts[phenotype] += 1
				# check if max voted phenotype is Inhibitor or inactive, otherwise continue
				if phenotype != "Inhibitor" and phenotype != "Inactive":
					continue



			# store smiles and activity labels
			if row[cols[0]] and row[cols[0]] in cid_smiles:
				smiles.append(cid_smiles[row[cols[0]]])
				if row[cols[1]] == "Active":
					activity.append("1")
				elif row[cols[1]] == "Inactive":
					activity.append("0")
				else:
					count_other_labels += 1
				#print(','.join([cid_smiles[row[cols[0]]]] + [row[c] for c in cols]), file=fout)

	# these counts does not correspond to unique compounds..
	with open(log_aid_wise, 'a') as f:
		print(f'{aid},{len(data[start:])}', end=',', file=f)
		for k,v in activity_counts.items():
			print(v, end=',', file=f)

		for k,v in phenotype_counts.items():
			print(v, end=',', file=f)
		print(f'{count_undetermined_phenotype},{count_missing_cid},{count_missing_smiles}', end=',', file=f)
		print(f'{activity.count("1")},{activity.count("0")}', file=f)

	return smiles, activity


def clean_duplicates(smiles, labels, target):
	smiles_labels = defaultdict(list)
	final_smiles, final_labels = list(), list()

	count_disparate_labels = 0
	count_same_labels = 0

	for sm, l in zip(smiles, labels):
		smiles_labels[sm].append(l)

	for sm, l in smiles_labels.items():
		# for disparate labels
		if len(l) > 1 and len(set(l)) > 1:
			count_disparate_labels += 1
			# check if majority voting can be done
			# else discard that compound
			if l.count("1") != l.count("0"):
				final_labels.append(max(set(l), key=l.count))
				final_smiles.append(sm)

		else:
			if len(l) > 1:
				count_same_labels += 1
			final_labels.append(l[0])
			final_smiles.append(sm)

	return final_smiles, final_labels, count_same_labels, count_disparate_labels



def process_combine_assays(assay_dir, aids, target, output_dir):
	combined_smiles, combined_labels = list(), list()
	combined = {}

	for aid in aids:
		sm, labels = read_file(os.path.join(assay_dir, f'AID_{aid}.csv'), aid)
		combined_smiles.extend(sm)
		combined_labels.extend(labels)

	final_smiles, final_labels, count_same_labels, count_disparate_labels \
		= clean_duplicates(combined_smiles, combined_labels, target)

	with open(log_target_wise, 'a') as f:
		print(f'{target},{len(combined_labels)},{combined_labels.count("1")},{combined_labels.count("0")}', end=',', file=f)
		print(f'{count_same_labels},{count_disparate_labels},{len(final_labels)},{final_labels.count("1")},{final_labels.count("0")}', file=f)

	with open(os.path.join(output_dir, f'{target}.csv'), 'w') as f:
		print("smiles,target", file=f)
		for sm, l in zip(final_smiles, final_labels):
			print(sm + "," + l, file=f)


def process_assays(assay_dir, cid_smiles_path, choosen_pfam_path, target_aids_list_path, output_dir):
	with open(log_aid_wise, 'w') as f:
		print('aid,#tested,#Active,#Inactive,#Inconclusive,#Unspecified,#Probe', end=',', file=f)
		print('#Inhibitor,#Activator,#Fluorescent,#Cytotoxic,#Inactive,#Inconclusive,#Antagonist,#Quencher', end=',', file=f)
		print('#undetermined_phenotype,#missing_cid,#missing_smiles,#active_smiles,#inactive_smiles', file=f)

	with open(log_target_wise, 'w') as f:
		print('protacxn,#tested,#active,#inactive,#smiles_same_labels,#smiles_disparate_labels,#final_total,#final_active,#final_inactive', file=f)

	read_cid_smiles(cid_smiles_path)
	target_aids, pfam_targets = read_target_aids_pfam(target_aids_list_path, choosen_pfam_path)

	for pfam, targets in pfam_targets.items():
		for target in targets:
			process_combine_assays(assay_dir, target_aids[target], target, output_dir)


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--assay_dir', type=str, required=True,
			help='Directory with raw assay .csv files containing data')
	parser.add_argument('--cid_smiles_path', type=str, required=True,
			help='Path to file containing mapping of CID and SMILES')
	parser.add_argument('--choosen_pfam_path', type=str, required=True,
			help='Path to file containing list of choosen pfams')
	parser.add_argument('--output_dir', type=str, required=True,
			help='Output directory to store processed files')
	parser.add_argument('--target_aids_list_path', type=str, required=True,
			help='Path to file containing list of AIDs grouped by target')

	process_assays(**vars(parser.parse_args()))
