import os
import sys
import csv
from argparse import ArgumentParser
from collections import defaultdict
from rdkit import Chem


# global dict() to store mappings of CID and SMILES
cid_smiles = {}

# read cid_smiles list into memory
def standardize_smiles(cid_smiles_path, output_path):
	with open(cid_smiles_path, 'r') as fp:
		count_invalid_smiles = 0
		count = 0

		for line in fp.readlines():
			if count % 50000 == 0:
				print(f'Reading line: {count}')
			count += 1
			tmp = line.strip().split('\t')
			try:
				cid_smiles[tmp[0]] = Chem.MolToSmiles(Chem.MolFromSmiles(tmp[1]))
			except:
				count_invalid_smiles += 1

	with open(output_path, 'w') as f:
		for cid, smiles in cid_smiles.items():
			print(cid + "\t" + smiles, file=f)
	print(f'#Invalid SMILES = {count_invalid_smiles}')


if __name__ == '__main__':
	standardize_smiles(sys.argv[1], sys.argv[2])
