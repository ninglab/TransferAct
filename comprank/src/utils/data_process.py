import os
import csv
from collections import defaultdict

import random
random.seed(123)

PROJ_DIR = os.path.join(os.getenv("HOME"), "CompRank/compgraphnet/")
DATA_DIR = os.path.join(PROJ_DIR, "data/")

def remove_duplicateic(comp_ic):
	ic_comp = defaultdict(list)
	for comp, ic in comp_ic.items():
		ic_comp[ic].append(comp)
	return {random.choice(v):k for k, v in ic_comp.items()}


def read_file(filename):
	cid_ic50 = {}
	with open(filename, 'r') as fp:
		lines = fp.readlines()
		header = lines[0].strip().split(',')

		ic50_field = 7

		if header[7] == 'IC50_Qualifier' or header[7] == 'IC50_Mean_Qualifier'\
				or header[7] == 'Qualifier':
			ic50_field = 8
		if header[7] == 'EC50 Replicate':
			ic50_filed = 10

		for line in lines[3:]:
			tmp = list(csv.reader([line]))[0]

			if tmp[3] == "Active" and tmp[2]:
				comp, ic50 = tmp[2], float(tmp[ic50_field])
				try:
					if comp in cid_ic50 and cid_ic50[comp] != ic50:
						cid_ic50[comp] = -1
					else:
						cid_ic50[comp] = ic50
				except:
					print(filename, tmp)

	# remove invalid, duplicates
	return {k:v for k,v in cid_ic50.items() if v > 0}


def create_pairs(compound_dict):
	pairs = []
	for cmp1, ic1 in compound_dict.items():
		for cmp2, ic2 in compound_dict.items():
			if ic1 < ic2 and cmp1 != cmp2:
				pairs.append((cmp1, cmp2, ic1, ic2))
	return pairs

def process(fl):
	fout = open(os.path.join(DATA_DIR, "active_unique", fl), 'w')

	cid_ic50 = read_file(os.path.join(DATA_DIR, "subset/", fl))

	cid_ic50 = remove_duplicateic(cid_ic50)

	for k, v in sorted(cid_ic50.items(), key=lambda x: x[1]):
		print(k + "," + str(v), file=fout)
	fout.close()

	if len(cid_ic50) >= 50:
		fout1 = open(os.path.join(DATA_DIR, "active_unique_50", fl), 'w')
		for k, v in sorted(cid_ic50.items(), key=lambda x: x[1]):
			print(k + "," + str(v), file=fout1)
		fout1.close()

	return cid_ic50

def main():
	for fl in os.listdir(os.path.join(DATA_DIR, "subset/")):
	#	fp = open(os.path.join(DATA_DIR, "pairs", fl), 'w')
		cid_ic50 = process(fl)
	#	pairs = create_pairs(cid_ic50)
	#	for pair in pairs:
	#		print(','.join(map(str, pair)), file=fp)
	#	fp.close()

if __name__ == '__main__':
	main()
