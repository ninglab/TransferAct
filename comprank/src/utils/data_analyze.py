import os
import csv
import numpy as np
import matplotlib.pyplot as plt

PROJ_DIR = os.path.join(os.getenv("HOME"), "CompRank/compgraphnet/")
DATA_DIR = os.path.join(PROJ_DIR, "data/")

missing_cid = 0
invalid_icd = 0
missing_icd = 0
duplicate_cid_icd = 0
duplicate_cid = 0

def read_file(filename):
	global missing_cid, invalid_icd, missing_icd, duplicate_cid, duplicate_cid_icd

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
			if tmp[3] == 'Active':
				if not tmp[2]:
					missing_cid += 1
				if float(tmp[ic50_field]) <= 0:
					invalid_icd += 1
				if not tmp[ic50_field]:
					missing_icd += 1
				if tmp[2] and tmp[2] in cid_ic50:
					duplicate_cid += 1
					if float(tmp[ic50_field]) in cid_ic50[tmp[2]]:
						print(filename, tmp[2])
						duplicate_cid_icd += 1

			if tmp[3] == "Active" and tmp[2]:
				try:
					ic50 = float(tmp[ic50_field])
					if tmp[2] in cid_ic50:
						cid_ic50[tmp[2]].append(ic50)
					else:
						cid_ic50[tmp[2]] = [ic50]
				except:
					print(filename, tmp)

	return cid_ic50


def process(fl):
#	fout = open(os.path.join(DATA_DIR, "active", fl), 'w')
	cid_ic50 = read_file(os.path.join(DATA_DIR, "subset/", fl))

	return {k:v[0] for k,v in cid_ic50.items() if len(v) == 1}

def read_processed_file(fl):
	cid_ic = {}
	with open(fl, 'r') as fp:
		for line in fp.readlines():
			cid, ic = line.strip().split(',')
			cid_ic[cid] = float(ic)
	return cid_ic

def ic_stats(iclist):
	return np.asarray([np.min(iclist), np.max(iclist), np.mean(iclist), np.std(iclist)])

def main():
	counts = []
	stats = []
	for fl in os.listdir(os.path.join(DATA_DIR, "active_unique_50/")):
#		fp = open(os.path.join(DATA_DIR, "pairs", fl), 'w')
#		cid_ic50 = process(fl)
		cid_ic50 = read_processed_file(os.path.join(DATA_DIR, "active_unique_50/", fl))
		counts.append(len(cid_ic50))
		stats.append(ic_stats(list(cid_ic50.values())))


	fig, ax = plt.subplots(2, 2)
	ax[0,0].plot(list(zip(*stats))[0], 'o', markersize=2)
	ax[0,0].set_title('Min')
	ax[0,1].plot(list(zip(*stats))[1], 'o', markersize=2)
	ax[0,1].set_title('Max')
	ax[1,0].plot(list(zip(*stats))[2], 'o', markersize=2)
	ax[1,0].set_title('Mean')
	ax[1,1].plot(list(zip(*stats))[3], 'o', markersize=2)
	ax[1,1].set_title('Stdv')
	fig.savefig('stats_ic_uniq.png')

	stats = np.asarray(stats)
	print(ic_stats(stats))
	print(np.mean(counts), np.max(counts), np.min(counts), np.std(counts))
	print(np.histogram(counts, bins=[1, 5, 10, 25, 50, 100, 150, 200]))

if __name__ == '__main__':
	main()
	print('Missing CID =', missing_cid)
	print('Missing IC = ', missing_icd)
	print('Invalid IC = ', invalid_icd)
	print('Duplicate IC = ', duplicate_cid)
	print('Duplicate IC and CID = ', duplicate_cid_icd)
