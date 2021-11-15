import numpy as np
from sklearn.metrics import ndcg_score


def compute_rankalign(pred_scores, true_ic, k=10):
	assert(k >= 1)
	pred_comp_rank = np.argsort(pred_scores, kind='mergesort')[::-1][:k]
	true_comp_rank = np.argsort(true_ic, kind='mergesort')[:k]
	return np.mean(pred_comp_rank == true_comp_rank)


def compute_ranking(pred_scores, true_ic):
	for i in range(len(true_ic)):
		count = 0
		all_ = 0
		for j in range(len(true_ic)):
			if true_ic[j] < true_ic[i]:
				all_ += 1
				if pred_scores[j] > pred_scores[i]:
					count += 1
		try:
			print(count, all_, count/float(all_))
		except:
			print(count, all_)

def compute_CI(pred_scores, true_ic):
	correct = 0
	n_pairs = 0

	for i in range(len(true_ic)):
		for j in range(len(true_ic)):
			if true_ic[i] < true_ic[j]:
				n_pairs += 1
				if pred_scores[i] > pred_scores[j]:
					correct += 1
	return correct/float(n_pairs)


def my_dcg_score(y_true, y_scores, k=10):
	order = np.argsort(y_scores)[::-1]
	y_true = np.take(y_true, order[:k])

def my_ndcg_score(y_true, y_scores, k=10):
	ideal = my_dcg_score(y_true, y_true, k)
	actual = my_dcg_score(y_true, y_scores, k)
	return actual/ideal

def compute_recall(pred_scores, true_ic, k=10):
	assert(k >= 1)
	pred_comp_rank = np.argsort(pred_scores, kind='mergesort')[::-1][:k]
	true_comp_rank = np.argsort(true_ic, kind='mergesort')[:k]
	return len(set(pred_comp_rank) & set(true_comp_rank))/float(k)

def compute_recall_topk(pred_scores, true_ic, K=10):
	assert(len(pred_scores) == len(true_ic))
	k = max(1, int(np.ceil(K/100.0*len(pred_scores))))
	assert(k >= 1)
	return compute_recall(pred_scores, true_ic, k)

def compute_ndcg(pred_scores, true_ic, k=10):
#	pred_topk = np.argsort(pred_scores)[::-1][:k]
	true_rel = np.argsort(np.argsort(np.negative(true_ic), kind='mergesort'))
	return ndcg_score(np.asarray([true_rel]), np.asarray([pred_scores]), k, ignore_ties=False)

def compute_ndcg_topk(pred_scores, true_ic, K=10):
	assert(len(pred_scores) == len(true_ic))
	k = max(1, int(np.ceil(K/100.0*len(pred_scores))))
	assert(k >= 1)
	return compute_ndcg(pred_scores, true_ic, k)
