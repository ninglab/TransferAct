from utils.metrics import *
from features.featurization import mol2graph

def evaluate(model, test_dataloader, args, draw=False):
	model.eval()
	correct = 0

	true_ic = []
	pred_scores = []
	n_pairs = 0

	for batch in test_dataloader:
		true_scores = [d.ic for d in batch]
		mols = [d.smiles for d in batch]
		features = [d.features for d in batch]

		if args.model == 'baseline':
			pred = model(features).data.cpu().flatten().tolist()
		else:
			molgraph = mol2graph(true_scores, mols)
			pred = model(molgraph, features, draw).data.cpu().flatten().tolist()

		true_ic.extend(true_scores)
		pred_scores.extend(pred)

	ci = compute_CI(pred_scores, true_ic)
	recall_top5 = compute_recall_topk(pred_scores, true_ic, 5)
	recall_top10 = compute_recall_topk(pred_scores, true_ic, 10)
	recall_3 = compute_recall(pred_scores, true_ic, k=3)
	recall_5 = compute_recall(pred_scores, true_ic, k=5)
	recall_10 = compute_recall(pred_scores, true_ic, k=10)
#	rank_3 = compute_rankalign(pred_scores, true_ic, k=3)
#	rank_5 = compute_rankalign(pred_scores, true_ic, k=5)
#	rank_10 = compute_rankalign(pred_scores, true_ic, k=10)
	ndcg_top5 = compute_ndcg_topk(pred_scores, true_ic, 5)
	ndcg_top10 = compute_ndcg_topk(pred_scores, true_ic, 10)
	ndcg_3 = compute_ndcg(pred_scores, true_ic, k=3)
	ndcg_5 = compute_ndcg(pred_scores, true_ic, k=5)
	ndcg_10 = compute_ndcg(pred_scores, true_ic, k=10)


	return pred_scores, true_ic,\
		{'CI' : ci, 'RECALL@5%': recall_top5, 'RECALL@10%': recall_top10,\
		'RECALL@3': recall_3, 'RECALL@5': recall_5, 'RECALL@10': recall_10, \
	#	'RANK_OVERLAP@3': rank_3, 'RANK_OVERLAP@5': rank_5, 'RANK_OVERLAP@10': rank_10, \
		'NDCG@5%': ndcg_top5, 'NDCG@10%': ndcg_top10, \
		'NDCG@3': ndcg_3, 'NDCG@5': ndcg_5, 'NDCG@10': ndcg_10}

