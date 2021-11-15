import torch
import torch.nn as nn
import torch.nn.functional as f

import numpy as np

class PairLoss(nn.Module):
	def __init__(self):
		super(PairLoss, self).__init__()
		self.loss = nn.Softplus(beta=1, threshold=50)

	def forward(self, score1, score2):
		#return torch.log(1 + torch.exp(-(score1 - score2)))
		return self.loss(-(score1 - score2))


class PairLoss_(nn.Module):
	def __init__(self):
		super(PairLoss_, self).__init__()
		self.loss = nn.Softplus(beta=1, threshold=50)

	def forward(self, scores, ics):
		pos, neg = self.get_pairs(ics)
		positives = scores[pos]
		negatives = scores[neg]
		return self.loss(-(positives - negatives))

	@staticmethod
	def get_pairs(ics):
		pos = []
		neg = []
		for i in range(len(ics)):
			for j in range(len(ics)):
				if ics[i] < ics[j]:
					pos.append(i)
					neg.append(j)
		return pos, neg

class NpairLoss(nn.Module):
	def __init__(self, l2_reg=0.02):
		super(NpairLoss, self).__init__()
		self.loss = nn.Softplus(beta=1, threshold=50)
		self.l2_reg = l2_reg

	def forward(self, embeddings, target):
		n_pairs, n_negatives = self.get_n_pairs(target)

		anchors = embeddings[n_pairs[:, 0]]    # (n, embedding_size)
		positives = embeddings[n_pairs[:, 1]]  # (n, embedding_size)
		negatives = embeddings[n_negatives]    # (n, n-1, embedding_size)

		losses = self.n_pair_loss(anchors, positives, negatives) #\
	#		+ self.l2_reg * self.l2_loss(anchors, positives)

		return losses

	@staticmethod
	def get_n_pairs(labels):
		n_pairs = []

		for label in set(labels):
			label_mask = (labels == label)
			label_indices = np.where(label_mask)[0]
			if len(label_indices) < 2:
				continue
			anchor, positive = np.random.choice(label_indices, 2, replace=False)
			n_pairs.append([anchor, positive])

		n_pairs = np.array(n_pairs)

		n_negatives = []
		for i in range(len(n_pairs)):
			negative = np.concatenate([n_pairs[:i, 1], n_pairs[i+1:, 1]])
			n_negatives.append(negative)

		n_negatives = np.array(n_negatives)

		return torch.LongTensor(n_pairs), torch.LongTensor(n_negatives)


	@staticmethod
	def n_pair_loss(anchor, positive, negative):
		anchor = torch.unsqueeze(anchor, dim=1)
		positive = torch.unsqueeze(positive, dim=1)

		#x1 = f.softplus(anchor-positive, 1, 50)
		#x2 = f.softplus(anchor-negative, 1, 50)
		#x = x1 - x2
		#x = negative-positive
		x = torch.matmul(anchor, (negative-positive).transpose(1, 2))
		loss = torch.sum(torch.log(1+torch.exp(x)))
		return loss

	@staticmethod
	def l2_loss(anchors, positives):
		return torch.sum(anchors**2 + positives**2) / anchors.shape[0]
