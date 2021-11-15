import torch
import torch.nn as nn

from models.mpnn import MPNN
from models.scoring import ScoreNet

class CompRankNet(nn.Module):
	def __init__(self, args, device='cpu', size=None):
		super(CompRankNet, self).__init__()
		self.encoder = MPNN(args, device)
		self.decoder = ScoreNet(args, size)

	def forward(self, x, features, draw=False):
		emb = self.encoder(x, features, draw)
		if self.decoder:
			score = self.decoder(emb)
			return score
		return emb

