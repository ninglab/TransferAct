import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

class ScoreNet(nn.Module):

	def __init__(self, args, input_size=None):
		super(ScoreNet, self).__init__()
		dim = args.hidden_dim

		if args.pooling == 'hier-attention':
			dim *= (args.message_steps-1)

		if args.pooling == 'all-concat':
			dim *= 3

		if args.use_features == 'rdkit_2d' or args.use_features=='rdkit_2d_normalized':
			dim += 200
		elif args.use_features == 'morgan' or args.use_features == 'morgan_count':
			dim += 2048
		elif args.use_features == 'morgan_tanimoto_bioassay':
			dim += input_size

		self.ff = nn.Linear(dim, 1)

	def forward(self, emb):
		score = self.ff(emb)
		return score

