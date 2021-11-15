import torch
import torch.nn as nn
import numpy as np

# for reproducibility
SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)

class Baseline(nn.Module):
	def __init__(self, args, device='cpu', hidden_size=None):
		super(Baseline, self).__init__()
		hidden_dim = 0
		if args.use_features == 'rdkit_2d' or args.use_features=='rdkit_2d_normalized':
			hidden_dim = 200
		elif args.use_features == 'morgan' or args.use_features == 'morgan_count':
			hidden_dim = 2048
		elif hidden_size:
			hidden_dim = hidden_size

		self.ff = nn.Linear(hidden_dim, 1)
		self.device = device
		self.args = args

	def forward(self, features):
		features = torch.from_numpy(np.stack(features)).float().to(self.device)
		return self.ff(features)
