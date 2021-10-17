from typing import List, Dict, Optional
import torch.nn as nn
import torch
from aada.utils.nn_utils import get_activation_function, initialize_weights

class DomainDiscriminator(nn.Module):


	def __init__(self, args, dim_input: int, dim_hidden: int, \
			dim_output:int, num_layers:Optional[int] = 1):

		"""
                :param args: A :class:`~aada.args.TrainArgs` object containing model arguments.
                :param dim_input: Vector dimension of the input features generated by GNN.
                :param dim_hidden: Number of units of the hidden layer(s) of FFN implementing the discriminator
		:param dim_output: Vector dimension of the output probability vector
					( = dim_input for local discriminator
					  = 1 for global discriminator)
                """

		super(DomainDiscriminator, self).__init__()
		self.activation = get_activation_function(args.activation)
		dropout = nn.Dropout(args.dropout)
		self.sigmoid = nn.Sigmoid()
		#bn =  nn.BatchNorm1d(dim_hidden)

		# Create FFN layers
		if num_layers == 1:
			ffn = [dropout, nn.Linear(dim_input, dim_output)]
		else:
			ffn = [dropout, nn.Linear(dim_input, dim_hidden)]

			for _ in range(num_layers - 2):
				ffn.extend([self.activation,
					dropout,
			#		bn,
					nn.Linear(dim_hidden, dim_hidden),
					])

			ffn.extend([self.activation,
				dropout,
			#	bn,
				nn.Linear(dim_hidden, dim_output),
				])

		# Create FFN model
		self.ffn = nn.Sequential(*ffn)

		#initialize_weights(self)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.ffn(x)
		return self.sigmoid(x)

	def get_parameters(self) -> List[Dict]:
		return [{"params": self.parameters(), "lr_mult": 1.}]
