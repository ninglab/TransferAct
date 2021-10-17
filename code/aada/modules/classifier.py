from typing import Tuple, Optional, List, Dict
import torch.nn as nn
import torch
from aada.utils.nn_utils import get_activation_function, initialize_weights

__all__ = ['Classifier']


class Classifier(nn.Module):

	def __init__(self, args, head: Optional[nn.Module] = None):

		"""
                :param args: A :class:`~aada.args.TrainArgs` object containing model arguments.
                :param head: A nn.Module implementing a pre-built feature scoring model.
                """
		super(Classifier, self).__init__()
		num_classes = 2 		# TODO: need to generalize this to multi-class
		dim_output  = 1 		# TODO: need to generalize this to multi-class

		self.activation = get_activation_function(args.activation)
		dropout = nn.Dropout(args.dropout)
		self.sigmoid = nn.Sigmoid()

		# Create FFN layers
		if args.ffn_num_layers == 1:
		#	ffn = [dropout, nn.Linear(args.hidden_size, dim_output)]
			ffn = [nn.Linear(args.hidden_size, dim_output)]
		else:
		#	ffn = [dropout, nn.Linear(args.hidden_size, args.ffn_hidden_size)]
			ffn = [nn.Linear(args.hidden_size, args.ffn_hidden_size)]

			for _ in range(args.ffn_num_layers - 2):
				ffn.extend([self.activation,
				#	dropout,
					nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
					])

			ffn.extend([self.activation,
			#	dropout,
				nn.Linear(args.ffn_hidden_size, dim_output),
				])

		if head:
			self.head = head
		else:
			self.head = nn.Sequential(*ffn)

		#initialize_weights(self)

	def forward(self, x: torch.Tensor):
		return torch.sigmoid(self.head(x))
