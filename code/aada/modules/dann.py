import torch
import torch.nn as nn
import torch.nn.functional as f
from typing import Optional

from .mpn import MPN
from .model import MoleculeModel
from .utils import binary_accuracy
from .domain_discriminator import DomainDiscriminator
from .grl import WarmStartGradientReverseLayer, GradientReversalLayer
from .classifier import Classifier

class DANNLoss(nn.Module):

	def __init__(self, args):
		super(DANNLoss, self).__init__()
		self.feature_generator 	  = MPN(args) # call MPN to return features from last layer
		self.discriminator = DomainDiscriminator(args, dim_input=args.hidden_size, \
						dim_hidden=args.global_discriminator_hidden_size, dim_output=1, \
						num_layers=args.global_discriminator_num_layers)

		self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=args.lamda, max_iters=1000, auto_step=True)
		self.lamda = args.lamda
		self.bce = nn.BCELoss()
		self.classifier = Classifier(args)
		self.domain_discriminator_accuracy = None

	def forward_train(self, source_batch,
		target_batch) -> torch.FloatTensor:

		source_mol_batch, source_features_batch, s_target_batch, source_atom_descriptors_batch = \
			source_batch.batch_graph(), source_batch.features(), source_batch.targets(), source_batch.atom_descriptors()

		f_s = self.feature_generator(source_mol_batch, source_features_batch, source_atom_descriptors_batch)

		target_mol_batch, target_features_batch, t_target_batch, target_atom_descriptors_batch = \
			target_batch.batch_graph(), target_batch.features(), target_batch.targets(), target_batch.atom_descriptors()

		# compute features out of GNN (DMPNN or A-DMPNN)
		f_t = self.feature_generator(target_mol_batch, target_features_batch, target_atom_descriptors_batch)

		# gradient reversal layer: only reverses the gradient from local discriminator
		f = self.grl(torch.cat((f_s, f_t), dim=0))

		dg = self.discriminator(f)
		dg_s, dg_t = dg.chunk(2, dim=0)

		dg_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
		dg_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)

		labels_s = self.classifier(f_s)
		labels_t = self.classifier(f_t)

		s_target_batch = torch.Tensor(s_target_batch).to(f_s.device)
		t_target_batch = torch.Tensor(t_target_batch).to(f_t.device)

		#entropy_loss = torch.mean(dg_s*entropy(torch.cat((labels_s, labels_t))))
		sample_score_s = 1 + entropy(torch.cat((dg_s, 1-dg_s)))
		sample_score_t = 1 + entropy(torch.cat((dg_t, 1-dg_t)))

		source_classification_loss = self.bce(labels_s, s_target_batch)
		target_classification_loss = self.bce(labels_t, t_target_batch)
		classification_loss = source_classification_loss

		source_adversarial_loss = self.bce(dg_s, dg_label_s)
		target_adversarial_loss = self.bce(dg_t, dg_label_t)
		adversarial_loss = source_adversarial_loss + target_adversarial_loss

		self.domain_discriminator_accuracy = 0.5 * (binary_accuracy(dg_s, dg_label_s) + binary_accuracy(dg_t, dg_label_t))

		return classification_loss + self.lamda*adversarial_loss, source_classification_loss, target_classification_loss, \
			source_adversarial_loss, target_adversarial_loss


	def forward_test(self, val_batch: torch.Tensor, featurizer: Optional[bool]=False) -> torch.FloatTensor:

		val_mol_batch, val_features_batch, val_target_batch, val_atom_descriptors_batch = \
			val_batch.batch_graph(), val_batch.features(), val_batch.targets(), val_batch.atom_descriptors()

		with torch.no_grad():
			f = self.feature_generator(val_mol_batch, val_features_batch, val_atom_descriptors_batch)

			if featurizer:
				return f

			val_pred = self.classifier(f)

		return val_pred


	def forward(self, source_batch: torch.Tensor,
			target_batch: Optional[torch.Tensor]=None, training: Optional[bool]=False,
			featurizer: Optional[bool]=False) -> torch.FloatTensor:

		if training:
			return self.forward_train(source_batch, target_batch)
		else:
			return self.forward_test(source_batch, featurizer)


def entropy(predictions):
	eps = 1e-5
	H = -predictions * torch.log(predictions + eps)
	tmp = H.chunk(2, dim=0)
	return tmp[0] + tmp[1]

