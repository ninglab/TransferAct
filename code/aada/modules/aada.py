import torch
import torch.nn as nn
import torch.nn.functional as f
from typing import Optional

from .mpn import MPN
from .model import MoleculeModel
from aada.utils.nn_utils import initialize_weights
from .utils import binary_accuracy
from .domain_discriminator import DomainDiscriminator
from .grl import WarmStartGradientReverseLayer, GradientReversalLayer
from .classifier import Classifier

class AADALoss(nn.Module):

	def __init__(self, args):
		super(AADALoss, self).__init__()

		if args.lamda !=0 and not args.exclude_local:
			self.grl = WarmStartGradientReverseLayer(alpha=1, lo=0., hi=args.lamda, max_iters=1000, auto_step=True)
		if args.lamda !=0 and not args.exclude_global:
			self.grl2 = WarmStartGradientReverseLayer(alpha=1, lo=0., hi=args.lamda, max_iters=1000, auto_step=True)

		self.alpha                                = args.alpha
		self.lamda                                = args.lamda
		self.bce                                  = nn.BCELoss()
		self.mpn_shared                           = args.mpn_shared
		self.exclude_local                        = args.exclude_local
		self.exclude_global                       = args.exclude_global
		self.classifier                           = Classifier(args)
		self.local_domain_discriminator_accuracy  = None
		self.global_domain_discriminator_accuracy = None

		#self.feature_generator = MoleculeModel(args, featurizer=True) # call molecule encoder to return features from last layer
		if self.mpn_shared:
			self.feature_generator = MPN(args)
		else:
			self.source_feature_generator = MPN(args)
			self.target_feature_generator = MPN(args)

		if self.lamda != 0 and not self.exclude_local:
			self.discriminator_local  = DomainDiscriminator(args, dim_input=args.hidden_size, \
						dim_hidden=args.local_discriminator_hidden_size, dim_output=args.hidden_size, \
						num_layers=args.local_discriminator_num_layers)

		if self.lamda != 0 and not self.exclude_global:
			self.discriminator_global = DomainDiscriminator(args, dim_input=args.hidden_size, \
						dim_hidden=args.global_discriminator_hidden_size, dim_output=1, \
						num_layers=args.global_discriminator_num_layers)
		initialize_weights(self)


	def forward_train(self, source_batch,
		target_batch):

		source_mol_batch, source_features_batch, s_target_batch, source_atom_descriptors_batch = \
			source_batch.batch_graph(), source_batch.features(), source_batch.targets(), source_batch.atom_descriptors()

		target_mol_batch, target_features_batch, t_target_batch, target_atom_descriptors_batch = \
			target_batch.batch_graph(), target_batch.features(), target_batch.targets(), target_batch.atom_descriptors()

		if self.mpn_shared:
			f_s = self.feature_generator(source_mol_batch, source_features_batch, source_atom_descriptors_batch)
			f_t = self.feature_generator(target_mol_batch, target_features_batch, target_atom_descriptors_batch)
		else:
			f_s = self.source_feature_generator(source_mol_batch, source_features_batch, source_atom_descriptors_batch)
			f_t = self.target_feature_generator(target_mol_batch, target_features_batch, target_atom_descriptors_batch)

		# compute features out of GNN (DMPNN or A-DMPNN)
		#f_t = self.feature_generator(target_mol_batch, target_features_batch, target_atom_descriptors_batch)

		if self.lamda != 0 and not self.exclude_local:
			# gradient reversal layer: only reverses the gradient from local discriminator
			f  = self.grl(torch.cat((f_s, f_t), dim=0))
			dl = self.discriminator_local(f)

			dl_s, dl_t = dl.chunk(2, dim=0)

			dl_label_s = torch.ones((f_s.size(0), f_s.size(1))).to(f_s.device)
			dl_label_t = torch.zeros((f_t.size(0), f_t.size(1))).to(f_t.device)

			weight_s = 1 + entropy(torch.cat((dl_s, 1-dl_s))) 	# higher score => higher transferability
			weight_t = 1 + entropy(torch.cat((dl_t, 1-dl_t)))

			fg_s = f_s*(1+weight_s.detach())
			fg_t = f_t*(1+weight_t.detach())
		else:
			f = torch.cat((f_s, f_t), dim=0)
			fg_s, fg_t = f_s.clone(), f_t.clone()

		if self.lamda != 0 and not self.exclude_global:
			# gradient reversal layer: only reverses the gradient from global discriminator
			fg = self.grl2(torch.cat((fg_s, fg_t), dim=0))
			dg = self.discriminator_global(fg)
			dg_s, dg_t = dg.chunk(2, dim=0)

			dg_label_s = torch.ones((fg_s.size(0), 1)).to(f_s.device)
			dg_label_t = torch.zeros((fg_t.size(0), 1)).to(f_t.device)


		# domain-wise classifier
		labels_s = self.classifier(fg_s)
		labels_t = self.classifier(fg_t)

		s_target_batch = torch.Tensor(s_target_batch).to(f_s.device)
		t_target_batch = torch.Tensor(t_target_batch).to(f_t.device)

		source_classification_loss = self.bce(labels_s, s_target_batch)
		target_classification_loss = self.bce(labels_t, t_target_batch)
		classification_loss        = self.alpha*source_classification_loss + target_classification_loss

		adversarial_loss = local_adversarial_loss = global_adversarial_loss = 0
		source_local_adversarial_loss = target_local_adversarial_loss = source_global_adversarial_loss = target_global_adversarial_loss = 0

		if self.lamda != 0 and not self.exclude_local:
			source_local_adversarial_loss = self.bce(dl_s, dl_label_s)
			target_local_adversarial_loss = self.bce(dl_t, dl_label_t)
			local_adversarial_loss        = source_local_adversarial_loss + target_local_adversarial_loss
			adversarial_loss             += local_adversarial_loss
			#self.local_domain_discriminator_accuracy = 0.5 * (binary_accuracy(dl_s, dl_label_s) + binary_accuracy(dl_t, dl_label_t))

		if self.lamda != 0 and not self.exclude_global:
			source_global_adversarial_loss = self.bce(dg_s, dg_label_s)
			target_global_adversarial_loss = self.bce(dg_t, dg_label_t)
			global_adversarial_loss        = source_global_adversarial_loss + target_global_adversarial_loss
			adversarial_loss              += global_adversarial_loss
			#self.global_domain_discriminator_accuracy = 0.5 * (binary_accuracy(dg_s, dg_label_s) + binary_accuracy(dg_t, dg_label_t))

		return classification_loss + self.lamda*adversarial_loss, source_classification_loss, target_classification_loss, \
			source_local_adversarial_loss, target_local_adversarial_loss, source_global_adversarial_loss, target_global_adversarial_loss #+ 0.1*entropy_loss


	def forward_test(self, val_batch: torch.Tensor, featurizer: Optional[bool]=False) -> torch.FloatTensor:

		val_mol_batch, val_features_batch, val_target_batch, val_atom_descriptors_batch = \
			val_batch.batch_graph(), val_batch.features(), val_batch.targets(), val_batch.atom_descriptors()

		with torch.no_grad():
			if self.mpn_shared:
				f = self.feature_generator(val_mol_batch, val_features_batch, val_atom_descriptors_batch)
			else:
				f = self.target_feature_generator(val_mol_batch, val_features_batch, val_atom_descriptors_batch)

			if self.lamda != 0 and not self.exclude_local:
				p        = self.discriminator_local(f)
				weight   = 1 + entropy(torch.cat((p, 1-p)))
				f_scaled = f*(1+weight)

				if featurizer:
					return f_scaled

				val_pred = self.classifier(f_scaled)
			else:
				val_pred = self.classifier(f)

				if featurizer:
					return f

		return val_pred


	def forward(self, source_batch: torch.Tensor,
			target_batch: Optional[torch.Tensor]=None, training: Optional[bool]=False,
			featurizer: Optional[bool]=False):

		if training:
			return self.forward_train(source_batch, target_batch)
		else:
			return self.forward_test(source_batch, featurizer)


def entropy(predictions):
	eps = 1e-5
	H   = -predictions * torch.log(predictions + eps)
	tmp = H.chunk(2, dim=0)
	return tmp[0] + tmp[1]

