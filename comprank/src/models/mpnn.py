import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GlobalAttention, SAGPooling

import numpy as np

from features.featurization import get_atom_fdim, get_bond_fdim
from utils.nn_utils import index_select_ND
from utils.common import viz_atom_attention

class MPNN(nn.Module):
	def __init__(self, args, device='cpu', atom_fdim = None, bond_fdim = None):
		super(MPNN, self).__init__()
		self.atom_fdim = atom_fdim or get_atom_fdim()
		self.bond_fdim = bond_fdim or get_bond_fdim(atom_messages=args.atom_messages)
		self.atom_messages = args.atom_messages
		self.hidden_dim = args.hidden_dim
		self.steps = args.message_steps
		self.layers_per_message = 1
		self.device = device
		self.pooling = args.pooling
		self.use_features = args.use_features
		self.viz_dir = args.viz_dir

		self.activation = nn.ReLU()
		self.dropout_layer = nn.Dropout(p=args.dropout)

		input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
		self.W_i = nn.Linear(input_dim, self.hidden_dim)
		if self.atom_messages:
			w_m_input_size = self.hidden_dim + self.bond_fdim
		else:
			w_m_input_size = self.hidden_dim

		self.W_m = nn.Linear(w_m_input_size, self.hidden_dim)
		self.W_a = nn.Linear(self.hidden_dim + self.atom_fdim, self.hidden_dim)

		if self.pooling == 'self-attention' or self.pooling == 'hier-attention' or self.pooling == 'all-concat':
			self.W_att_1 = nn.Linear(self.hidden_dim, args.attn_dim)
			self.W_att_2 = nn.Linear(args.attn_dim, 1)

		self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_dim), requires_grad=False)

	def aggregate(self, message, a2b, b2a, b2revb):
		if self.atom_messages:
			nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
			nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
			nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
			message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
		else:
			# m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
			neigh_message = index_select_ND(message, a2b)	# num_atoms x max_num_bonds x hidden_dim
			agg_message   = neigh_message.sum(dim = 1)	# num_atoms x hidden_dim
			rev_message   = message[b2revb]			# num_bonds x hidden_dim
			message	      = agg_message[b2a] - rev_message	# num_bonds x hidden_dim
		return message

	def update(self, input, message):
		message = self.W_m(message)
		message = self.activation(input + message)
		message = self.dropout_layer(message)
		return message

	def _readout(self, atom_h, smiles=None, ic=None, draw=False):
		if self.pooling == 'mean':
			return atom_h.mean(dim=0)

		elif self.pooling == 'sum':
			return atom_h.sum(dim=0)

		elif self.pooling == 'max':
			return atom_h.max(dim=0)[0]

		elif self.pooling == 'attention':
			scores = torch.softmax(torch.matmul(atom_h, atom_h.T), dim=1)
			x = torch.matmul(scores, atom_h)
			return torch.mean(x, dim=0)
			#return torch.cat((atom_h, x), dim=1).mean(dim=0)

		elif self.pooling == 'self-attention' or self.pooling == 'hier-attention':
			temp_h = self.activation(self.W_att_1(atom_h))
			scores = torch.softmax(self.W_att_2(temp_h), dim=0)
			x = (1+scores)*atom_h
			if smiles and ic and self.viz_dir and draw:
				viz_atom_attention(self.viz_dir, smiles, ic, scores)
			return torch.sum(x, dim=0)

		elif self.pooling == 'all-concat':
			scores = torch.softmax(self.W_att(atom_h), dim=0)
			x = torch.sum(scores*atom_h, dim=0)
			t = torch.cat([atom_h.mean(dim=0), atom_h.max(dim=0)[0], x])
			return t

	def readout(self, atom_h, a_scope, mol_graph, draw):
		mol_vecs = []
		for i, (a_start, a_size) in enumerate(a_scope):
			if a_size == 0:
				mol_vecs.append(self.cached_zero_vector)
			else:
				temp_h = atom_h.narrow(0, a_start, a_size)
				mol_vec = self._readout(temp_h, mol_graph.smiles_batch[i],\
							mol_graph.ic_batch[i], draw)
				mol_vecs.append(mol_vec)

		mol_vecs = torch.stack(mol_vecs, dim = 0)
		return mol_vecs


	def forward(self, mol_graph, features=None, draw=False):

		f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components(atom_messages=self.atom_messages)
		f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.to(self.device), f_bonds.to(self.device),\
							a2b.to(self.device), b2a.to(self.device), b2revb.to(self.device)

		if self.atom_messages:
			a2a = mol_graph.get_a2a().to(self.device)

		# Input
		if self.atom_messages:
			input = self.W_i(f_atoms)  # num_atoms x hidden_dim
		else:
			input = self.W_i(f_bonds)  # num_bonds x hidden_dim

		message = self.activation(input)
		atom_h_timesteps = []

		for step in range(self.steps - 1):
			aggregated_message = self.aggregate(message, a2b, b2a, b2revb)
			message = self.update(input, aggregated_message)

			incoming_a_message = index_select_ND(message, a2b)
			a_message = incoming_a_message.sum(dim=1)
			a_input = torch.cat([f_atoms, a_message], dim=1)
			atom_h = self.activation(self.W_a(a_input))
			atom_h = self.dropout_layer(atom_h)
			atom_h_timesteps.append(atom_h)

		## maybe add normalization layers??
		#if self.layer_norm:
		#	message = self.layer_norm(message)

		"""
		incoming_a_message = index_select_ND(message, a2b)
		a_message = incoming_a_message.sum(dim=1)
		a_input = torch.cat([f_atoms, a_message], dim=1)
		atom_h = self.activation(self.W_a(a_input))
		"""
		#readout
		mol_vecs = []
		if self.pooling == 'hier-attention':
			for _atom_h in atom_h_timesteps:
				temp = self.readout(_atom_h, a_scope, mol_graph, draw)
				mol_vecs.append(temp)
			mol_vecs = torch.cat([_ for _ in torch.stack(mol_vecs)], dim=1)

		else:
			mol_vecs = self.readout(atom_h_timesteps[-1], a_scope, mol_graph, draw)


		if self.use_features:
			features = torch.from_numpy(np.stack(features)).float().to(self.device)
			features_batch = features.to(mol_vecs)
			if len(features_batch.shape) == 1:
				features_batch = features_batch.view([1, features_batch.shape[0]])
			mol_vecs = torch.cat([mol_vecs, features_batch], dim=1)


		return mol_vecs
