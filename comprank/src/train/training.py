import numpy as np
from torch.autograd import Variable
import torch
from torch.nn.utils import clip_grad_norm_

from features.features_generators import *
from features.featurization import mol2graph
from utils.nn_utils import compute_pnorm, compute_gnorm
from torchviz import make_dot

def unpair(batch):
	mols1 = [d.smiles for d in list(zip(*batch))[0]]
	mols2 = [d.smiles for d in list(zip(*batch))[1]]

	ics1 = [d.ic for d in list(zip(*batch))[0]]
	ics2 = [d.ic for d in list(zip(*batch))[1]]

	features1 = [d.features for d in list(zip(*batch))[0]]
	features2 = [d.features for d in list(zip(*batch))[1]]

	list_mols = []
	for i in mols1+mols2:
		if i not in list_mols:
			list_mols.append(i)
	pos, neg = [], []

	for mol1, mol2 in zip(mols1, mols2):
		pos.append(list_mols.index(mol1))
		neg.append(list_mols.index(mol2))

	list_ics = [0]*len(list_mols)
	for i, (ic1, ic2) in enumerate(list(zip(ics1, ics2))):
		list_ics[pos[i]] = ic1
		list_ics[neg[i]] = ic2

	list_features = [None]*len(list_mols)
	for i, (f1, f2) in enumerate(list(zip(features1, features2))):
		if f1 is not None:
			list_features[pos[i]] = f1[:]
		if f2 is not None:
			list_features[neg[i]] = f2[:]

	return pos, neg, list_mols, list_ics, list_features


def train_step(model, train_dataloader, criterion, optimizer, args):
	model.train()
	total_loss = 0
	labels = None

	for i, batch in enumerate(train_dataloader):
		if batch is None:
			continue

		if args.loss_fn == 'pair':
			mols1 = [d.smiles for d in list(zip(*batch))[0]]
			mols2 = [d.smiles for d in list(zip(*batch))[1]]

			ics1 = [d.ic for d in list(zip(*batch))[0]]
			ics2 = [d.ic for d in list(zip(*batch))[1]]

			features1 = [d.features for d in list(zip(*batch))[0]]
			features2 = [d.features for d in list(zip(*batch))[1]]


			if args.model != 'baseline':
				molgraph1, molgraph2 = mol2graph(ics1, mols1), mol2graph(ics2, mols2)

		elif args.loss_fn == 'pair2':
			pos, neg, list_mols, list_ics, list_features = unpair(batch)

			if args.model != 'baseline':
				molgraph = mol2graph(list_ics, list_mols)


		elif args.loss_fn == 'pairc': #npair
			mols = [d.smiles for d in batch]
			features = [d.features for d in batch]
			labels = [d.label for d in batch]
			ics = [d.ic for d in batch]

			if args.model != 'baseline':
				molgraph = mol2graph(ics, mols)

		optimizer.zero_grad()
		batch_loss = 0

		if args.loss_fn == 'pair':
			if args.model == 'baseline':
				pred1 = model(features1)
				pred2 = model(features2)
			else:
				pred1 = model(molgraph1, features1)
				pred2 = model(molgraph2, features2)

			batch_loss = torch.mean(criterion(pred1, pred2), dim=0)

		if args.loss_fn == 'pair2':
			if args.model == 'baseline':
				pred = model(list_features)
			else:
				pred = model(molgraph, list_features)

			pred1 = pred[pos]
			pred2 = pred[neg]
			batch_loss = torch.mean(criterion(pred1, pred2), dim=0)

		elif args.loss_fn == 'pairc': #npair
			if args.model == 'baseline':
				pred = model(features)
			else:
				pred = model(molgraph, features)

			batch_loss = torch.mean(criterion(pred, ics), dim=0) #npair
			#batch_loss = torch.mean(criterion(pred, labels), dim=0) #npair

		## regularization
		if args.regularization:
			batch_loss = batch_loss + args.regularization*compute_pnorm(model)**2

		#make_dot(batch_loss, params=dict(model.named_parameters())).render("attached", format='png')

		batch_loss.backward()
		'''
		if args.grad_clip:
			clip_grad_norm_([p for p in model.parameters() if p.grad is not None], args.grad_clip)
		'''
		optimizer.step()
		total_loss += batch_loss.item()

	return total_loss/len(train_dataloader)


