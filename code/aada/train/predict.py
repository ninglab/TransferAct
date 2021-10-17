from typing import List

import torch
from tqdm import tqdm

from aada.data import MoleculeDataLoader, MoleculeDataset, StandardScaler
from aada.modules import MoleculeModel


def predict(model,
                        data_loader: MoleculeDataLoader,
                        disable_progress_bar: bool = False,
                        scaler: StandardScaler = None) -> List[List[float]]:
        """
        Makes predictions on a dataset using an ensemble of models.

        :param model: A :class:`~aada.models.model.MoleculeModel`.
        :param data_loader: A :class:`~aada.data.data.MoleculeDataLoader`.
        :param disable_progress_bar: Whether to disable the progress bar.
        :param scaler: A :class:`~aada.features.scaler.StandardScaler` object fit on the training targets.
        :return: A list of lists of predictions. The outer list is molecules while the inner list is tasks.
        """
        model.eval()

        preds = []

        for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
                # Prepare batch
                batch: MoleculeDataset
                mol_batch, features_batch, atom_descriptors_batch = batch.batch_graph(), batch.features(), batch.atom_descriptors()

                # Make predictions
                with torch.no_grad():
                        batch_preds = model(batch, training=False)

                batch_preds = batch_preds.data.cpu().numpy()

                # Inverse scale if regression
                if scaler is not None:
                        batch_preds = scaler.inverse_transform(batch_preds)

                # Collect vectors
                batch_preds = batch_preds.tolist()
                preds.extend(batch_preds)

        return preds
