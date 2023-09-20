import os

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import losses
from ..networks.networks import TPUGraphNetwork


def evaluate_tile_network(
    network: TPUGraphNetwork, dataloader: DataLoader, save_path: str | bytes | os.PathLike = None
):
    """
    Evaluates the tile network on the given dataloader
    :param network: The network to evaluate
    :param dataloader: The dataloader to use
    :param save_path: If not None, the path where to save the predictions etc. (NPZ file)
    :return: The average loss (log mse) and the average slowdown
    """

    # get the dset from the dataloader
    dataset = dataloader.dataset

    # evaluate the network
    pbar = tqdm(dataloader, postfix={"loss": 0})
    predictions = []
    labels = []
    loss_vals = []
    for batch_idx, (features, runtimes, edges, graphs) in enumerate(pbar):
        # eval the network
        pred_runtimes = network.accumulate_runtime(features, edges, graphs)
        predictions.append(np.array([p.cpu().detach().numpy() for p in pred_runtimes]))
        labels.append(np.array([r.cpu().detach().numpy() for r in runtimes]))

        # calculate the loss and log it
        loss = losses.square_loss(pred=pred_runtimes, label=runtimes, log=True)
        loss_vals.append(loss.item())
        pbar.set_postfix({"loss": loss.item()})

    # the average loss
    avg_loss = np.mean(loss_vals)

    # concat predictions and labels and split according to files
    predictions = np.concatenate(predictions, axis=0)
    split_predictions = np.split(predictions, dataset.offsets[:-1])
    labels = np.concatenate(labels, axis=0)
    split_labels = np.split(labels, dataset.offsets[:-1])

    # calculate the average slowdown
    slowdowns, top_ks = losses.slowdown(split_predictions, split_labels)
    avg_slowdown = np.mean(slowdowns)

    # save everything if save_path is not None
    if save_path is not None:
        # save the predictions and labels
        np.savez(
            save_path,
            predictions=predictions,
            labels=labels,
            offsets=dataset.offsets,
            file_list=np.array([str(f) for f in dataset.file_list]),
            slowdowns=slowdowns,
            top_ks=top_ks,
        )

    # return the average loss and slowdown
    return avg_loss, avg_slowdown
