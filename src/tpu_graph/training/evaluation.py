import os

import numpy as np
import torch
from scipy.stats import kendalltau
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import losses
from ..networks.networks import TPUGraphNetwork


def evaluate_network(network: TPUGraphNetwork, dataloader: DataLoader):
    """
    Evaluates the network and returns the average loss, prediction and labels
    :param network: The network to evaluate
    :param dataloader: The dataloader to use
    :return: The average loss, predictions and labels
    """

    # evaluate the network
    pbar = tqdm(dataloader, postfix={"loss": 0})
    predictions = []
    labels = []
    loss_vals = []
    with torch.no_grad():
        for batch_idx, (features, lengths, runtimes, edge_index) in enumerate(pbar):
            # eval the network
            pred_runtimes = network(features, edge_index, lengths)
            predictions.append(pred_runtimes.cpu().detach().numpy())
            labels.append(torch.squeeze(runtimes).cpu().detach().numpy())

            # calculate the loss and log it
            loss = losses.square_loss(pred=pred_runtimes, label=runtimes, log=True)
            loss_vals.append(loss.item())
            pbar.set_postfix({"loss": loss.item()})

    # the average loss
    avg_loss = np.mean(loss_vals)

    # concat predictions and labels and split according to files
    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)

    # return the average loss, predictions and labels
    return avg_loss, predictions, labels


def evaluate_tile_network(
    network: TPUGraphNetwork,
    dataloader: DataLoader,
    save_path: str | bytes | os.PathLike = None,
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
    avg_loss, predictions, labels = evaluate_network(network, dataloader)

    # split the predictions and labels according to the files
    split_predictions = np.split(predictions, dataset.offsets[:-1])
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
            top_ks=np.array(top_ks, dtype=object),
        )

    # return the average loss and slowdown
    return avg_loss, avg_slowdown


def evaluate_layout_network(
    network: TPUGraphNetwork,
    dataloader: DataLoader,
    save_path: str | bytes | os.PathLike = None,
):
    """
    Evaluates the layout network on the given dataloader
    :param network: The network to evaluate
    :param dataloader: The dataloader to use
    :param save_path: The path where to save the predictions etc. (NPZ file)
    :return: The average loss (log mse) and the average Kendall's Tau
    """

    # get the dset from the dataloader
    dataset = dataloader.dataset

    # evaluate the network
    avg_loss, predictions, labels = evaluate_network(network, dataloader)

    # split the predictions and labels according to the files
    split_predictions = np.split(predictions, dataset.offsets[:-1])
    split_labels = np.split(labels, dataset.offsets[:-1])

    # calculate the average Kendall's Tau
    kendalls = []
    rankings = []
    for p, l in zip(split_predictions, split_labels):
        p_argsort = np.argsort(p)
        l_argsort = np.argsort(l)
        rankings.append(p_argsort)
        kendalls.append(kendalltau(p_argsort, l_argsort).correlation)

    # get the average Kendall's Tau
    avg_kendall = np.mean(kendalls)

    # save everything if save_path is not None
    if save_path is not None:
        # save the predictions and labels
        np.savez(
            save_path,
            predictions=predictions,
            labels=labels,
            offsets=dataset.offsets,
            file_list=np.array([str(f) for f in dataset.file_list]),
            kendalls=np.array(kendalls),
            rankings=np.array(rankings, dtype=object),
        )

    # return the average loss and Kendall's Tau
    return avg_loss, avg_kendall
