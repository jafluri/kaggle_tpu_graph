import os

import numpy as np
import torch
from scipy.stats import kendalltau
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..networks.networks import TPUGraphNetwork


def evaluate_network(network: TPUGraphNetwork, dataloader: DataLoader, device="cuda"):
    """
    Evaluates the network and returns the average loss, prediction and labels
    :param network: The network to evaluate
    :param dataloader: The dataloader to use
    :param device: The device to use
    :return: The graph embeddings, predictions and labels
    """

    # evaluate the network
    pbar = tqdm(dataloader)
    predictions = []
    labels = []
    embedding_vals = []
    with torch.no_grad():
        for batch_idx, (features, lengths, runtimes, edge_index) in enumerate(pbar):
            # to device
            features = features.to(device)
            runtimes = runtimes.to(device)
            edge_index = edge_index.to(device)

            # eval the network
            graph_embedding, pred_runtimes = network(features, edge_index, lengths)
            # graph embeddings should be list, batch, out with list dim 1
            embedding_vals.append(graph_embedding.cpu().detach().numpy()[0])
            # pred runtimes should be batch, 1
            predictions.append(pred_runtimes.cpu().detach().numpy().ravel())
            labels.append(runtimes.cpu().detach().numpy().ravel())

    # concat embeddings, predictions and labels and split according to files
    embedding_vals = np.concatenate(embedding_vals, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)

    # return the average loss, predictions and labels
    return embedding_vals, predictions, labels


def evaluate_layout_network(
    network: TPUGraphNetwork,
    dataloader: DataLoader,
    save_path: str | bytes | os.PathLike = None,
    device="cuda",
):
    """
    Evaluates the layout network on the given dataloader
    :param network: The network to evaluate
    :param dataloader: The dataloader to use
    :param save_path: The path where to save the predictions etc. (NPZ file)
    :param device: The device to use
    :return: The average loss (log mse) and the average Kendall's Tau
    """

    # get the dset from the dataloader
    dataset = dataloader.dataset

    # evaluate the network
    embedding_vals, predictions, labels = evaluate_network(network, dataloader, device=device)

    # split the predictions and labels according to the files
    split_predictions = np.split(predictions, dataset.offsets[:-1])
    split_labels = np.split(labels, dataset.offsets[:-1])

    # calculate the average Kendall's Tau
    kendalls = []
    rankings = []
    for p, l in zip(split_predictions, split_labels):
        # create the rankings
        p_argsort = np.argsort(p)
        l_argsort = np.argsort(l)
        p_rank = np.empty_like(p_argsort)
        p_rank[p_argsort] = np.arange(len(p_argsort))
        l_rank = np.empty_like(l_argsort)
        l_rank[l_argsort] = np.arange(len(l_argsort))
        rankings.append(p_rank)

        # evaluate the Kendall's Tau
        kendalls.append(kendalltau(l_rank, p_rank).correlation)

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
            embedding_vals=embedding_vals,
        )

    # return the average loss and Kendall's Tau
    return avg_kendall
