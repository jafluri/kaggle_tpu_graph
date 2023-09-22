import numpy as np
import torch


def square_loss(pred, label, log=False):
    """
    Calculates the square loss between pred and label
    :param pred: A list of predicted values
    :param label: A list of labels
    :param log: If True, values are taken as log values
    :return: The loss
    """

    loss = 0
    for p, l in zip(pred, label):
        if log:
            loss += torch.mean((torch.log(p + 1) - torch.log(l + 1)) ** 2)
        else:
            loss += torch.mean((p - l) ** 2)

    return loss / len(pred)


def slowdown(pred: list[np.ndarray], label: list[np.ndarray], k: int = 5):
    """
    Calculates the average slowdown for the predictions and labels
    :param pred: A list of predicted runtimes, each entry corresponds to the runtimes of a graph in different
                 configurations
    :param label: The same as pred but for the labels
    :param k: The top k predictions to use for the slowdown calculation (See kaggle for more info)
    :return: The slowdown for each graph and the top k predictions (integers)
    """

    top_ks = []
    slowdowns = []
    for p, l in zip(pred, label):
        # we are need the top k predictions
        top_k = np.argsort(p)[:k]
        top_ks.append(top_k)

        # now we calculate the slowdown
        slowdown = 2 - np.min(l[top_k]) / np.min(l)
        slowdowns.append(slowdown)

    # to numpy (we don't transform the top k since they might not always have full length
    slowdowns = np.array(slowdowns)

    return slowdowns, top_ks
