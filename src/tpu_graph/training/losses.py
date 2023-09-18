import torch


def square_loss(pred, label, label_log=False):
    """
    Calculates the square loss between pred and label
    :param pred: A list of predicted values
    :param label: A list of labels
    :param label_log: If True, values are taken as log values
    :return: The loss
    """

    loss = 0
    for p, l in zip(pred, label):
        if label_log:
            loss += torch.mean((p - torch.log(l)) ** 2)
        else:
            loss += torch.mean((p - l) ** 2)

    return loss
