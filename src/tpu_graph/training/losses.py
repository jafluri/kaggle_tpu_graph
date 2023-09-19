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
