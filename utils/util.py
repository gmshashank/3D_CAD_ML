import numpy as np
import torch
import torch.nn.functional as F


def cal_loss(pred, target, smoothing=True):
    "Calculate the Cross Entropy Loss and apply label smoothing if needed"
    target = target.contiguous().view(-1)
    if smoothing:
        eps = 0.2
        num_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (num_class - 1)
        log_probs = F.log_softmax(pred, dim=1)
        loss = -1 * (one_hot * log_probs).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, target, reduction="mean")

    return loss


def accuracy_score(y_true, y_prob):
    # https://discuss.pytorch.org/t/calculate-accuracy-in-binary-classification/91758
    assert y_true.ndim == 1 and y_true.size() == y_prob.size()
    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum().item() / y_true.size(0)
