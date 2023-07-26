import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x, y, labels):
        dists = torch.pairwise_distance(x, y)

        return torch.mean((1 - labels) * torch.pow(dists, 2) +
                          torch.pow(torch.clamp(self.margin - dists, min=0.), 2) * labels)


class SimCSELoss(nn.Module):
    def __init__(self, temp=0.5, sup=False):
        super(SimCSELoss, self).__init__()
        self.temp = temp
        self.sup = sup

    def forward(self, x):
        sims = torch.cosine_similarity(x.unsqueeze(0), x.unsqueeze(1), dim=-1) / self.temp
        if not self.sup:
            labels = torch.arange(x.size(0))
            labels = (labels - labels % 2 * 2) + 1
            sims -= torch.eye(x.size(0)) * 1e12
            return torch.mean(F.cross_entropy(sims, labels))
        rows = torch.arange(0, sims.size(0), 3)
        cols = torch.arange(0, sims.size(0))
        cols = cols[cols % 3 != 0]
        sims = sims[rows, :]
        sims = sims[:, cols]
        return torch.mean(F.cross_entropy(sims, torch.arange(0, cols.shape[0], 2)))


class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, dist_type='cos'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.dist_type = dist_type

    def forward(self, a, p, n):
        if self.dist_type == 'cos':
            return torch.mean(torch.relu(torch.cosine_similarity(a, n)
                                         - torch.cosine_similarity(a, p) + self.margin))
        return torch.mean(torch.relu(torch.pairwise_distance(a, p) -
                                     torch.pairwise_distance(a, n) + self.margin))


class RMSELoss(nn.Module):
    def __init__(self, reduction='none', eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(self.mse(y_pred.float(), y_true.float()) + self.eps)
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss


def MCRMSE(y_true, y_pred):
    losses = []
    for i in range(y_true.shape[1]):
        pred = y_pred[:, i]
        target = y_true[:, i]
        losses.append(mean_squared_error(pred, target, squared=False))

    return np.mean(losses), losses


def get_score(y_true, y_pred):
    return MCRMSE(y_true, y_pred)