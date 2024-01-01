#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from torch import nn


## log_mse_loss(x, y) == torch.sqrt(torchmetrics.functional.mean_squared_log_error(x-1, y-1))
##                    == sklearn.metrics.mean_squared_log_error(x-1, y-1, squared=False)
def log_mse_loss(preds, labels):
    clipped_preds = torch.clamp(preds, 1, float("inf"))
    mse = torch.sqrt(F.mse_loss(torch.log(clipped_preds), torch.log(labels)))
    return mse


def init_cnn(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
