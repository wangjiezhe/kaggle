import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import Metric


## log_mse_loss(x, y) == torch.sqrt(torchmetrics.functional.mean_squared_log_error(x-1, y-1))
##                    == sklearn.metrics.mean_squared_log_error(x-1, y-1, squared=False)
def log_mse_loss(preds, labels):
    clipped_preds = torch.clamp(preds, 1, float("inf"))
    mse = torch.sqrt(F.mse_loss(torch.log(clipped_preds), torch.log(labels)))
    return mse


class RMSLE(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("sum_squared_log_diff", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        clipped_preds = torch.clamp(y_pred, 1, float("inf"))
        squared_log_diff = torch.pow(torch.log(clipped_preds) - torch.log(y_true), 2)
        self.sum_squared_log_diff += torch.sum(squared_log_diff)
        self.num_samples += y_true.numel()

    def compute(self):
        rmsle = torch.sqrt(self.sum_squared_log_diff / self.num_samples)
        return rmsle


def init_cnn(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def predict(model, data):
    data.setup("predict")
    features = data.predict_data.tensors[0].to("cpu")
    preds = model.to("cpu")(features).detach().numpy()
    submission = pd.concat(
        [data.predict_data_id, pd.Series(preds.reshape(1, -1)[0])],
        axis=1,
        keys=("Id", "SalePrice"),
    )
    if os.path.exists("submission.csv"):
        os.rename("submission.csv", "submission_old.csv")
    submission.to_csv("submission.csv", index=False)
