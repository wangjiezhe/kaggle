#!/usr/bin/env python3

import os.path

import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn, optim
from utils import init_cnn, log_mse_loss


class Classifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.train_loss = torch.tensor(float("inf"))
        self.train_loss_min = torch.tensor(float("inf"))
        self.val_loss_min = torch.tensor(float("inf"))
        self.test_loss = torch.tensor(float("inf"))
        self.input_size = 330
        self.output_size = 1

    def training_step(self, batch):
        features, labels = batch
        preds = self(features)
        self.train_loss = log_mse_loss(preds, labels)
        if self.train_loss < self.train_loss_min:
            self.train_loss_min = self.train_loss
        self.log("train_loss", self.train_loss, prog_bar=True)
        return self.train_loss

    def validation_step(self, batch):
        features, labels = batch
        preds = self(features)
        self.val_loss = log_mse_loss(preds, labels)
        if self.val_loss < self.val_loss_min:
            self.val_loss_min = self.train_loss
        self.log("val_loss", self.val_loss, prog_bar=True)
        return self.val_loss

    def test_step(self, batch):
        features, labels = batch
        preds = self(features)
        self.test_loss = log_mse_loss(preds, labels)
        self.log("test_loss", self.test_loss, prog_bar=True)
        return self.test_loss

    def on_test_end(self):
        self.logger.log_hyperparams(
            self.hparams,
            {
                "hp/train_loss": self.train_loss,
                "hp/train_loss_min": self.train_loss_min,
                "hp/val_loss": self.val_loss,
                "hp/val_loss_min": self.val_loss_min,
                "hp/test_loss": self.test_loss,
            },
        )

    def predict_step(self, batch):
        return None

    def on_predict_end(self):
        features = self.trainer.datamodule.predict_data.tensors[0].to("cuda")
        preds = self.net(features).detach().to("cpu").numpy()
        submission = pd.concat(
            [self.trainer.datamodule.predict_data_id, pd.Series(preds.reshape(1, -1)[0])],
            axis=1,
            keys=("Id", "SalePrice"),
        )
        if os.path.exists("submission.csv"):
            os.rename("submission.csv", "submission_old.csv")
        submission.to_csv("submission.csv", index=False)


class MLP(Classifier):
    def __init__(self, lr=1, momentum=0, weight_decay=1e-4, hidden_size=[1024, 64], optim="SGD"):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        input_size = self.input_size
        for i, s in enumerate(hidden_size):
            output_size = s
            self.net.add_module(
                f"fc{i+1}",
                nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU(), nn.Dropout()),
            )
            input_size = s
        self.net.add_module("last", nn.Linear(input_size, self.output_size))
        self.inited = False

    def setup(self, stage=None):
        if not self.inited:
            self.net.apply(init_cnn)
            self.inited = True

    def forward(self, X):
        return self.net(X)

    def configure_optimizers(self):
        match self.hparams["optim"]:
            case "Adadelta":
                return optim.Adadelta(self.parameters())
            case "Adam":
                return optim.Adam(self.parameters())
            case "AdamW":
                return optim.AdamW(self.parameters())
            case _:
                return optim.SGD(
                    self.parameters(),
                    lr=self.hparams.lr,
                    momentum=self.hparams.momentum,
                    weight_decay=self.hparams.weight_decay,
                )


class DenseMLPBlock(nn.Module):
    def __init__(self, input_size, feature_size, layers, dropout=0.5):
        super().__init__()
        blk = []
        for i in range(layers):
            blk.append(
                nn.Sequential(
                    nn.BatchNorm1d(input_size),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                    nn.Linear(input_size, feature_size),
                )
            )
            input_size = feature_size
        self.net = nn.Sequential(*blk)

    def forward(self, x):
        y = self.net(x)
        return torch.cat((x, y), dim=1)


class DenseMLP(Classifier):
    def __init__(
        self,
        arch=((64, 2), (128, 2), (256, 2), (512, 2)),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.inited = False
        blk = []
        input_size = self.input_size
        for feature_size, layers in arch:
            blk.append(DenseMLPBlock(input_size, feature_size, layers))
            input_size += feature_size
            blk.append(
                nn.Sequential(
                    nn.BatchNorm1d(input_size),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(input_size, input_size // 2),
                )
            )
            input_size //= 2
        self.net = nn.Sequential(*blk, nn.Linear(input_size, self.output_size))

    def forward(self, x):
        return self.net(x)

    def setup(self, stage=None):
        if not self.inited:
            self.apply_init()

    def apply_init(self, dataloader=None):
        if dataloader is None:
            dataloader = self.trainer.datamodule.train_dataloader()
        dummy_batch = next(iter(dataloader))[0][0:2]
        self.forward(dummy_batch)
        self.net.apply(init_cnn)
        self.inited = True

    def configure_optimizers(self):
        self.hparams["optim"] = "SGD"
        return torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
