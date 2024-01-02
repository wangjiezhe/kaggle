import os

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import Accuracy
from torchvision import models


class Classifier(pl.LightningModule):
    def __init__(self, output_size=176):
        super().__init__()
        self.input_size = [224, 224]
        self.train_acc = Accuracy("multiclass", num_classes=output_size, average="micro")
        self.val_acc = Accuracy("multiclass", num_classes=output_size, average="micro")
        self.test_acc = Accuracy("multiclass", num_classes=output_size, average="micro")
        self.predict_labels = []

    def on_train_epoch_start(self):
        self.train_acc.reset()

    def on_validation_epoch_start(self):
        self.val_acc.reset()

    def on_test_epoch_start(self):
        self.test_acc.reset()

    def training_step(self, batch):
        features, labels = batch
        preds = self(features)
        loss = F.cross_entropy(preds, labels)
        acc = self.train_acc(preds, labels)
        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True)
        return loss

    def validation_step(self, batch):
        features, labels = batch
        preds = self(features)
        loss = F.cross_entropy(preds, labels)
        acc = self.val_acc(preds, labels)
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)
        return loss

    def test_step(self, batch):
        features, labels = batch
        preds = self(features)
        loss = F.cross_entropy(preds, labels)
        acc = self.test_acc(preds, labels)
        self.log_dict({"test_loss": loss, "test_acc": acc}, prog_bar=True)
        return loss

    def on_test_end(self):
        self.logger.log_hyperparams(
            self.hparams,
            {
                "hp/train_acc": self.train_acc.compute(),
                "hp/val_acc": self.val_acc.compute(),
                "hp/test_acc": self.test_acc.compute(),
            },
        )

    def predict_step(self, batch):
        features = batch[0]
        labels = self(features).argmax(axis=1)
        self.predict_labels.extend(labels.tolist())

    def on_predict_end(self):
        submission = pd.concat(
            [
                self.trainer.datamodule.predict_images,
                pd.Series([self.trainer.datamodule.label_map[i] for i in self.predict_labels], name="label"),
            ],
            axis=1,
        )
        if os.path.exists("submission.csv"):
            os.rename("submission.csv", "submission_old.csv")
        submission.to_csv("submission.csv", index=False)
        self.predict_labels.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())


class ResNet(Classifier):
    def __init__(self, resnet, weight=None, output_size=176):
        super().__init__(output_size)
        self.save_hyperparameters()
        self.net = resnet(weight)
        self.net.fc = nn.LazyLinear(output_size)

    def forward(self, X):
        return self.net(X)


class ResNet_pretrained(Classifier):
    def __init__(self, resnet, weight=None, output_size=176):
        super().__init__(output_size)
        self.save_hyperparameters()
        self.net = resnet(weight)
        self.net.fc = nn.LazyLinear(output_size)

    def forward(self, X):
        return self.net(X)

    def configure_optimizers(self):
        params_1x = [
            param for name, param in self.named_parameters() if name not in ["net.fc.weight", "net.fc.bias"]
        ]
        return torch.optim.AdamW(
            [{"params": params_1x, "lr": 0.0001}, {"params": self.net.fc.parameters(), "lr": 0.001}]
        )


class ResNet18(ResNet):
    def __init__(self):
        super().__init__(models.resnet18)


class ResNet18_pretrained(ResNet_pretrained):
    def __init__(self):
        super().__init__(models.resnet18, weight=models.ResNet18_Weights.DEFAULT)


class ResNet34(ResNet):
    def __init__(self):
        super().__init__(models.resnet34)


class ResNet34_pretrained(ResNet_pretrained):
    def __init__(self):
        super().__init__(models.resnet34, weight=models.ResNet34_Weights.DEFAULT)


class ResNet50(ResNet):
    def __init__(self):
        super().__init__(models.resnet50)


class ResNet50_pretrained(ResNet_pretrained):
    def __init__(self):
        super().__init__(models.resnet50, weight=models.ResNet50_Weights.DEFAULT)
