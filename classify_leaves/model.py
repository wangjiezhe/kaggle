from datetime import datetime

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import Accuracy
from torchvision import models


class Classifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.output_size = 176
        self.train_acc = Accuracy("multiclass", num_classes=self.output_size, average="micro")
        self.val_acc = Accuracy("multiclass", num_classes=self.output_size, average="micro")
        self.test_acc = Accuracy("multiclass", num_classes=self.output_size, average="micro")
        self.test_top5_acc = Accuracy("multiclass", num_classes=self.output_size, average="micro", top_k=5)
        self.predict_labels = []

    def on_train_epoch_start(self):
        self.train_acc.reset()

    def on_validation_epoch_start(self):
        self.val_acc.reset()

    def on_test_epoch_start(self):
        self.test_acc.reset()
        self.test_top5_acc.reset()

    def training_step(self, batch):
        features, labels = batch
        preds = self(features)
        loss = F.cross_entropy(preds, labels)
        with torch.no_grad():
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
        top5_acc = self.test_top5_acc(preds, labels)
        self.log_dict({"test_loss": loss, "test_acc": acc, "test_top5_acc": top5_acc}, prog_bar=True)
        return loss

    def on_test_end(self):
        self.logger.log_hyperparams(
            self.hparams,
            {
                "hp/train_acc": self.train_acc.compute(),
                "hp/val_acc": self.val_acc.compute(),
                "hp/test_acc": self.test_acc.compute(),
                "hp/test_top5_acc": self.test_top5_acc.compute(),
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
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission.to_csv(f"submission_{now}.csv", index=False)
        self.predict_labels.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())


class DefinedNet(Classifier):
    def __init__(self, resnet, weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=["resnet"])
        self.pretrained = True if weights else False

        self.net = resnet(weights=weights)
        layers = [k for k, _ in self.net.named_children()]
        match layers[-1]:
            case "fc":
                # ResNet, RegNet, ResNeXt, Inception, GoogleNet, ShuffleNet
                assert isinstance(self.net.fc, nn.Linear)
                self.net.fc = nn.LazyLinear(self.output_size)
            case "classifier":
                if isinstance(self.net.classifier, nn.Linear):
                    # DenseNet
                    self.net.classifier = nn.LazyLinear(self.output_size)
                elif isinstance(self.net.classifier[-1], nn.Linear):
                    # VGG, ConvNeXt[2022], EfficientNet, AlexNet, MNASNet, MobileNet
                    self.net.classifier[-1] = nn.LazyLinear(self.output_size)
                else:
                    # SqueezeNet
                    assert layers == ["features", "classifier"]
                    assert len(self.net.classifier) == 4
                    self.net.classifier[1] = nn.LazyConv2d(
                        self.output_size, kernel_size=(1, 1), stride=(1, 1)
                    )
            case "head":
                # Swin
                assert isinstance(self.net.heads, nn.Linear)
                self.net.head = nn.LazyLinear(self.output_size)
            case "heads":
                # ViT
                assert isinstance(self.net.heads, nn.Linear)
                self.net.heads = nn.LazyLinear(self.output_size)
            case _:
                raise ValueError(f"{resnet.__name__} cannot be converted")

    def forward(self, X):
        return self.net(X)

    ## FIXME
    def configure_optimizers(self):
        if self.pretrained:
            params_1x = [
                param
                for name, param in self.named_parameters()
                if name not in ["net.fc.weight", "net.fc.bias"]
            ]
            return torch.optim.AdamW(
                [{"params": params_1x, "lr": 0.0001}, {"params": self.net.fc.parameters(), "lr": 0.001}]
            )
        else:
            return torch.optim.AdamW(self.parameters())


class ResNet(Classifier):
    def __init__(self, resnet, weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=["resnet"])
        self.pretrained = True if weights else False

        self.net = resnet(weights=weights)
        assert isinstance(self.net.fc, nn.Linear)
        self.net.fc = nn.LazyLinear(self.output_size)

    def forward(self, X):
        return self.net(X)

    def configure_optimizers(self):
        if self.pretrained:
            params_1x = [
                param
                for name, param in self.named_parameters()
                if name not in ["net.fc.weight", "net.fc.bias"]
            ]
            return torch.optim.AdamW(
                [{"params": params_1x, "lr": 0.0001}, {"params": self.net.fc.parameters(), "lr": 0.001}]
            )
        else:
            return torch.optim.AdamW(self.parameters())


class ResNet_pretrained(ResNet):
    def __init__(self):
        super().__init__(weights="DEFAULT")


RegNet = ResNet
ResNeXt = ResNet
Inception = ResNet
GoogleNet = ResNet
ShuffleNet = ResNet
RegNet_pretrained = ResNet_pretrained
ResNeXt_pretrained = ResNet_pretrained
Inception_pretrained = ResNet_pretrained
GoogleNet_pretrained = ResNet_pretrained
ShuffleNet_pretrained = ResNet_pretrained


class DenseNet(Classifier):
    def __init__(self, resnet, weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=["resnet"])
        self.pretrained = True if weights else False

        self.net = resnet(weights=weights)
        assert isinstance(self.net.classifier, nn.Linear)
        self.net.classifier = nn.LazyLinear(self.output_size)

    def forward(self, X):
        return self.net(X)

    def configure_optimizers(self):
        if self.pretrained:
            params_1x = [
                param
                for name, param in self.named_parameters()
                if name not in ["net.classifier.weight", "net.classifier.bias"]
            ]
            return torch.optim.AdamW(
                [
                    {"params": params_1x, "lr": 0.0001},
                    {"params": self.net.classifier.parameters(), "lr": 0.001},
                ]
            )
        else:
            return torch.optim.AdamW(self.parameters())


class AlexNet(Classifier):
    def __init__(self, resnet, weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=["resnet"])
        self.pretrained = True if weights else False

        self.net = resnet(weights=weights)
        assert isinstance(self.net.classifier[-1], nn.Linear)
        # VGG, ConvNeXt[2022], EfficientNet, AlexNet, MNASNet, MobileNet
        self.net.classifier[-1] = nn.LazyLinear(self.output_size)

    def forward(self, X):
        return self.net(X)

    def configure_optimizers(self):
        if self.pretrained:
            params_1x = [
                param
                for name, param in self.named_parameters()
                if name not in ["net.classifier." + k for k, _ in self.net.classifier.named_parameters()]
            ]
            return torch.optim.AdamW(
                [
                    {"params": params_1x, "lr": 0.0001},
                    {"params": self.net.classifier.parameters(), "lr": 0.001},
                ]
            )
        else:
            return torch.optim.AdamW(self.parameters())


VGG = AlexNet
EfficientNet = AlexNet
MNASNet = AlexNet
MobileNet = AlexNet
ConvNeXt = AlexNet


class ViT(Classifier):
    def __init__(self, resnet, weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=["resnet"])
        self.pretrained = True if weights else False

        self.net = resnet(weights=weights)
        assert isinstance(self.net.heads, nn.Linear)
        self.net.heads = nn.LazyLinear(self.output_size)

    def forward(self, X):
        return self.net(X)

    def configure_optimizers(self):
        if self.pretrained:
            params_1x = [
                param
                for name, param in self.named_parameters()
                if name not in ["net.heads.weight", "net.heads.bias"]
            ]
            return torch.optim.AdamW(
                [{"params": params_1x, "lr": 0.0001}, {"params": self.net.heads.parameters(), "lr": 0.001}]
            )
        else:
            return torch.optim.AdamW(self.parameters())
