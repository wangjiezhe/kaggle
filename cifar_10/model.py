from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import Accuracy, ConfusionMatrix
from torchvision import models
from utils import NUM_CLASSES

__all__ = ["Classifier", "DefinedNet", "PretrainedNet", "ResNet", "RegNet", "ConvNeXt"]


class Classifier(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.train_acc = Accuracy("multiclass", num_classes=NUM_CLASSES, average="micro")
        self.val_acc = Accuracy("multiclass", num_classes=NUM_CLASSES, average="micro")
        self.test_top1_acc = Accuracy("multiclass", num_classes=NUM_CLASSES, average="micro")
        self.test_top2_acc = Accuracy("multiclass", num_classes=NUM_CLASSES, average="micro", top_k=2)
        self.test_top5_acc = Accuracy("multiclass", num_classes=NUM_CLASSES, average="micro", top_k=5)
        self.test_top9_acc = Accuracy("multiclass", num_classes=NUM_CLASSES, average="micro", top_k=9)
        self.confusion_matrix = ConfusionMatrix("multiclass", num_classes=NUM_CLASSES)

    def on_train_epoch_start(self):
        self.train_acc.reset()

    def on_validation_epoch_start(self):
        self.val_acc.reset()

    def on_test_epoch_start(self):
        self.test_top1_acc.reset()
        self.test_top2_acc.reset()
        self.test_top5_acc.reset()
        self.test_top9_acc.reset()
        self.confusion_matrix.reset()

    def training_step(self, batch):
        features, labels = batch
        preds = self(features)
        loss = F.cross_entropy(preds, labels)
        metric = {"train_loss": loss}
        if self.trainer.datamodule.mix is None:
            with torch.no_grad():
                metric["train_acc"] = self.train_acc(preds, labels)
        self.log_dict(metric, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        if self.trainer.datamodule.mix is None:
            self.log_dict({"train_epoch_acc": self.train_acc.compute()})

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
        top1_acc = self.test_top1_acc(preds, labels)
        top2_acc = self.test_top2_acc(preds, labels)
        top5_acc = self.test_top5_acc(preds, labels)
        top9_acc = self.test_top9_acc(preds, labels)
        self.confusion_matrix(preds, labels)
        self.log_dict(
            {
                "test_loss": loss,
                "test_top1_acc": top1_acc,
                "test_top2_acc": top2_acc,
                "test_top5_acc": top5_acc,
                "test_top9_acc": top9_acc,
            }
        )
        return loss

    def on_test_epoch_end(self):
        self.logger.log_hyperparams(
            self.hparams,
            {
                "hp/train_acc": self.train_acc.compute(),
                "hp/val_acc": self.val_acc.compute(),
                "hp/test_acc": self.test_top1_acc.compute(),
            },
        )

    def predict_step(self, batch):
        features = batch
        preds = self(features).argmax(axis=1)
        return preds.tolist()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())


class DefinedNet(Classifier):
    def __init__(self, model: str, pretrained=False, use_timm=False):
        super().__init__()
        self.save_hyperparameters()
        if use_timm:
            import timm

            available_models = timm.list_models()
            if model not in available_models:
                raise ValueError(f"{model} should be one of {available_models}.")
            self.backbone = timm.create_model(model, pretrained=pretrained)
        else:
            available_models = models.list_models()
            if model not in available_models:
                raise ValueError(f"{model} should be one of {available_models}.")
            weights = "DEFAULT" if pretrained else None
            self.backbone = models.get_model(model, weights=weights)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LazyLinear(NUM_CLASSES),
        )

    def forward(self, x):
        y = self.backbone(x)
        y = self.classifier(y)
        return y

    def configure_optimizers(self):
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()))


class PretrainedNet(Classifier):
    def __init__(self, model: str, pretrained=False):
        super().__init__()
        self.save_hyperparameters()

        available_models = models.list_models()
        if model not in available_models:
            raise ValueError(f"{model} should be one of {available_models}.")
        weights = "DEFAULT" if pretrained else None
        self.net = models.get_model(model, weights=weights)
        layers = [k for k, _ in self.net.named_children()]
        match layers[-1]:
            case "fc":
                # ResNet, RegNet, ResNeXt, Inception, GoogleNet, ShuffleNet
                assert isinstance(self.net.fc, nn.Linear)
                self.net.fc = nn.LazyLinear(NUM_CLASSES)
            case "classifier":
                if isinstance(self.net.classifier, nn.Linear):
                    # DenseNet
                    self.net.classifier = nn.LazyLinear(NUM_CLASSES)
                elif isinstance(self.net.classifier[-1], nn.Linear):
                    # VGG, ConvNeXt[2022], EfficientNet, AlexNet, MNASNet, MobileNet
                    self.net.classifier[-1] = nn.LazyLinear(NUM_CLASSES)
                else:
                    # SqueezeNet
                    assert layers == ["features", "classifier"]
                    assert len(self.net.classifier) == 4    # type: ignore
                    self.net.classifier[1] = nn.LazyConv2d(NUM_CLASSES, kernel_size=(1, 1), stride=(1, 1))
            case "head":
                # Swin
                assert isinstance(self.net.heads, nn.Linear)
                self.net.head = nn.LazyLinear(NUM_CLASSES)
            case "heads":
                # ViT
                assert isinstance(self.net.heads, nn.Linear)
                self.net.heads = nn.LazyLinear(NUM_CLASSES)
            case _:
                raise ValueError(f"{model} cannot be converted")

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()))


class ResNet(Classifier):
    def __init__(self, model: str, pretrained=False):
        super().__init__()
        self.save_hyperparameters()
        available_models = models.list_models(include="resnet*")
        if model not in available_models:
            raise ValueError(f"{model} should be one of {available_models}.")
        weights = "DEFAULT" if pretrained else None
        self.net = models.get_model(model, weights=weights)
        self.net.fc = nn.LazyLinear(NUM_CLASSES)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()))


class RegNet(Classifier):
    def __init__(
        self, model: str, pretrained=False, cosine_t_max: Optional[int] = None, verbose=False, **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cosine_t_max = cosine_t_max
        self.pretrained = pretrained
        self.verbose = verbose
        available_models = models.list_models(include="regnet*")
        if model not in available_models:
            raise ValueError(f"{model} should be one of {available_models}.")
        if self.pretrained:
            self.net = models.get_model(model, weights="DEFAULT", **kwargs)
            self.net.fc = nn.LazyLinear(NUM_CLASSES)
        else:
            self.net = models.get_model(model, num_classes=NUM_CLASSES, **kwargs)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()))
        if self.cosine_t_max:
            return {
                "optimizer": optimizer,
                "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, self.cosine_t_max, eta_min=0.00001, verbose=self.verbose
                ),
            }
        else:
            return optimizer


class ConvNeXt(Classifier):
    def __init__(self, model: str, pretrained=False):
        super().__init__()
        self.save_hyperparameters()
        available_models = models.list_models(include="convnext*")
        if model not in available_models:
            raise ValueError(f"{model} should be one of {available_models}.")
        weights = "DEFAULT" if pretrained else None
        self.net = models.get_model(model, weights=weights)
        self.net.classifier[-1] = nn.LazyLinear(NUM_CLASSES)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()))
