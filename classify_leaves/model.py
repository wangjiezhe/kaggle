from datetime import datetime

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import Accuracy, ConfusionMatrix
from torchvision import models
from utils import cutmix_or_mixup, NUM_CLASSES


class Classifier(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # self.train_acc = Accuracy("multiclass", num_classes=NUM_CLASSES, average="micro")
        self.val_acc = Accuracy("multiclass", num_classes=NUM_CLASSES, average="micro")
        self.test_top1_acc = Accuracy("multiclass", num_classes=NUM_CLASSES, average="micro")
        self.test_top2_acc = Accuracy("multiclass", num_classes=NUM_CLASSES, average="micro", top_k=2)
        self.test_top5_acc = Accuracy("multiclass", num_classes=NUM_CLASSES, average="micro", top_k=5)
        self.test_top9_acc = Accuracy("multiclass", num_classes=NUM_CLASSES, average="micro", top_k=9)
        self.predict_labels = []
        self.confusion_matrix = ConfusionMatrix("multiclass", num_classes=NUM_CLASSES)
        self.translation_map = None

    # def on_train_epoch_start(self):
    #     self.train_acc.reset()

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
        # features, labels = cutmix_or_mixup(*batch)
        preds = self(features)
        loss = F.cross_entropy(preds, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     features, labels = batch
    #     with torch.no_grad():
    #         preds = self(features)
    #     acc = self.train_acc(preds, labels)
    #     self.log_dict({"train_loss": outputs["loss"], "train_acc": acc}, prog_bar=True)

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

    def on_test_end(self):
        self.logger.log_hyperparams(
            self.hparams,
            {
                # "hp/train_acc": self.train_acc.compute(),
                "hp/val_acc": self.val_acc.compute(),
                "hp/test_acc": self.test_top1_acc.compute(),
            },
        )

        confmat = self.confusion_matrix.compute()
        errors = confmat.clone()
        errors.fill_diagonal_(0)
        values, indices = errors.view(-1).topk(20)
        values, indices = values.tolist(), indices.tolist()
        real_indicies = [(i // NUM_CLASSES, i % NUM_CLASSES) for i in indices]

        label_map = self.trainer.datamodule.label_map
        if not self.translation_map:
            df_trans = pd.read_csv("plant_name.csv", encoding="utf-8")
            self.translation_map = {k: v for k, v in zip(df_trans.iloc[:, 0], df_trans.iloc[:, 1])}

        df = pd.DataFrame(
            [
                [
                    t := label_map[row],
                    self.translation_map[t],
                    confmat[row, row].item(),
                    f := label_map[col],
                    self.translation_map[f],
                    value,
                ]
                for value, (row, col) in zip(values, real_indicies)
            ],
            columns=("True label", "准确值", "Corrects", "Predict label", "预测值", "Wrongs"),
        )
        print(df.to_markdown())

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
                    assert len(self.net.classifier) == 4
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
        self.net.fc = nn.LazyLinear(NUM_CLASSES)

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
    def __init__(self, resnet):
        super().__init__(resnet, weights="DEFAULT")


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
        self.net.classifier = nn.LazyLinear(NUM_CLASSES)

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
        self.net.classifier[-1] = nn.LazyLinear(NUM_CLASSES)

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
        self.net.heads = nn.LazyLinear(NUM_CLASSES)

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
