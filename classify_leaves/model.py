import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import Accuracy, ConfusionMatrix
from torchvision import models
from utils import NUM_CLASSES


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
        if self.trainer.datamodule.mix is None:  # type: ignore
            with torch.no_grad():
                metric["train_acc"] = self.train_acc(preds, labels)
        self.log_dict(metric, prog_bar=True)
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
        self.logger.log_hyperparams(  # type: ignore
            self.hparams,  # type: ignore
            {
                "hp/train_acc": self.train_acc.compute(),
                "hp/val_acc": self.val_acc.compute(),
                "hp/test_acc": self.test_top1_acc.compute(),
            },
        )

    def predict_step(self, batch):
        features = batch[0]
        preds = self(features).argmax(axis=1)
        return preds

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())


class DefinedNet(Classifier):
    def __init__(self, model: str, weights=None):
        super().__init__()
        self.save_hyperparameters()

        available_models = models.list_models()
        if model not in available_models:
            raise ValueError(f"{model} should be one of {available_models}.")
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
    def __init__(self, model: str, weights=None):
        available_models = [m for m in models.list_models() if m.startswith("resnet")]
        if model not in available_models:
            raise ValueError(f"{model} should be one of {available_models}.")
        super().__init__()
        self.save_hyperparameters()
        self.net = models.get_model(model, weights=weights)
        self.net.fc = nn.LazyLinear(NUM_CLASSES)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()))


class RegNet(Classifier):
    def __init__(self, model: str, weights=None):
        available_models = [m for m in models.list_models() if m.startswith("regnet")]
        if model not in available_models:
            raise ValueError(f"{model} should be one of {available_models}.")
        super().__init__()
        self.save_hyperparameters()
        self.net = models.get_model(model, weights=weights)
        self.net.fc = nn.LazyLinear(NUM_CLASSES)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()))


class ConvNeXt(Classifier):
    def __init__(self, model: str, weights=None):
        available_models = [m for m in models.list_models() if m.startswith("convnext")]
        if model not in available_models:
            raise ValueError(f"{model} should be one of {available_models}.")
        super().__init__()
        self.save_hyperparameters()
        self.net = models.get_model(model, weights=weights)
        self.net.classifier[-1] = nn.LazyLinear(NUM_CLASSES)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()))
