from datetime import datetime
from typing import Optional

import pandas as pd
import torch
from pytorch_lightning.callbacks import BaseFinetuning, BasePredictionWriter, Callback, ModelCheckpoint
from utils import NUM_CLASSES

__all__ = [
    "Submission",
    "ConfMat",
    "ResNetFineTune",
    "RegNetFineTune",
    "ConvNeXtFineTune",
    "save_best_train_loss",
    "save_best_val_acc",
    "save_best_val_loss",
    "save_last",
]


class Submission(BasePredictionWriter):
    def __init__(self, now=True):
        super().__init__(write_interval="epoch")
        self.now = now

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        data = trainer.datamodule
        df = pd.read_csv(data.predict_csv)
        imgs = df["id"]
        submission = pd.concat(
            [
                imgs,
                pd.Series(
                    [data.label_map[i] for i in torch.stack(predictions).view(-1).tolist()],
                    name="label",
                ),
            ],
            axis=1,
        )
        if self.now:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"submission_{now}.csv"
        else:
            file_name = "submission.csv"
        submission.to_csv(file_name, index=False)


class ConfMat(Callback):
    def __init__(self):
        super().__init__()

    def on_test_epoch_end(self, trainer, pl_module):
        confmat = pl_module.confusion_matrix.compute()
        errors = confmat.clone()
        errors.fill_diagonal_(0)
        values, indices = errors.view(-1).topk(20)
        values, indices = values.tolist(), indices.tolist()
        real_indicies = [(i // NUM_CLASSES, i % NUM_CLASSES) for i in indices]

        label_map = trainer.datamodule.label_map

        df = pd.DataFrame(
            [
                [
                    label_map[row],
                    confmat[row, row].item(),
                    label_map[col],
                    value,
                ]
                for value, (row, col) in zip(values, real_indicies)
            ],
            columns=("True label", "Corrects", "Predict label", "Wrongs"),
        )
        print(df.to_markdown())


class ResNetFineTune(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10, freeze_again_at_epoch: Optional[int] = None, train_bn=True):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self._freeze_again_at_epoch = freeze_again_at_epoch
        self._train_bn = train_bn

    def freeze_before_training(self, pl_module):
        backbone = [pl_module.net.get_submodule(k) for k, _ in pl_module.net.named_children() if k != "fc"]
        self.freeze(backbone, train_bn=self._train_bn)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        backbone = [pl_module.net.get_submodule(k) for k, _ in pl_module.net.named_children() if k != "fc"]
        if current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=backbone,
                optimizer=optimizer,
                train_bn=(not self._train_bn),
            )
        elif self._unfreeze_at_epoch is not None and current_epoch == self._freeze_again_at_epoch:
            optimizer.param_group.pop()


class RegNetFineTune(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10, freeze_again_at_epoch: Optional[int] = None, train_bn=True):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self._freeze_again_at_epoch = freeze_again_at_epoch
        self._train_bn_when_freezing = train_bn

    def freeze_before_training(self, pl_module):
        self.freeze([pl_module.net.stem, pl_module.net.trunk_output], train_bn=self._train_bn_when_freezing)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        if current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=[pl_module.net.stem, pl_module.net.trunk_output],
                optimizer=optimizer,
                train_bn=(not self._train_bn_when_freezing),
            )
        elif self._unfreeze_at_epoch is not None and current_epoch == self._freeze_again_at_epoch:
            optimizer.param_group.pop()


class ConvNeXtFineTune(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10, freeze_again_at_epoch: Optional[int] = None):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self._freeze_again_at_epoch = freeze_again_at_epoch

    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.net.features)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        if current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=pl_module.net.features,
                optimizer=optimizer,
            )
        elif self._unfreeze_at_epoch is not None and current_epoch == self._freeze_again_at_epoch:
            optimizer.param_group.pop()


def save_best_val_acc(prefix, dirpath=None):
    if prefix:
        filename = f"{prefix}-" + "{epoch:02d}-{val_acc:.4f}"
    else:
        filename = "{epoch:02d}-{val_acc:.4f}"
    return ModelCheckpoint(
        dirpath=dirpath,
        monitor="val_acc",
        filename=filename,
        save_weights_only=True,
        mode="max",
        save_on_train_epoch_end=False,
        save_top_k=2,
    )


def save_best_val_loss(prefix, dirpath=None):
    if prefix:
        filename = f"{prefix}-" + "{epoch:02d}-{val_loss:.4f}"
    else:
        filename = "{epoch:02d}-{val_loss:.4f}"
    return ModelCheckpoint(
        dirpath=dirpath,
        filename=filename,
        monitor="val_loss",
        save_weights_only=True,
        save_on_train_epoch_end=False,
        save_top_k=2,
    )


def save_best_train_loss(prefix, dirpath=None):
    if prefix:
        filename = f"{prefix}-" + "{epoch:02d}-{train_loss:.4f}"
    else:
        filename = "{epoch:02d}-{train_loss:.4f}"
    return ModelCheckpoint(
        dirpath=dirpath,
        filename=filename,
        monitor="train_loss",
        save_weights_only=True,
        save_top_k=2,
    )


def save_last(prefix, dirpath=None):
    if prefix:
        filename = f"{prefix}-" + "{epoch:02d}-{step}"
    else:
        filename = "{epoch:02d}-{step}"
    return ModelCheckpoint(
        dirpath=dirpath,
        filename=filename,
        save_weights_only=True,
        save_last=True,
    )
