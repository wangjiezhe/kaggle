import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from pytorch_lightning.callbacks import BaseFinetuning, BasePredictionWriter, BatchSizeFinder, Callback, ModelCheckpoint
from torchinfo import summary
from utils import NUM_CLASSES

__all__ = [
    "Submission",
    "ConfMatTopK",
    "ConfMatPlot",
    "ResNetFineTune",
    "RegNetFineTune",
    "ConvNeXtFineTune",
    "save_best_train_loss",
    "save_best_val_acc",
    "save_best_val_loss",
    "save_last",
    "LayerSummary",
]

log = logging.getLogger(__name__)


class Submission(BasePredictionWriter):
    def __init__(self):
        super().__init__(write_interval="epoch")

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        data = trainer.datamodule  # type: ignore
        now = datetime.now().strftime("%Y%m%d_%H%M%S")

        df = pd.read_csv(data.predict_csv)
        imgs = df["id"]
        columns = df.columns.tolist()
        columns.pop(0)

        if data.five_crop:
            predictions_centercrop, predictions_fivecrop = predictions
        else:
            predictions_centercrop = predictions

        # 使用 `softmax` 函数进行归一化
        preds = [row.tolist() for item in predictions_centercrop for row in item]
        submission = pd.concat([imgs, pd.DataFrame(preds, columns=columns)], axis=1)
        submission.to_csv(f"submission_{now}.csv", index=False)

        if data.five_crop:
            preds5 = [
                row.tolist()
                for item in predictions_fivecrop  # type: ignore
                # 对五张图片的预测结果进行合并（直接取平均？）
                for row in item.view(-1, 5, item.shape[-1]).mean(dim=1).softmax(dim=1)
            ]
            submission5 = pd.concat([imgs, pd.DataFrame(preds5, columns=columns)], axis=1)
            submission5.to_csv(f"submission5_{now}.csv", index=False)


# 显示预测错误最多的 k 个结果
class ConfMatTopK(Callback):
    def __init__(self, top_k):
        self.top_k = top_k

    def on_test_epoch_end(self, trainer, pl_module):
        confmat = pl_module.confusion_matrix.compute()
        errors = confmat.clone()
        errors.fill_diagonal_(0)
        values, indices = errors.view(-1).topk(self.top_k)
        values, indices = values.tolist(), indices.tolist()
        real_indicies = [(i // NUM_CLASSES, i % NUM_CLASSES) for i in indices]

        label_map = trainer.datamodule.label_map  # type: ignore

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
            columns=["True label", "Corrects", "Predict label", "Wrongs"],
        )
        print(df.to_markdown())


# 直接打印混淆矩阵
class ConfMatPlot(Callback):
    def on_test_epoch_end(self, trainer, pl_module):
        pl_module.confusion_matrix.plot()


class ResNetFineTune(BaseFinetuning):
    def __init__(
        self,
        unfreeze_at_epoch=10,
        freeze_again_at_epoch: Optional[int] = None,
        initial_denom_lr=10,
        train_bn=True,
        verbose=False,
    ):
        super().__init__()
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.freeze_again_at_epoch = freeze_again_at_epoch
        self.initial_denom_lr = initial_denom_lr
        self.train_bn_when_freezing = train_bn
        self.verbose = verbose

    def freeze_before_training(self, pl_module):
        backbone = [pl_module.net.get_submodule(k) for k, _ in pl_module.net.named_children() if k != "fc"]
        self.freeze(backbone, train_bn=self.train_bn_when_freezing)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        backbone = [pl_module.net.get_submodule(k) for k, _ in pl_module.net.named_children() if k != "fc"]
        if current_epoch == self.unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=backbone,
                optimizer=optimizer,
                initial_denom_lr=self.initial_denom_lr,
                train_bn=(not self.train_bn_when_freezing),
            )
            if self.verbose:
                log.info(
                    f"Current lr: {optimizer.param_groups[0]['lr']}, " f"Backbone lr: {optimizer.param_groups[1]['lr']}"
                )
        elif self.unfreeze_at_epoch is not None and current_epoch == self.freeze_again_at_epoch:
            self.freeze(backbone, train_bn=self.train_bn_when_freezing)
            optimizer.param_groups.pop()


class RegNetFineTune(BaseFinetuning):
    def __init__(
        self,
        unfreeze_at_epoch=10,
        freeze_again_at_epoch: Optional[int] = None,
        initial_denom_lr=10,
        train_bn=True,
    ):
        super().__init__()
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.freeze_again_at_epoch = freeze_again_at_epoch
        self.train_bn_when_freezing = train_bn
        self.initial_denom_lr = initial_denom_lr

        if self.freeze_again_at_epoch is not None:
            assert self.unfreeze_at_epoch < self.freeze_again_at_epoch

    def freeze_before_training(self, pl_module):
        self.freeze([pl_module.net.stem, pl_module.net.trunk_output], train_bn=self.train_bn_when_freezing)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        if current_epoch == self.unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=[pl_module.net.stem, pl_module.net.trunk_output],
                optimizer=optimizer,
                initial_denom_lr=self.initial_denom_lr,
                train_bn=(not self.train_bn_when_freezing),
            )
        elif self.unfreeze_at_epoch is not None and current_epoch == self.freeze_again_at_epoch:
            self.freeze([pl_module.net.stem, pl_module.net.trunk_output], train_bn=self.train_bn_when_freezing)
            optimizer.param_groups.pop()


class ConvNeXtFineTune(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10, freeze_again_at_epoch: Optional[int] = None, initial_denom_lr=10):
        super().__init__()
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.freeze_again_at_epoch = freeze_again_at_epoch
        self.initial_denom_lr = initial_denom_lr

    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.net.features)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        if current_epoch == self.unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=pl_module.net.features,
                optimizer=optimizer,
                initial_denom_lr=self.initial_denom_lr,
            )
        elif self.unfreeze_at_epoch is not None and current_epoch == self.freeze_again_at_epoch:
            self.freeze(pl_module.net.features)
            optimizer.param_groups.pop()


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
        save_last="link",
    )


class LayerSummary(Callback):
    def __init__(self, depth=3):
        self.depth = depth

    def on_fit_start(self, trainer, pl_module):
        batch, _ = next(iter(trainer.datamodule.val_dataloader()))  # type: ignore
        batch = batch.to(pl_module.device)
        summary(
            pl_module,
            input_data=batch,
            col_names=("input_size", "output_size", "num_params", "mult_adds", "trainable"),
            depth=self.depth,
        )


class FineTuneBatchSizeFinder(BatchSizeFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.scale_batch_size(trainer, pl_module)
