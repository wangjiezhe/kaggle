#!/usr/bin/python3

import pytorch_lightning as pl
import torch
from callback import *
from data import LeavesData
from model import *
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import models
from torchvision.transforms import v2
from utils import *

torch.set_float32_matmul_precision("high")

train_aug = v2.Compose(
    [
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(90),  # type: ignore
        v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    ]
)


def train(
    model,
    epoch,
    best_model=False,
    callbacks=None,
    logger_version=None,
    checkpoint_path=None,
    weights_path=None,
    save_checkpoint_path=None,
    save_weights_path=None,
    batch_size=128,
    num_workers=8,
    aug=train_aug,
    split=True,
    mix=None,
    test=True,
    predict=False,
    **kwargs
):
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path)
        model.load_state_dict(state["state_dict"], strict=False)
    elif weights_path is not None:
        state = torch.load(weights_path)
        assert "state_dict" not in state.keys()
        model.load_state_dict(state, strict=False)

    data = LeavesData(batch_size=batch_size, num_workers=num_workers, aug=aug, split=split, mix=mix)
    logger = TensorBoardLogger(".", default_hp_metric=False, version=logger_version)

    ck = [save_last]
    if test:
        ck += [analyse_confusion_matrix]
    if predict:
        ck += [save_prediction]
    if best_model:
        ck += [save_best_train_loss, save_best_val_loss, save_best_val_acc]
    if callbacks:
        if isinstance(callbacks, Callback):
            callbacks = [callbacks]
        ck += callbacks

    trainer = pl.Trainer(max_epochs=epoch, logger=logger, callbacks=ck, **kwargs)  # type: ignore
    trainer.fit(model, data)
    if test:
        trainer.test(model, data)
    if predict:
        trainer.predict(model, data)
    if save_checkpoint_path is not None:
        trainer.save_checkpoint(save_checkpoint_path)
    if save_weights_path is not None:
        torch.save(model.state_dict(), save_weights_path)


def predict(model, batch_size=128, checkpoint_path=None, weights_path=None, write=True):
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path)
        model.load_state_dict(state["state_dict"], strict=False)
    elif weights_path is not None:
        state = torch.load(weights_path)
        assert "state_dict" not in state.keys()
        model.load_state_dict(state, strict=False)
    data = LeavesData(batch_size=batch_size)
    logger = TensorBoardLogger(".", default_hp_metric=False)
    callbacks = [save_prediction] if write else None
    trainer = pl.Trainer(max_epochs=0, logger=logger, callbacks=callbacks)  # type: ignore
    return trainer.predict(model, data)


# ResNet18/34: batch_size=128
# ResNet50, RegNetX_1.6GF: batch_size=64
# RegNetX_3.2/8GF, RegNetY_3.2GF, ConvNeXt Small: batch_size=32
# RegNetY_8GF: batch_size=24
# ConvNeXt Base:


def train_convnext_1():
    model = ConvNeXt("convnext_small", weights="DEFAULT")
    train(
        model,
        20,
        predict=True,
        batch_size=32,
        split=False,
        best_model=True,
        logger_version="ConvNeXt small - Finetune",
    )


def train_convnext_2():
    model = ConvNeXt("convnext_small", weights="DEFAULT")
    train(
        model,
        20,
        predict=True,
        batch_size=32,
        split=False,
        best_model=True,
        logger_version="ConvNeXt small - Finetune - Freeze",
        callbacks=ConvNeXtFineTune(train_bn=True),
    )


def train_convnext_3():
    model = ConvNeXt("convnext_small", weights="DEFAULT")
    train(
        model,
        20,
        predict=True,
        batch_size=32,
        split=False,
        best_model=True,
        logger_version="ConvNeXt small - Freeze - Mixup",
        callbacks=ConvNeXtFineTune(train_bn=False),
        mix=mixup,
    )


def train_regnet_1():
    model = RegNet("regnet_x_3_2gf", weights="DEFAULT")
    train(
        model,
        30,
        predict=True,
        batch_size=32,
        split=False,
        best_model=True,
        logger_version="RegNetX_3.2GF - Freeze",
        callbacks=ConvNeXtFineTune(),
    )


if __name__ == "__main__":
    # model = ResNet_pretrained(models.regnet_x_3_2gf, models.RegNet_X_3_2GF_Weights.IMAGENET1K_V2)
    # train(model, 8, predict=True, batch_size=32)
    model = RegNet("regnet_y_3_2gf", weights="DEFAULT")
    round = 3
    for i in range(round):
        train(model, 4, predict=True, batch_size=32)
