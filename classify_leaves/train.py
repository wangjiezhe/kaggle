#!/usr/bin/python3

import pytorch_lightning as pl
import torch
from callback import *
from data import LeavesData
from model import *
from pytorch_lightning.loggers import TensorBoardLogger
from termcolor import colored
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
    **kwargs,
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

    model_name = model.hparams.get("model", model.__class__.__name__)
    if model_name:
        print(colored(f"Start training {model_name} in {epoch} epoches...", "green"))
    ck = [save_last(model_name)]
    if test:
        ck += [ConfMat()]
    if predict:
        ck += [Submission()]
    if best_model:
        ck += [
            save_best_train_loss(model_name),
            save_best_val_loss(model_name),
            save_best_val_acc(model_name),
        ]
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
    callbacks = [Submission()] if write else None
    trainer = pl.Trainer(max_epochs=0, logger=logger, callbacks=callbacks)  # type: ignore
    return trainer.predict(model, data)


# ResNet18/34: batch_size=128
# ResNet50, RegNetX_1.6GF: batch_size=64
# RegNetX_3.2/8GF, RegNetY_3.2GF: batch_size=32
# RegNetY_8GF: batch_size=24
# ConvNeXt Small: batch_size=32 (混合模式)
# ConvNeXt Base: batch_size=32


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
        callbacks=ConvNeXtFineTune(),
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
        callbacks=ConvNeXtFineTune(),
        mix=mixup,
    )


def train_convnext_4():
    model = ConvNeXt("convnext_tiny", weights="DEFAULT")
    train(
        model,
        20,
        predict=True,
        batch_size=32,
        split=False,
        best_model=True,
        logger_version="ConvNeXt tiny - Freeze",
        callbacks=ConvNeXtFineTune(),
    )


def train_regnet_1():
    model = RegNet("regnet_x_3_2gf", weights="DEFAULT")
    train(
        model,
        20,
        predict=True,
        batch_size=32,
        split=False,
        best_model=True,
        logger_version="RegNetX_3.2GF - Freeze",
        callbacks=RegNetFineTune(train_bn=False),
    )


def train_regnet_2():
    model = RegNet("regnet_y_3_2gf", weights="DEFAULT")
    train(
        model,
        20,
        predict=True,
        batch_size=32,
        split=False,
        best_model=True,
        logger_version="RegNetY_3.2GF - Freeze",
        callbacks=RegNetFineTune(train_bn=False),
    )


def train_regnet_3():
    model = RegNet("regnet_x_1_6gf", weights="DEFAULT")
    train(
        model,
        20,
        predict=True,
        batch_size=64,
        split=False,
        best_model=True,
        logger_version="RegNetX_1.6GF - Freeze",
        callbacks=RegNetFineTune(train_bn=True),
    )


def train_regnet_4():
    model = RegNet("regnet_y_1_6gf", weights="DEFAULT")
    train(
        model,
        20,
        predict=True,
        batch_size=32,
        split=False,
        best_model=True,
        logger_version="RegNetY_1.6GF - Freeze",
        callbacks=RegNetFineTune(train_bn=True),
    )


def train_regnet_5():
    model = RegNet("regnet_x_3_2gf", weights="DEFAULT")
    train(
        model,
        20,
        predict=True,
        batch_size=32,
        split=False,
        best_model=True,
        logger_version="RegNetX_3.2GF - Freeze (correct)",
        callbacks=RegNetFineTune(train_bn=True),
    )


def train_regnet_6():
    model = RegNet("regnet_y_3_2gf", weights="DEFAULT")
    train(
        model,
        20,
        predict=True,
        batch_size=32,
        split=False,
        best_model=True,
        logger_version="RegNetY_3.2GF - Freeze (correct)",
        callbacks=RegNetFineTune(train_bn=True),
    )


def train_regnet_7():
    model = RegNet("regnet_y_3_2gf")
    train(
        model,
        20,  # acturally stopped after 10 epoches
        checkpoint_path="./lightning_logs/RegNetY_3.2GF - Freeze (correct)/checkpoints/regnet_y_3_2gf-epoch=19-step=11220.ckpt",
        predict=True,
        batch_size=32,
        split=False,
        best_model=True,
        logger_version="RegNetY_3.2GF - Freeze (+20)",
        callbacks=RegNetFineTune(train_bn=True),
    )


def train_regnet_8():
    model = RegNet("regnet_y_3_2gf")
    train(
        model,
        10,
        checkpoint_path="./checkpoints/regnet_y_3_2gf-epoch=09-step=5610.ckpt",
        predict=True,
        batch_size=32,
        split=False,
        best_model=True,
        logger_version="RegNetY_3.2GF - Freeze (20+10+10)",
        callbacks=RegNetFineTune(train_bn=True),
    )


def train_regnet_9():
    model = RegNet("regnet_y_3_2gf")
    train(
        model,
        10,
        checkpoint_path="./checkpoints/regnet_y_3_2gf-epoch=09-step=5610-v1.ckpt",
        predict=True,
        batch_size=32,
        split=False,
        best_model=True,
        logger_version="RegNetY_3.2GF - Freeze (20+10+10+10)",
        callbacks=RegNetFineTune(unfreeze_at_epoch=0, train_bn=True),
    )


def train_regnet_10():
    model = RegNet("regnet_y_3_2gf")
    train(
        model,
        10,
        checkpoint_path="./checkpoints/regnet_y_3_2gf-epoch=09-step=5610-v2.ckpt",
        predict=True,
        batch_size=32,
        split=False,
        best_model=True,
        logger_version="RegNetY_3.2GF - Freeze (20+10+10+10+10)",
        callbacks=RegNetFineTune(unfreeze_at_epoch=10, train_bn=True),
    )


def train_resnet_1():
    model = ResNet("resnet50", weights="DEFAULT")
    train(
        model,
        20,
        predict=True,
        batch_size=32,
        split=False,
        best_model=True,
        logger_version="ResNet50 - Freeze",
        callbacks=ResNetFineTune(),
    )


def train_resnet_2():
    model = ResNet("resnet34", weights="DEFAULT")
    train(
        model,
        20,
        predict=True,
        batch_size=32,
        split=False,
        best_model=True,
        logger_version="ResNet34 - Freeze",
        callbacks=ResNetFineTune(),
    )


def train_resnet_3():
    model = ResNet("resnet18", weights="DEFAULT")
    train(
        model,
        20,
        predict=True,
        batch_size=32,
        split=False,
        best_model=True,
        logger_version="ResNet18 - Freeze",
        callbacks=ResNetFineTune(),
    )


if __name__ == "__main__":
    # train_convnext_1()
    # train_convnext_2()
    # train_convnext_3()
    # train_convnext_4()
    # train_regnet_1()
    # train_regnet_2()
    # train_regnet_3()
    # train_regnet_4()
    # train_regnet_5()
    # train_resnet_1()
    # train_resnet_2()
    # train_resnet_3()
    # train_regnet_6()
    # train_regnet_7()
    # train_regnet_8()
    # train_regnet_9()
    train_regnet_10()
