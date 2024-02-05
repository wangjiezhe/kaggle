#!/usr/bin/python3

import re
from typing import Callable, List, Optional, Union

import pytorch_lightning as pl
import torch
from callback import *
from data import *
from model import *
from pytorch_lightning.callbacks import BackboneFinetuning, Callback
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from termcolor import colored
from torchvision.transforms import v2
from utils import *

torch.set_float32_matmul_precision("medium")


def train(
    model: Union[pl.LightningModule, str],
    epoch: int,
    pretrained=False,
    best_model=True,
    best_model_dirpath: Optional[str] = None,
    callbacks: Optional[Union[List[Callback], Callback]] = None,
    logger_version: Optional[Union[int, str, bool]] = None,
    checkpoint_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    save_checkpoint_path: Optional[str] = None,
    save_weights_path: Optional[str] = None,
    data=None,
    batch_size=128,
    num_workers=8,
    resize=224,
    aug: Optional[Callable] = None,
    split=True,
    split_random_seed=42,
    five_crop=False,
    mix: Optional[Callable] = None,
    test=True,
    prediction=True,
    **kwargs,
):
    if isinstance(model, str):
        model = DefinedNet(model, pretrained=pretrained)

    if checkpoint_path is not None:
        state = torch.load(checkpoint_path)
        model.load_state_dict(state["state_dict"], strict=False)
    elif weights_path is not None:
        state = torch.load(weights_path)
        assert "state_dict" not in state.keys()
        model.load_state_dict(state, strict=False)

    if data is None:
        data = DogData(
            batch_size=batch_size,
            num_workers=num_workers,
            resize=resize,
            aug=aug,
            split=split,
            split_random_seed=split_random_seed,
            five_crop=five_crop,
            mix=mix,
        )
    tb_logger = TensorBoardLogger(".", default_hp_metric=False, version=logger_version)
    csv_logger = CSVLogger(".", name="csv_logs", version=logger_version, flush_logs_every_n_steps=10)

    model_name = model.hparams.get("model", model.__class__.__name__)
    if model_name:
        print(colored(f"Start training {model_name} in {epoch} epoches...", "green"))
    ck: List[Callback] = [save_last(model_name, dirpath=best_model_dirpath)]
    if test:
        ck += [ConfMatTopK(20)]
    if prediction:
        ck += [Submission()]
    if best_model:
        best_val_acc = save_best_val_acc(model_name, dirpath=best_model_dirpath)
        ck += [
            save_best_train_loss(model_name, dirpath=best_model_dirpath),
            save_best_val_loss(model_name, dirpath=best_model_dirpath),
            best_val_acc,
        ]
    if callbacks:
        if isinstance(callbacks, Callback):
            callbacks = [callbacks]
        ck += callbacks

    trainer = pl.Trainer(max_epochs=epoch, logger=[tb_logger, csv_logger], callbacks=ck, **kwargs)
    trainer.fit(model, data)
    if test:
        trainer.test(model, data)
    if save_checkpoint_path is not None:
        trainer.save_checkpoint(save_checkpoint_path)
    if save_weights_path is not None:
        torch.save(model.state_dict(), save_weights_path)

    if prediction:
        trainer.predict(model, data)
        if best_model:
            best_val_acc_epoch = int(re.findall(r"epoch=(\d+)", best_val_acc.best_model_path)[0])  # type: ignore
            if best_val_acc_epoch < trainer.current_epoch - 1:
                trainer.predict(model, data, ckpt_path=best_val_acc.best_model_path)  # type: ignore


def predict(model, data=None, batch_size=128, five_crop=False, checkpoint_path=None, weights_path=None, write=True):
    if isinstance(model, str):
        model = DefinedNet(model)
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path)
        model.load_state_dict(state["state_dict"], strict=False)
    elif weights_path is not None:
        state = torch.load(weights_path)
        assert "state_dict" not in state.keys()
        model.load_state_dict(state, strict=False)
    if data is None:
        data = DogData(batch_size=batch_size, five_crop=five_crop)
    callbacks: Optional[List[Callback]] = [Submission()] if write else None
    trainer = pl.Trainer(max_epochs=0, logger=False, callbacks=callbacks)
    return trainer.predict(model, data)


def train_resnet18_1():
    model = ResNet("resnet18", pretrained=True)
    train(
        model,
        30,
        batch_size=32,
        num_workers=16,
        callbacks=ResNetFineTune(30),
        logger_version="ResNet18_freeze_30",
        prediction=False,
    )


def train_resnet18_2():
    model = ResNet("resnet18", pretrained=True)
    train(
        model,
        30,
        batch_size=32,
        num_workers=16,
        callbacks=ResNetFineTune(5, 25),
        logger_version="ResNet18_freeze_5_25",
        prediction=False,
    )


def train_resnet18_3():
    model = ResNet("resnet18", pretrained=True)
    train(
        model,
        30,
        batch_size=32,
        num_workers=16,
        callbacks=ResNetFineTune(5, 25),
        logger_version="ResNet18_mix_freeze_5_25",
        mix=cutmix_or_mixup,
        prediction=False,
    )


def train_resnet18_4():
    model = ResNet("resnet18", pretrained=True)
    train(
        model,
        30,
        batch_size=32,
        num_workers=16,
        callbacks=ResNetFineTune(5, 25),
        logger_version="ResNet18_mix_freeze_5_25_autoaug",
        mix=cutmix_or_mixup,
        aug=v2.AutoAugment(),
        prediction=False,
    )


def train_resnet34_1():
    model = ResNet("resnet34", pretrained=True)
    train(
        model,
        30,
        batch_size=32,
        num_workers=16,
        callbacks=ResNetFineTune(5, 25),
        logger_version="ResNet34_freeze_5_25",
        prediction=False,
    )


def train_resnet50_1():
    model = ResNet("resnet50", pretrained=True)
    train(
        model,
        30,
        batch_size=32,
        num_workers=16,
        callbacks=ResNetFineTune(5, 25),
        logger_version="ResNet50_freeze_5_25",
        prediction=False,
    )


def train_resnet101_1():
    model = ResNet("resnet101", pretrained=True)
    train(
        model,
        30,
        batch_size=32,
        num_workers=16,
        callbacks=ResNetFineTune(5, 25),
        logger_version="ResNet101_freeze_5_25",
        prediction=False,
    )


def train_resnet152_1():
    model = ResNet("resnet152", pretrained=True)
    train(
        model,
        30,
        batch_size=16,
        num_workers=16,
        callbacks=ResNetFineTune(5, 25),
        logger_version="ResNet152_freeze_5_25",
        prediction=False,
    )


def train_resnet152_2():
    model = ResNet("resnet152", pretrained=True)
    train(
        model,
        10,
        batch_size=16,
        num_workers=16,
        callbacks=ResNetFineTune(30),
        logger_version="ResNet152_mix_freeze_all",
        mix=cutmix_or_mixup,
        prediction=True,
    )


def train_resnet152_3():
    train(
        "resnet152",
        10,
        pretrained=True,
        batch_size=16,
        num_workers=16,
        callbacks=BackboneFinetuning(30, lambda_func=lambda epoch: 0.95, verbose=True),
        logger_version="ResNet152_mix_freeze_all_DefinedNet",
        mix=cutmix_or_mixup,
        prediction=True,
    )


def train_resnet152_4():
    model = ResNet("resnet152", pretrained=True)
    train(
        model,
        10,
        batch_size=16,
        num_workers=16,
        callbacks=ResNetFineTune(0),
        logger_version="ResNet152_finetune",
        prediction=True,
        check_val_every_n_epoch=2,
        split=False,
        best_model=False,
    )


def train_resnet152_5():
    model = ResNet("resnet152", pretrained=True)
    train(
        model,
        20,
        batch_size=32,
        num_workers=16,
        callbacks=ResNetFineTune(30),
        logger_version="ResNet152_resize96_mix_freeze_all",
        mix=cutmix_or_mixup,
        prediction=False,
        resize=96,
    )


def train_resnet152_6():
    model = ResNet("resnet152", pretrained=True)
    train(
        model,
        20,
        batch_size=32,
        num_workers=16,
        callbacks=ResNetFineTune(5, 15),
        logger_version="ResNet152_resize96_mix_freeze_5_15",
        mix=cutmix_or_mixup,
        prediction=False,
        resize=96,
    )


def train_regnety_1():
    model = RegNet("regnet_y_3_2gf", pretrained=True)
    train(
        model,
        30,
        batch_size=32,
        num_workers=16,
        callbacks=ResNetFineTune(5, 25),
        logger_version="RegNetY_3.2GF_mix_freeze_5_25",
        mix=cutmix_or_mixup,
        prediction=False,
    )


def train_regnety_2():
    model = RegNet("regnet_y_3_2gf", pretrained=True)
    train(
        model,
        30,
        batch_size=32,
        num_workers=16,
        callbacks=RegNetFineTune(50),
        logger_version="RegNetY_3.2GF_mix_freeze_all",
        mix=cutmix_or_mixup,
        prediction=False,
    )


def train_regnety_3():
    model = RegNet("regnet_y_8gf", pretrained=True)
    train(
        model,
        30,
        batch_size=24,
        num_workers=8,
        callbacks=RegNetFineTune(50),
        logger_version="RegNetY_8GF_mix_freeze_all",
        mix=cutmix_or_mixup,
        prediction=False,
    )


def train_regnety_3_5():
    model = RegNet("regnet_y_16gf", pretrained=True)
    train(
        model,
        30,
        batch_size=16,
        num_workers=8,
        callbacks=RegNetFineTune(50),
        logger_version="RegNetY_16GF_mix_freeze_all",
        mix=cutmix_or_mixup,
        prediction=False,
    )


def train_convnext_1():
    model = ConvNeXt("convnext_tiny", pretrained=True)
    train(
        model,
        30,
        batch_size=32,
        num_workers=8,
        callbacks=ConvNeXtFineTune(5, 25),
        logger_version="ConvNeXt_Tiny_freeze_5_25",
        prediction=False,
    )


def train_convnext_2():
    model = ConvNeXt("convnext_tiny", pretrained=True)
    train(
        model,
        30,
        batch_size=32,
        num_workers=8,
        callbacks=ConvNeXtFineTune(5, 25),
        logger_version="ConvNeXt_Tiny_mix_freeze_5_25",
        mix=cutmix_or_mixup,
        prediction=False,
    )


def train_convnext_3():
    model = ConvNeXt("convnext_tiny", pretrained=True)
    train(
        model,
        30,
        batch_size=32,
        num_workers=8,
        callbacks=ConvNeXtFineTune(50),
        logger_version="ConvNeXt_Tiny_mix_freeze_all",
        mix=cutmix_or_mixup,
        prediction=False,
    )


def train_convnext_4():
    model = ConvNeXt("convnext_small", pretrained=True)
    train(
        model,
        30,
        batch_size=32,
        num_workers=8,
        callbacks=ConvNeXtFineTune(50),
        logger_version="ConvNeXt_Small_mix_freeze_all",
        mix=cutmix_or_mixup,
        prediction=False,
    )


def train_convnext_5():
    model = ConvNeXt("convnext_small", pretrained=True)
    train(
        model,
        30,
        batch_size=24,
        num_workers=8,
        callbacks=ConvNeXtFineTune(5, 25, 100),
        logger_version="ConvNeXt_Small_mix_freeze_5_25",
        mix=cutmix_or_mixup,
        prediction=False,
    )


def train_convnext_6():
    model = ConvNeXt("convnext_base", pretrained=True)
    train(
        model,
        30,
        batch_size=32,
        num_workers=8,
        callbacks=ConvNeXtFineTune(50),
        logger_version="ConvNeXt_Base_mix_freeze_all",
        mix=cutmix_or_mixup,
        prediction=True,
    )


def train_convnext_7():
    model = ConvNeXt("convnext_base", pretrained=True)
    train(
        model,
        20,
        batch_size=16,
        num_workers=8,
        callbacks=ConvNeXtFineTune(5, 15, 100),
        logger_version="ConvNeXt_Base_mix_freeze_5_15",
        mix=cutmix_or_mixup,
        prediction=True,
    )


def train_convnext_8():
    model = ConvNeXt("convnext_large", pretrained=True)
    train(
        model,
        30,
        batch_size=32,
        num_workers=8,
        callbacks=ConvNeXtFineTune(50),
        logger_version="ConvNeXt_Large_mix_freeze_all",
        mix=cutmix_or_mixup,
        prediction=True,
    )


def train_convnext_9():
    model = ConvNeXt("convnext_large", pretrained=True)
    train(
        model,
        20,
        batch_size=8,
        num_workers=8,
        callbacks=ConvNeXtFineTune(5, 15, 100),
        logger_version="ConvNeXt_Large_mix_freeze_5_15",
        mix=cutmix_or_mixup,
        prediction=True,
    )


def train_regnety_4():
    model = RegNet("regnet_y_32gf", pretrained=True)
    train(
        model,
        30,
        batch_size=16,
        num_workers=8,
        callbacks=[RegNetFineTune(50), LayerSummary()],
        logger_version="RegNetY_32GF_mix_freeze_all",
        mix=cutmix_or_mixup,
        prediction=False,
    )


def train_regnety_5():
    model = RegNet("regnet_y_32gf", pretrained=True)
    train(
        model,
        30,
        batch_size=32,
        num_workers=8,
        callbacks=[RegNetFineTune(50, train_bn=False), LayerSummary()],
        logger_version="RegNetY_32GF_mix_freeze_all_no_bn",
        mix=cutmix_or_mixup,
        prediction=False,
    )


def train_regnety_6():
    model = RegNet("regnet_y_128gf", pretrained=True)
    train(
        model,
        30,
        batch_size=4,
        num_workers=8,
        callbacks=RegNetFineTune(50),
        logger_version="RegNetY_128GF_mix_freeze_all_no_bn",
        mix=cutmix_or_mixup,
        prediction=False,
    )


def train_convnext_10():
    model = ConvNeXt("convnext_small")
    train(
        model,
        100,
        batch_size=32,
        num_workers=8,
        callbacks=[LayerSummary()],
        logger_version="ConvNeXt_Small_fp16",
        prediction=False,
        precision="16-mixed",
    )


if __name__ == "__main__":
    # train_resnet18_1()
    # train_resnet18_2()
    # train_resnet18_3()
    # train_resnet18_4()
    # train_resnet34_1()
    # train_resnet50_1()
    # train_resnet101_1()
    # train_resnet152_1()
    # train_resnet152_2()
    # train_resnet152_3()
    # train_resnet152_4()
    # train_resnet152_5()
    # train_resnet152_6()
    # train_regnety_1()
    # train_regnety_2()
    # train_regnety_3()
    # train_convnext_1()
    # train_convnext_2()
    # train_convnext_3()
    # train_convnext_4()
    # train_convnext_5()
    # train_convnext_6()
    # train_convnext_7()
    # train_convnext_8()
    # train_convnext_9()
    # train_regnety_4()
    # train_regnety_5()
    # train_regnety_6()
    train_convnext_10()
    pass
