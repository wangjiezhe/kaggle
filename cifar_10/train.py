#!/usr/bin/python3

import re
from typing import Callable, List, Optional, Union

import pytorch_lightning as pl
import torch
from callback import *
from callback import LayerSummary
from data import *
from model import *
from pytorch_lightning.callbacks import BackboneFinetuning, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from termcolor import colored
from torchinfo import summary
from torchvision.ops import Conv2dNormActivation
from utils import *

torch.set_float32_matmul_precision("medium")


def train(
    model: Union[pl.LightningModule, str],
    epoch: int,
    pretrained=False,
    best_model=True,
    best_model_dirpath: Optional[str] = None,
    callbacks: Optional[Union[List[Callback], Callback]] = None,
    logger_version: Optional[Union[int, str]] = None,
    checkpoint_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    save_checkpoint_path: Optional[str] = None,
    save_weights_path: Optional[str] = None,
    data=None,
    batch_size=128,
    num_workers=8,
    aug: Optional[Callable] = None,
    split=True,
    split_random=False,
    mix: Optional[Callable] = None,
    test=True,
    predict=True,
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
        data = Cifar10Data(
            batch_size=batch_size,
            num_workers=num_workers,
            aug=aug,
            split=split,
            split_random=split_random,
            mix=mix,
        )
    logger = TensorBoardLogger(".", default_hp_metric=False, version=logger_version)

    model_name = model.hparams.get("model", model.__class__.__name__)
    if model_name:
        print(colored(f"Start training {model_name} in {epoch} epoches...", "green"))
    ck = [save_last(model_name, dirpath=best_model_dirpath)]
    if test:
        ck += [ConfMatTop20()]
    if predict:
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

    trainer = pl.Trainer(max_epochs=epoch, logger=logger, callbacks=ck, **kwargs)
    trainer.fit(model, data)
    if test:
        trainer.test(model, data)
    if save_checkpoint_path is not None:
        trainer.save_checkpoint(save_checkpoint_path)
    if save_weights_path is not None:
        torch.save(model.state_dict(), save_weights_path)

    if predict:
        trainer.predict(model, data)
        if best_model:
            best_val_acc_epoch = int(re.findall("epoch=(\d+)", best_val_acc.best_model_path)[0])
            if best_val_acc_epoch < trainer.current_epoch - 1:
                trainer.predict(model, data, ckpt_path=best_val_acc.best_model_path)


def predict(model, data=None, batch_size=128, checkpoint_path=None, weights_path=None, write=True):
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
        data = Cifar10Data(batch_size=batch_size)
    callbacks = [Submission()] if write else None
    trainer = pl.Trainer(max_epochs=0, logger=False, callbacks=callbacks)
    return trainer.predict(model, data)


def train_resnet18_1():
    model = ResNet("resnet18", pretrained=True)
    train(
        model,
        100,
        callbacks=ResNetFineTune(10, 90),
        logger_version="ResNet18_freeze_10_90",
        split_random=True,
    )


def train_resnet18_2():
    model = ResNet("resnet18")
    train(model, 100, logger_version="ResNet18", split_random=True)


def train_resnet18_3():
    train(
        "resnet18",
        100,
        pretrained=True,
        callbacks=BackboneFinetuning(10, verbose=True),
        logger_version="DefinedNet_resnet18_freeze_10",
        split_random=True,
    )


def train_resnet18_4():
    train(
        "resnet18",
        100,
        pretrained=True,
        callbacks=BackboneFinetuning(10, lambda_func=lambda epoch: 1.0, verbose=True),
        logger_version="DefinedNet_resnet18_freeze_10_1x",
        split_random=True,
    )


def train_resnet18_5():
    train(
        "resnet18",
        100,
        pretrained=True,
        callbacks=BackboneFinetuning(10, lambda_func=lambda epoch: 0.95, verbose=True),
        logger_version="DefinedNet_resnet18_freeze_10_0.95x",
        split_random=True,
    )


def train_resnet34_1():
    model = ResNet("resnet34", pretrained=True)
    train(
        model,
        100,
        callbacks=ResNetFineTune(10, 90),
        logger_version="ResNet34_freeze_10_90",
        split_random=True,
    )


def train_resnet18_6():
    model = ResNet("resnet18")
    train(model, 100, logger_version="ResNet18_mix", mix=cutmix_or_mixup, num_workers=16, split_random=True)


def train_resnet50_1():
    model = ResNet("resnet50", pretrained=True)
    train(
        model,
        100,
        callbacks=ResNetFineTune(10, 90),
        logger_version="ResNet50_freeze_10_90",
        split_random=True,
        num_workers=16,
    )


def train_resnet101_1():
    model = ResNet("resnet101", pretrained=True)
    train(
        model,
        100,
        callbacks=ResNetFineTune(10, 90),
        logger_version="ResNet101_freeze_10_90",
        split_random=True,
    )


def train_resnet152_1():
    model = ResNet("resnet152", pretrained=True)
    train(
        model,
        100,
        callbacks=ResNetFineTune(10, 90),
        logger_version="ResNet152_freeze_10_90",
        split_random=True,
        num_workers=16,
    )


def train_resnet50_2():
    model = ResNet("resnet50", pretrained=True)
    train(
        model,
        100,
        callbacks=ResNetFineTune(10, 90),
        logger_version="ResNet50_mix_freeze_10_90",
        split_random=True,
        num_workers=16,
        mix=cutmix_or_mixup,
    )


def train_resnet50_3():
    model = ResNet("resnet50", pretrained=True)
    data = ResizedCifar10Data(batch_size=128, num_workers=16, split_random=True, mix=cutmix_or_mixup)
    train(
        model,
        100,
        data=data,
        callbacks=ResNetFineTune(10, 90),
        logger_version="ResNet50_resize96_mix_freeze_10_90",
    )


def train_resnet50_4():
    model = ResNet("resnet50", pretrained=True)
    data = ResizedCifar10Data(resize=224, batch_size=32, split_random=True, mix=cutmix_or_mixup)
    epoch = 25
    train(
        model,
        epoch,
        data=data,
        callbacks=ResNetFineTune(5, epoch - 5),
        logger_version=f"ResNet50_resize224_mix_freeze_10_{epoch-5}",
    )


def train_resnet50_5():
    model = ResNet("resnet50")
    model.net.conv1 = torch.nn.LazyConv2d(64, kernel_size=3, padding=1, bias=False)
    model.eval()
    print(
        summary(
            model, depth=2, input_size=(128, 3, 32, 32), col_names=("input_size", "output_size", "num_params")
        )
    )
    train(
        model,
        100,
        logger_version="ResNet50_mix_conv1=3x3",
        split_random=True,
        mix=cutmix_or_mixup,
        num_workers=16,
    )


def train_regnetx_1():
    class Stem(Conv2dNormActivation):
        def __init__(self, width_in, width_out, norm_layer, activation_layer):
            super().__init__(
                width_in,
                width_out,
                kernel_size=3,
                stride=1,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )

    model = RegNet("regnet_x_3_2gf", stem_type=Stem)
    model.eval()
    print(summary(model, input_size=(128, 3, 32, 32), col_names=("input_size", "output_size", "num_params")))
    train(model, 100, logger_version="RegNetX_3.2GF_mix_stem", mix=cutmix_or_mixup)


def train_regnetx_2():
    class Stem(Conv2dNormActivation):
        def __init__(self, width_in, width_out, norm_layer, activation_layer):
            super().__init__(
                width_in,
                width_out,
                kernel_size=3,
                stride=1,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )

    epoch = 100
    model = RegNet("regnet_x_3_2gf", cosine_t_max=epoch, stem_type=Stem)
    train(
        model, epoch, logger_version="RegNetX_3.2GF_fp16_mix_stem", mix=cutmix_or_mixup, precision="16-mixed"
    )


def train_regnetx_3():
    class Stem(Conv2dNormActivation):
        def __init__(self, width_in, width_out, norm_layer, activation_layer):
            super().__init__(
                width_in,
                width_out,
                kernel_size=3,
                stride=1,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )

    epoch = 100
    model = RegNet("regnet_x_3_2gf", cosine_t_max=epoch, stem_type=Stem)
    train(
        model,
        epoch,
        logger_version="RegNetX_3.2GF_bf16_mix_stem",
        mix=cutmix_or_mixup,
        precision="bf16-mixed",
    )


def train_regnetx_4():
    model = RegNet("regnet_x_3_2gf")
    data = ResizedCifar10Data(resize=96, batch_size=128, mix=cutmix_or_mixup)
    train(model, 100, data=data, logger_version="RegNetX_3.2GF_resize96")


def train_regnetx_4_1():
    model = RegNet("regnet_x_3_2gf")
    data = ResizedCifar10Data(resize=96, batch_size=128, mix=cutmix_or_mixup)
    train(
        model,
        100,
        data=data,
        logger_version="RegNetX_3.2GF_resize96_2",
        checkpoint_path="./lightning_logs/RegNetX_3.2GF_resize96/checkpoints/regnet_x_3_2gf-epoch=99-step=35200.ckpt",
    )


def train_regnetx_5():
    model = RegNet("regnet_x_3_2gf", pretrained=True)
    train(
        model,
        100,
        callbacks=ResNetFineTune(10, 90),
        logger_version="RegNetX_3.2GF_mix_freeze_10_90",
        split=True,
        split_random=False,
        mix=cutmix_or_mixup,
    )


def train_regnetx_6():
    model = RegNet("regnet_x_3_2gf", pretrained=True)
    data = ResizedCifar10Data(resize=96, batch_size=128, mix=cutmix_or_mixup)
    train(
        model,
        100,
        data=data,
        callbacks=ResNetFineTune(10, 90),
        logger_version="RegNetX_3.2GF_resize96_mix_freeze_10_90",
    )


def train_regnety_1():
    model = RegNet("regnet_y_3_2gf", pretrained=True)
    data = ResizedCifar10Data(resize=96, batch_size=128, mix=cutmix_or_mixup)
    train(
        model,
        100,
        data=data,
        callbacks=ResNetFineTune(10, 90),
        logger_version="RegNetY_3.2GF_resize96_mix_freeze_10_90",
    )


def train_convnext_1():
    model = ConvNeXt("convnext_tiny", pretrained=True)
    data = ResizedCifar10Data(resize=96, batch_size=128, mix=cutmix_or_mixup)
    epoch = 50
    freeze = 5
    train(
        model,
        epoch,
        data=data,
        callbacks=[ConvNeXtFineTune(freeze, epoch - freeze), LayerSummary()],
        logger_version=f"ConvNeXt_Tiny_resize96_mix_freeze_{freeze}_{epoch - freeze}",
    )


def train_convnext_2():
    model = ConvNeXt("convnext_small", pretrained=True)
    data = ResizedCifar10Data(resize=96, batch_size=128, mix=cutmix_or_mixup)
    epoch = 20
    freeze = 5
    train(
        model,
        epoch,
        data=data,
        callbacks=[ConvNeXtFineTune(freeze, epoch - freeze), LayerSummary()],
        logger_version=f"ConvNeXt_Small_resize96_mix_freeze_{freeze}_{epoch - freeze}",
    )


def train_convnext_3():
    model = ConvNeXt("convnext_base", pretrained=True)
    data = ResizedCifar10Data(resize=96, batch_size=64, mix=cutmix_or_mixup)
    train(
        model,
        10,
        data=data,
        callbacks=[ConvNeXtFineTune(1, 8), LayerSummary()],
        logger_version="ConvNeXt_Base_resize96_mix_freeze_1_8",
    )


def train_convnext_4():
    model = ConvNeXt("convnext_base", pretrained=True)
    data = ResizedCifar10Data(resize=96, batch_size=64)
    train(
        model,
        10,
        data=data,
        callbacks=[ConvNeXtFineTune(1, 8), LayerSummary()],
        logger_version="ConvNeXt_Base_resize96_freeze_1_8",
    )


if __name__ == "__main__":
    # train_resnet18_1()
    # train_resnet18_2()
    # train_resnet18_3()
    # train_resnet18_4()
    # train_resnet18_5()
    # train_resnet34_1()
    # train_resnet18_6()
    # train_resnet50_1()
    # train_resnet101_1()
    # train_resnet152_1()
    # train_resnet50_2()
    # train_resnet50_3()
    # train_resnet50_4()
    # train_resnet50_5()
    # train_regnetx_1()
    # train_regnetx_2()
    # train_regnetx_3()
    # train_regnetx_4()
    # train_regnetx_4_1()
    # train_regnetx_5()
    # train_regnetx_6()
    # train_regnety_1()
    # train_convnext_1()
    # train_convnext_2()
    # train_convnext_3()
    train_convnext_4()
    pass
