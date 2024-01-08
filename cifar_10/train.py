#!/usr/bin/python3

from typing import Callable, List, Optional, Union

import pytorch_lightning as pl
import torch
from callback import *
from data import *
from model import *
from pytorch_lightning.loggers import TensorBoardLogger
from termcolor import colored
from torchvision import models
from torchvision.transforms import v2
from utils import *

torch.set_float32_matmul_precision("high")


def train(
    model: Union[pl.LightningDataModule, str],
    epoch: int,
    pretrained=False,
    best_model=True,
    callbacks: Optional[Union[List[pl.Callback], pl.Callback]] = None,
    logger_version: Optional[Union[int, str]] = None,
    checkpoint_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    save_checkpoint_path: Optional[str] = None,
    save_weights_path: Optional[str] = None,
    batch_size=128,
    num_workers=8,
    aug: Optional[Callable] = None,
    split=False,
    mix: Optional[Callable] = None,
    test=True,
    predict=True,
    **kwargs,
):
    if isinstance(model, str):
        if pretrained:
            model = models.get_model(model, weights="DEFAULT")
        else:
            model = models.get_model(model)

    if checkpoint_path is not None:
        state = torch.load(checkpoint_path)
        model.load_state_dict(state["state_dict"], strict=False)
    elif weights_path is not None:
        state = torch.load(weights_path)
        assert "state_dict" not in state.keys()
        model.load_state_dict(state, strict=False)

    data = Cifar10Data(batch_size=batch_size, num_workers=num_workers, aug=aug, split=split, mix=mix)
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
        if isinstance(callbacks, pl.Callback):
            callbacks = [callbacks]
        ck += callbacks

    trainer = pl.Trainer(max_epochs=epoch, logger=logger, callbacks=ck, **kwargs)
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
    data = Cifar10Data(batch_size=batch_size)
    logger = TensorBoardLogger(".", default_hp_metric=False)
    callbacks = [Submission()] if write else None
    trainer = pl.Trainer(max_epochs=0, logger=logger, callbacks=callbacks)
    return trainer.predict(model, data)


if __name__ == "__main__":
    pass
