#!/usr/bin/python3

import pytorch_lightning as pl
import torch
from data import LeavesData
from model import *
from pytorch_lightning.loggers import TensorBoardLogger
from regex import B
from torchvision import models

torch.set_float32_matmul_precision("high")


def train(
    model,
    epoch,
    logger_version=None,
    save_path=None,
    load_path=None,
    save_pickle_path=None,
    load_pickle_path=None,
    batch_size=128,
    num_workers=8,
    log_every_n_steps=5,
    test=True,
    predict=False,
    split=True,
):
    data = LeavesData(batch_size=batch_size, split=split, num_workers=num_workers)
    if load_path is not None:
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    elif load_pickle_path is not None:
        checkpoint = torch.load(load_pickle_path)
        model.load_state_dict(checkpoint, strict=False)
    logger = TensorBoardLogger(".", default_hp_metric=False, version=logger_version)
    trainer = pl.Trainer(max_epochs=epoch, logger=logger, log_every_n_steps=log_every_n_steps)
    trainer.fit(model, data)
    if test:
        trainer.test(model, data)
    if predict:
        trainer.predict(model, data)
    if save_path is not None:
        trainer.save_checkpoint(save_path)
    if save_pickle_path is not None:
        torch.save(model.state_dict(), save_pickle_path)


def predict(model, batch_size=128, load_path=None, load_pickle_path=None):
    if load_path is not None:
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    elif load_pickle_path is not None:
        checkpoint = torch.load(load_pickle_path)
        model.load_state_dict(checkpoint, strict=False)
    data = LeavesData(batch_size=batch_size)
    logger = TensorBoardLogger(".", default_hp_metric=False)
    trainer = pl.Trainer(max_epochs=1, logger=logger)
    trainer.predict(model, data)


if __name__ == "__main__":
    # ResNet18/34: batch_size=128
    # ResNet50, RegNetX_1.6GF: batch_size=64
    # RegNetX_3.2/8GF: batch_size=32
    model = ResNet_pretrained(models.regnet_x_3_2gf, models.regnet.RegNet_X_3_2GF_Weights.IMAGENET1K_V2)
    train(model, 8, predict=True, batch_size=32)
