#!/usr/bin/python3

import pytorch_lightning as pl
import torch
from data import KaggleData
from model import DenseMLP, MLP
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision("high")


def train(
    model,
    epoch,
    logger_version=None,
    save_path=None,
    load_path=None,
    batch_size=256,
    test=True,
    predict=False,
    split=True,
):
    data = KaggleData(batch_size=batch_size, split=split)
    if load_path is not None:
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model.inited = True
    logger = TensorBoardLogger(".", default_hp_metric=False, version=logger_version)
    trainer = pl.Trainer(max_epochs=epoch, logger=logger, log_every_n_steps=5)
    trainer.fit(model, data)
    if test:
        trainer.test(model, data)
    if predict:
        trainer.predict(model, data)
    if save_path is not None:
        trainer.save_checkpoint(save_path)


if __name__ == "__main__":
    model = MLP(optim="Adam")
    train(model, 800, predict=True)
