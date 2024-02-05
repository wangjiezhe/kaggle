#!/usr/bin/python3

import pytorch_lightning as pl
import torch
from data import KaggleData
from model import MLP
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision("high")


def train(
    net,
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
        net.load_state_dict(checkpoint["state_dict"], strict=False)
        net.inited = True
    logger = TensorBoardLogger(".", default_hp_metric=False, version=logger_version)
    trainer = pl.Trainer(max_epochs=epoch, logger=logger, log_every_n_steps=5)
    trainer.fit(net, data)
    if test:
        trainer.test(net, data)
    if predict:
        trainer.predict(net, data)
    if save_path is not None:
        trainer.save_checkpoint(save_path)


if __name__ == "__main__":
    model = MLP(optimizer="Adam")
    run_round = 8
    for i in range(run_round):
        train(model, 100, predict=(i >= run_round - 1))
