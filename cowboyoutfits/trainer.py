import os

import lightning as L
import torch
from data import *
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from model import *
from utils import *

os.environ["HTTP_PROXY"] = "http://127.0.0.1:1084/"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:1084/"

torch.set_float32_matmul_precision("medium")

MODEL_NAME = "fasterrcnn_resnet50_fpn_alt_oversample"
EPOCH = 30
RANDOM_SEED = 42

# model = Cowboy_FasterRCNN(MODEL_NAME, pretrained=True, trainable_backbone_layers=0)
# model = Cowboy_FasterRCNN.load_from_checkpoint("./fasterrcnn_resnet50_fpn_epoch=50.ckpt")
# model = Cowboy_FasterRCNN_resnet("resnet50", pretrained=True)
# model = Cowboy_FasterRCNN_resnet.load_from_checkpoint(
#     "./fasterrcnn_resnet50_fpn_alt_oversample_epoch=30_fix.ckpt"
# )
model = Cowboy_FasterRCNN_resnet.load_from_checkpoint("fasterrcnn_resnet50_fpn_alt_epoch=30_fix.ckpt")

data = CowboyData(split_random_seed=RANDOM_SEED, train_batch_size=1)

tb_logger = TensorBoardLogger(".", default_hp_metric=False)
csv_logger = CSVLogger(".", name="csv_logs")
wandb_logger = WandbLogger(project="cowboy_logs")

trainer = L.Trainer(
    max_epochs=EPOCH,
    logger=[tb_logger, csv_logger, wandb_logger],
    precision="16-mixed",
    # accelerator="gpu",
    # devices=2,
    # strategy="ddp_notebook",
)

trainer.fit(model, data)

# trainer.save_checkpoint(f"{MODEL_NAME}_epoch={EPOCH}.ckpt")
trainer.save_checkpoint(f"{MODEL_NAME}_epoch={EPOCH+30}.ckpt")
