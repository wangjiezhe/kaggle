import os

import lightning as L
import torch
from data import *
from lightning.pytorch.loggers import WandbLogger
from model import *

os.environ["HTTP_PROXY"] = "http://127.0.0.1:1084/"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:1084/"

torch.set_float32_matmul_precision("medium")

RANDOM_SEED = 42

model = Cowboy_FasterRCNN.load_from_checkpoint("./fasterrcnn_resnet50_fpn_epoch=50.ckpt")
data = CowboyData(split_random_seed=RANDOM_SEED)
wandb_logger = WandbLogger(project="cowboy_logs")

trainer = L.Trainer(max_epochs=0, logger=wandb_logger, precision="16-mixed")

data.setup("fit")

trainer.validate(model, data)
print(model.val_mAP.compute())
