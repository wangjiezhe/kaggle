import lightning as L
import torch
from data import *
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from model import *
from utils import *

torch.set_float32_matmul_precision("medium")

MODEL_NAME = "fasterrcnn_resnet50_fpn_alt"
EPOCH = 30
RANDOM_SEED = 42

# model = Cowboy_FasterRCNN(MODEL_NAME, pretrained=True, trainable_backbone_layers=0)
# model = Cowboy_FasterRCNN.load_from_checkpoint("./fasterrcnn_resnet50_fpn_epoch=50.ckpt")
model = Cowboy_FasterRCNN_resnet("resnet50", pretrained=True)

data = CowboyData(split_random_seed=RANDOM_SEED)

tb_logger = TensorBoardLogger(".", default_hp_metric=False)
csv_logger = CSVLogger(".", name="csv_logs")

trainer = L.Trainer(
    max_epochs=EPOCH,
    logger=[tb_logger, csv_logger],
    precision="16-mixed",
    # accelerator="gpu",
    # devices=2,
    # strategy="ddp_notebook",
    log_every_n_steps=10,
)

trainer.fit(model, data)

trainer.save_checkpoint(f"{MODEL_NAME}_epoch={EPOCH}.ckpt")
# trainer.save_checkpoint(f"{MODEL_NAME}_epoch={EPOCH+50}.ckpt")
