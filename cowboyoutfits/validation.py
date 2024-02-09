import lightning as L
import torch
from data import *
from model import *
from torchmetrics.detection import MeanAveragePrecision

torch.set_float32_matmul_precision("medium")

RANDOM_SEED = 42

model = Cowboy_FasterRCNN.load_from_checkpoint("./fasterrcnn_resnet50_fpn_epoch=50.ckpt")
data = CowboyData(split_random_seed=RANDOM_SEED)

trainer = L.Trainer(max_epochs=0, logger=False, precision="16-mixed")

data.setup("fit")

trainer.validate(model, data)
print(model.val_mAP.compute())
