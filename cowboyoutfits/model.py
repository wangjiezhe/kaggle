import lightning as L
import torch
import torchvision
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils import *

__all__ = [
    "Detector",
    "FasterRCNN",
]


class Detector(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.val_mAP = MeanAveragePrecision(backend="faster_coco_eval")

    def on_validation_epoch_end(self):
        self.val_mAP.reset()

    def training_step(self, batch):
        images, targets = batch
        # keys:
        # loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg
        loss_dict = self(images, targets)
        # 直接求和？
        losses = sum(loss for loss in loss_dict.values())
        self.log_dict(loss_dict, sync_dist=True)
        return losses

    def validation_step(self, batch):
        images, targets = batch
        outputs = self(images)
        self.val_mAP.update(outputs, targets)

    def on_validation_epoch_end(self):
        self.log_dict(
            {
                f"hp/{k}": v * 100 if v > 0 else v
                for k, v in self.val_mAP.compute().items()
                if k.startswith("ma")
            },
            sync_dist=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        self.hparams["optimizer"] = "AdamW"
        return optimizer
        # warmup_factor = 1.0 / 1000
        # warmup_iters = min(1000, len(self.datamodule.train_dataloader()) - 1)
        # lr_scheduler_config = {
        #     "lr_scheduler": torch.optim.lr_scheduler.StepLR(
        #         optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        #     ),
        #     "interval": "step",
        # }
        # return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


class FasterRCNN(Detector):
    def __init__(self, model: str, pretrained=True):
        super().__init__()
        self.save_hyperparameters()

        available_models = self.list_models()
        if model not in available_models:
            raise ValueError(f"{model} should be one of {available_models}.")
        weights = "DEFAULT" if pretrained else None

        self.net = torchvision.models.get_model(model, weights=weights)
        in_features = self.net.roi_heads.box_predictor.cls_score.in_features
        self.net.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    def forward(self, *args):
        return self.net(*args)

    @classmethod
    def list_models(cls):
        return torchvision.models.list_models(include="fasterrcnn*")
