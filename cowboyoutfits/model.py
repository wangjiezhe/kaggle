import lightning as L
import torch
import torchvision
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils import *

__all__ = [
    "Detector",
    "Cowboy_FasterRCNN",
    "Cowboy_FasterRCNN_resnet",
]


class Detector(L.LightningModule):
    def __init__(self):
        super().__init__()
        # backend='faster_coco_eval' with `class_metrics=True` will cause AssertionError
        self.val_mAP = MeanAveragePrecision(class_metrics=True)

    def on_validation_epoch_start(self):
        self.val_mAP.reset()

    def training_step(self, batch):
        images, targets = batch
        # keys:
        # loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg
        loss_dict = self(images, targets)
        # 直接求和？
        losses = sum(loss for loss in loss_dict.values())
        self.log_dict(
            loss_dict,
            # sync_dist=True,
        )
        return losses

    def validation_step(self, batch):
        images, targets = batch
        outputs = self(images)
        self.val_mAP.update(outputs, targets)

    def on_validation_epoch_end(self):
        res = self.val_mAP.compute()
        # https://github.com/Lightning-AI/pytorch-lightning/issues/18803#issuecomment-1920342156
        # res = {k: v.to('cuda') for k, v in res.items()}
        metrics = {
            "hp/map": res["map"] * 100,
            "hp/map_50": res["map_50"] * 100,
            "hp/map_75": res["map_75"] * 100,
            "hp/map_small": res["map_small"] * 100,
            "hp/map_medium": res["map_medium"] * 100,
            "hp/map_large": res["map_large"] * 100,
            "hp/mar_1": res["mar_1"] * 100,
            "hp/mar_10": res["mar_10"] * 100,
            "hp/mar_100": res["mar_100"] * 100,
            "hp/mar_small": res["mar_small"] * 100,
            "hp/mar_medium": res["mar_medium"] * 100,
            "hp/mar_large": res["mar_large"] * 100,
        }
        for idx, map_per_cls, mar_100_per_cls in zip(
            res["classes"].tolist(), res["map_per_class"], res["mar_100_per_class"]
        ):
            name = self.trainer.datamodule.idx_to_name[idx]
            metrics[f"hp/map({name})"] = map_per_cls * 100
            metrics[f"hp/mar_100({name})"] = mar_100_per_cls * 100
        self.log_dict(
            metrics,
            # sync_dist=True,
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


class Cowboy_FasterRCNN(Detector):
    def __init__(self, model: str, pretrained=True, trainable_backbone_layers=None):
        super().__init__()
        self.save_hyperparameters()

        available_models = self.list_models()
        if model not in available_models:
            raise ValueError(f"{model} should be one of {available_models}.")
        weights = "DEFAULT" if pretrained else None

        self.net = torchvision.models.get_model(
            model, weights=weights, trainable_backbone_layers=trainable_backbone_layers
        )
        in_features = self.net.roi_heads.box_predictor.cls_score.in_features
        self.net.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    def forward(self, *args):
        return self.net(*args)

    @classmethod
    def list_models(cls):
        return torchvision.models.list_models(include="fasterrcnn*")


class Cowboy_FasterRCNN_resnet(FasterRCNN, Detector):
    def __init__(self, backbone: str, pretrained=True, trainable_backbone_layers=3):
        # self.save_hyperparameters()
        available_backbones = self.list_models()
        if backbone not in available_backbones:
            raise ValueError(f"{backbone} should be one of {available_backbones}.")
        weights = "DEFAULT" if pretrained else None
        backbone = resnet_fpn_backbone(
            backbone_name=backbone, weights=weights, trainable_layers=trainable_backbone_layers
        )
        super().__init__(backbone=backbone, num_classes=NUM_CLASSES)

    @classmethod
    def list_models(cls):
        return torchvision.models.list_models(
            include="resne*",
        ) + torchvision.models.list_models(
            include="wide_resne*",
        )
