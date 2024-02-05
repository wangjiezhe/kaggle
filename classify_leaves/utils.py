from torch import nn
from torchvision.transforms import v2

__all__ = [
    "NUM_CLASSES",
    "cutmix",
    "mixup",
    "cutmix_or_mixup",
    "init_cnn",
]

NUM_CLASSES = 176

cutmix = v2.CutMix(num_classes=NUM_CLASSES)
mixup = v2.MixUp(num_classes=NUM_CLASSES)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])


def init_cnn(m):
    if type(m) is nn.Linear or type(m) is nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
