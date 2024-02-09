import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

__all__ = [
    "NUM_CLASSES",
    "init_cnn",
    "plot",
    "plot_loss_and_acc",
    "VALIDATION_IMAGES_ID_42",
]


NUM_CLASSES = 6


def init_cnn(m):
    if type(m) is nn.Linear or type(m) is nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


# https://github.com/rasbt/comparing-automatic-augmentation-blog/blob/main/notebooks/helper_utilities.py
def plot_loss_and_acc(log_dir, loss_ylim=(0.0, 0.9), acc_ylim=(0.3, 1.0), save_loss=None, save_acc=None):
    metrics = pd.read_csv(f"{log_dir}/metrics.csv")

    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i  # type: ignore
        aggreg_metrics.append(agg)

    df_metrics = pd.DataFrame(aggreg_metrics)
    df_metrics[["train_loss", "val_loss"]].plot(grid=True, legend=True, xlabel="Epoch", ylabel="Loss")

    plt.ylim(loss_ylim)
    if save_loss is not None:
        plt.savefig(save_loss)

    df_metrics[["train_acc", "val_acc"]].plot(grid=True, legend=True, xlabel="Epoch", ylabel="ACC")

    plt.ylim(acc_ylim)
    if save_acc is not None:
        plt.savefig(save_acc)


# https://github.com/pytorch/vision/blob/v0.17.0/gallery/transforms/helpers.py
def plot(imgs, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)  # type: ignore
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(
                    img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=0.65  # type: ignore
                )

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()


VALIDATION_IMAGES_ID_42 = [
    12389486278301481112,
    12063210065453926035,
    275081509518152865,
    9046773008794662425,
    9046773008794662425,
    6454634400439108759,
    2442486155143431682,
    16339904583227417633,
    3950715469133251220,
    14154706182445838081,
    14210680104795534067,
    14210680104795534067,
    3542243747386200876,
    1148681491280851830,
    55765707822788877,
    188684046137016491,
    9472435701143540480,
    1458774397483234228,
    702451770036056859,
    993453226575663877,
    2435954503637012821,
    10474442222049809275,
    13071488121744982452,
    14693519167164264804,
    9503268088049120024,
    8869885324579176806,
    17996576784191853207,
    13334572087714108074,
    2704366883452039217,
    17996576784191853207,
    5397884438644359282,
    10961498568206976066,
    6995932726905766043,
    17358335937359154405,
    11836437202926667642,
    1626138972757263702,
    4705627804717320902,
    10513418629012326047,
    4575816280787421233,
    16636246187966138834,
    6487504139028333667,
    1781078496580242076,
    13435039019669356837,
    5076910691331049625,
    15618392129589020879,
    8990877779853150471,
    4132048462086361616,
    12620824811464491657,
    5687035767527705617,
    10322816502648832885,
]
