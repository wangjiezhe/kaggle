import pandas as pd
from matplotlib import pyplot as plt
from torch import nn
from torchvision.transforms import v2

__all__ = [
    "NUM_CLASSES",
    "cutmix",
    "mixup",
    "cutmix_or_mixup",
    "init_cnn",
]

NUM_CLASSES = 120

cutmix = v2.CutMix(num_classes=NUM_CLASSES)
mixup = v2.MixUp(num_classes=NUM_CLASSES)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])


def init_cnn(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


## https://github.com/rasbt/comparing-automatic-augmentation-blog/blob/main/notebooks/helper_utilities.py
def plot_loss_and_acc(log_dir, loss_ylim=(0.0, 0.9), acc_ylim=(0.3, 1.0), save_loss=None, save_acc=None):
    metrics = pd.read_csv(f"{log_dir}/metrics.csv")

    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
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
