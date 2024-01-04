from datetime import datetime

import pandas as pd
import torch
from pytorch_lightning.callbacks import BasePredictionWriter, Callback, ModelCheckpoint
from utils import NUM_CLASSES


class Submission(BasePredictionWriter):
    def __init__(self, now=True):
        super().__init__(write_interval="epoch")
        self.now = now

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        data = trainer.datamodule
        submission = pd.concat(
            [
                data.predict_images,
                pd.Series(
                    [data.label_map[i] for i in torch.stack(predictions).view(-1).tolist()],
                    name="label",
                ),
            ],
            axis=1,
        )
        if self.now:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"submission_{now}.csv"
        else:
            file_name = "submission.csv"
        submission.to_csv(file_name, index=False)


class ConfMat(Callback):
    def __init__(self):
        super().__init__()
        df_trans = pd.read_csv("plant_name.csv", encoding="utf-8")
        self.translation_map = {k: v for k, v in zip(df_trans.iloc[:, 0], df_trans.iloc[:, 1])}

    def on_test_epoch_end(self, trainer, pl_module):
        confmat = pl_module.confusion_matrix.compute()
        errors = confmat.clone()
        errors.fill_diagonal_(0)
        values, indices = errors.view(-1).topk(20)
        values, indices = values.tolist(), indices.tolist()
        real_indicies = [(i // NUM_CLASSES, i % NUM_CLASSES) for i in indices]

        label_map = trainer.datamodule.label_map

        df = pd.DataFrame(
            [
                [
                    t := label_map[row],
                    self.translation_map[t],
                    confmat[row, row].item(),
                    f := label_map[col],
                    self.translation_map[f],
                    value,
                ]
                for value, (row, col) in zip(values, real_indicies)
            ],
            columns=("True label", "准确值", "Corrects", "Predict label", "预测值", "Wrongs"),
        )
        print(df.to_markdown())


save_best_val_acc = ModelCheckpoint(
    filename="{epoch:02d}-{val_acc:.4f}",
    save_weights_only=True,
    mode="max",
    save_on_train_epoch_end=False,
    save_top_k=2,
)
save_best_val_loss = ModelCheckpoint(
    filename="{epoch:02d}-{val_loss:.4f}", save_weights_only=True, save_on_train_epoch_end=False
)
save_best_train_loss = ModelCheckpoint(filename="{epoch:02d}-{train_loss:.4f}", save_weights_only=True)

save_prediction = Submission(now=True)
analyse_confusion_matrix = ConfMat()
