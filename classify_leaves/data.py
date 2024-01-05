import os.path

import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, default_collate, random_split, TensorDataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2


class LeavesData(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=64,
        num_workers=8,
        aug=None,
        data_dir="./data",
        train_images_dir="images_cleaned_train",
        split=True,
        normalize=True,
        mix=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["aug", "mix"])
        self.data_dir = data_dir
        self.train_images_dir = os.path.join(data_dir, train_images_dir)
        self.predict_csv = os.path.join(data_dir, "test.csv")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split
        self.mix = mix
        trans = v2.Compose([v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True)])
        self.trans_predict = (
            v2.Compose([trans, v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            if normalize
            else trans
        )
        self.trans_train = v2.Compose([aug, self.trans_predict]) if aug else self.trans_predict
        self.label_map = {
            v: k for k, v in ImageFolder(self.train_images_dir, self.trans_predict).class_to_idx.items()
        }

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_dataset = ImageFolder(self.train_images_dir, self.trans_train)
            if self.split:
                self.train_data, self.val_data = random_split(train_dataset, [0.9, 0.1])
            else:
                self.train_data = self.val_data = train_dataset
        if stage == "test":
            self.test_data = ImageFolder(self.train_images_dir, self.trans_predict)
        if stage == "predict":
            self.predict_images = pd.read_csv(self.predict_csv)
            self.predict_data = TensorDataset(
                torch.stack([self.image_to_tensor(f) for f in self.predict_images["image"].tolist()])  # type: ignore
            )

    def image_to_tensor(self, filename):
        with Image.open(os.path.join(self.data_dir, filename)) as img:
            img_tensor = self.trans_predict(img)
        return img_tensor

    def train_dataloader(self):
        if self.mix is not None:
            collate_fn = lambda batch: self.mix(*default_collate(batch))  # type: ignore
        else:
            collate_fn = None

        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
