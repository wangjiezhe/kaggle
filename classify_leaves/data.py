import os.path

import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2


class LeavesData(pl.LightningDataModule):
    def __init__(
        self, batch_size=64, num_workers=8, aug=None, train_images_dir="./data/images_train", split=True
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_images_dir = train_images_dir
        self.predict_images_dir = "./data"
        self.predict_csv = "./data/test.csv"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split
        self.trans_predict = v2.Compose([v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True)])
        if aug:
            self.trans_train = v2.Compose([aug, self.trans_predict])
        else:
            self.trans_train = self.trans_predict

    def setup(self, stage=None):
        dataset = ImageFolder(self.train_images_dir, self.trans_train)
        self.label_map = {v: k for k, v in dataset.class_to_idx.items()}
        if stage == "fit" or stage is None:
            if self.split:
                self.train_data, self.val_data = random_split(dataset, [0.9, 0.1])
            else:
                self.train_data = dataset
                _, self.val_data = random_split(dataset, [0.9, 0.1])
        if stage == "test" or stage is None:
            self.test_data = dataset
        if stage == "predict" or stage is None:
            self.predict_images = pd.read_csv(self.predict_csv)
            self.predict_data = TensorDataset(
                torch.stack([self.image_to_tensor(f) for f in self.predict_images["image"].tolist()])
            )

    def image_to_tensor(self, filename):
        with Image.open(os.path.join(self.predict_images_dir, filename)) as img:
            img_tensor = self.trans_predict(img)
        return img_tensor

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
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
