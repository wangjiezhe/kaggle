import os.path

import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, default_collate, random_split
from torchvision.datasets import ImageFolder, VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import v2


class PredictDataset(VisionDataset):
    def __init__(self, root, csv, transform=None):
        super().__init__(root, transform=transform)
        df = pd.read_csv(csv)
        self.samples = [os.path.join(root, f"{image_id}.png") for image_id in df["id"].tolist()]

    def __getitem__(self, index):
        path = self.samples[index]
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.samples)


class Cifar10Data(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=128,
        num_workers=8,
        aug=None,
        data_dir="./data",
        train_images_dirname="train",
        predict_images_dirname="test",
        split=True,
        valid_ratio=0.1,
        mix=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["aug", "mix"])
        self.data_dir = os.path.expanduser(data_dir)
        self.train_images_dir = os.path.join(data_dir, train_images_dirname)
        self.predict_images_dir = os.path.join(data_dir, predict_images_dirname)
        self.predict_csv = os.path.join(data_dir, "sampleSubmission.csv")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split
        self.mix = mix
        self.valid_ratio = valid_ratio
        self.trans_predict = v2.Compose(
            [
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        if aug is None:
            aug = v2.Compose(
                [
                    v2.Resize(40),
                    v2.RandomResizedCrop(32, scale=(0.64, 1), ratio=(1.0, 1.0)),
                    v2.RandomHorizontalFlip(),
                ]
            )
        self.trans_train = v2.Compose([aug, self.trans_predict]) if aug else self.trans_predict
        self.label_map = {
            v: k for k, v in ImageFolder(self.train_images_dir, self.trans_predict).class_to_idx.items()
        }

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_dataset = ImageFolder(self.train_images_dir, self.trans_train)
            if self.split:
                assert 0.0 < self.valid_ratio < 1.0
                self.train_data, self.val_data = random_split(
                    train_dataset, [1.0 - self.valid_ratio, self.valid_ratio]
                )
            else:
                self.train_data = self.val_data = train_dataset
        if stage == "test":
            self.test_data = ImageFolder(self.train_images_dir, self.trans_predict)
        if stage == "predict":
            self.predict_data = PredictDataset(self.predict_images_dir, self.predict_csv, self.trans_predict)

    def train_dataloader(self):
        if self.mix is not None:
            collate_fn = lambda batch: self.mix(*default_collate(batch))
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
