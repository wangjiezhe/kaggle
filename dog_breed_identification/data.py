import os.path
from math import ceil

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, default_collate, random_split
from torchvision import tv_tensors
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import v2

__all__ = ["TrainDataset", "PredictDataset", "DogData"]


class TrainDataset(VisionDataset):
    def __init__(self, root, csv, transform=None):
        super().__init__(root, transform=transform)
        df = pd.read_csv(csv)
        self.samples = [os.path.join(root, f"{image_id}.jpg") for image_id in df["id"].tolist()]
        targets = df["breed"].tolist()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(set(targets)))}
        self.targets = [self.class_to_idx[label] for label in targets]

    def __getitem__(self, index):
        path = self.samples[index]
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        target = self.targets[index]
        return sample, target

    def __len__(self):
        return len(self.samples)


class PredictDataset(VisionDataset):
    def __init__(self, root, csv, transform=None):
        super().__init__(root, transform=transform)
        df = pd.read_csv(csv)
        self.samples = [os.path.join(root, f"{image_id}.jpg") for image_id in df["id"].tolist()]

    def __getitem__(self, index):
        path = self.samples[index]
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.samples)


class BatchMultiCrop(v2.Transform):
    def forward(self, samples):
        return tv_tensors.Image(torch.stack(samples))


class DogData(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=128,
        num_workers=8,
        resize=224,
        aug=None,
        data_dir="./data",
        train_images_dirname="train",
        predict_images_dirname="test",
        split=True,
        split_random_seed=42,
        valid_ratio=0.1,
        five_crop=False,
        mix=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["aug", "mix"])
        self.data_dir = os.path.expanduser(data_dir)
        self.train_images_dir = os.path.join(data_dir, train_images_dirname)
        self.predict_images_dir = os.path.join(data_dir, predict_images_dirname)
        self.train_csv = os.path.join(data_dir, "labels.csv")
        self.predict_csv = os.path.join(data_dir, "sample_submission.csv")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split
        self.split_random_seed = split_random_seed
        self.mix = mix
        self.valid_ratio = valid_ratio
        self.five_crop = five_crop
        if aug is None:
            aug = v2.TrivialAugmentWide()
        self.trans_train = v2.Compose(
            [
                aug,
                v2.RandomResizedCrop(resize, scale=(0.64, 1), antialias=True),
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.trans_test = v2.Compose(
            [
                v2.Resize(resize),
                v2.CenterCrop(resize),
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.trans_predict = v2.Compose(
            [
                v2.Resize(ceil(resize / 0.8), antialias=True),
                v2.FiveCrop(resize),
                v2.PILToTensor(),
                BatchMultiCrop(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.label_map = {
            v: k
            for k, v in TrainDataset(
                self.train_images_dir, self.train_csv, self.trans_train
            ).class_to_idx.items()
        }

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_dataset = TrainDataset(self.train_images_dir, self.train_csv, self.trans_train)
            val_dataset = TrainDataset(self.train_images_dir, self.train_csv, self.trans_test)
            if self.split:
                assert 0.0 < self.valid_ratio < 1.0
                self.train_data, _ = random_split(
                    train_dataset,
                    [1.0 - self.valid_ratio, self.valid_ratio],
                    generator=torch.Generator().manual_seed(self.split_random_seed),
                )
                _, self.val_data = random_split(
                    val_dataset,
                    [1.0 - self.valid_ratio, self.valid_ratio],
                    generator=torch.Generator().manual_seed(self.split_random_seed),
                )
            else:
                self.train_data = train_dataset
                self.val_data = val_dataset
        if stage == "test":
            self.test_data = TrainDataset(self.train_images_dir, self.train_csv, self.trans_test)
        if stage == "predict":
            self.predict_data = PredictDataset(self.predict_images_dir, self.predict_csv, self.trans_test)
            self.predict_data_fivecrop = PredictDataset(
                self.predict_images_dir, self.predict_csv, self.trans_predict
            )

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
        d1 = DataLoader(
            self.predict_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        d2 = DataLoader(
            self.predict_data_fivecrop,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        if self.five_crop:
            return [d1, d2]
        else:
            return d1
