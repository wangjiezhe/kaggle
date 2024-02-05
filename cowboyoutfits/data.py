import os.path

import lightning as L
import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, random_split
from torchvision import tv_tensors
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2
from torchvision.ops import box_convert
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

__all__ = [
    "CowboyData",
]


class CowboyData(L.LightningDataModule):
    def __init__(
        self,
        batch_size=4,
        num_workers=8,
        data_dir="./data",
        split=True,
        split_random_seed=42,
        valid_ratio=0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = os.path.expanduser(data_dir)
        self.train_annFile = os.path.join(self.data_dir, "train.json")
        self.train_images_dir = os.path.join(self.data_dir, "images")
        self.val_csv = os.path.join(self.data_dir, "valid.csv")
        self.test_csv = os.path.join(self.data_dir, "test.csv")
        self.split = split
        self.split_random_seed = split_random_seed
        self.valid_ratio = valid_ratio
        coco = COCO(self.train_annFile)
        self.label_to_idx = {k: idx + 1 for idx, k in enumerate(coco.cats.keys())}
        self.label_to_name = {k: v["name"] for k, v in coco.cats.items()}
        self.idx_to_name = {idx + 1: v["name"] for idx, (k, v) in enumerate(coco.cats.items())}

    def target_transform(self, target):
        target["labels"].apply_(lambda x: self.label_to_idx[x])
        return target

    def train_transforms(self, image, target):
        transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.RandomPhotometricDistort(),
                v2.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0}),
                v2.RandomIoUCrop(),
                v2.RandomHorizontalFlip(),
                v2.SanitizeBoundingBoxes(),
                v2.ToDtype(torch.float32, scale=True),  # type: ignore
            ]
        )
        target = self.target_transform(target)

        return transforms(image, target)

    def val_transforms(self, image, target):
        transform = v2.Compose([v2.ToImage, v2.ToDtype(torch.float32, scale=True)])  # type: ignore
        image = transform(image)
        target = self.target_transform(target)
        return image, target

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_dataset = CocoDetection(
                self.train_images_dir, self.train_annFile, transforms=self.train_transforms
            )
            train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
            val_dataset = CocoDetection(
                self.train_images_dir, self.train_annFile, transforms=self.val_transforms
            )
            val_dataset = wrap_dataset_for_transforms_v2(val_dataset)
            if self.split:
                assert 0.0 < self.valid_ratio < 1.0
                self.train_data, _ = random_split(
                    train_dataset,
                    [1.0 - self.valid_ratio, self.valid_ratio],
                    generator=torch.Generator().manual_seed(self.split_random_seed),  # type: ignore
                )
                _, self.val_data = random_split(
                    val_dataset,
                    [1.0 - self.valid_ratio, self.valid_ratio],
                    generator=torch.Generator().manual_seed(self.split_random_seed),  # type: ignore
                )
            else:
                self.train_data = train_dataset
                self.val_data = val_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: tuple(zip(*batch)),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: tuple(zip(*batch)),
        )
