import os.path
import random

import lightning as L
import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import tv_tensors
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2
from torchvision.transforms import v2
from utils import *

__all__ = [
    "CowboyData",
]


class CowboyData(L.LightningDataModule):
    def __init__(
        self,
        train_batch_size=2,
        inference_batch_size=1,
        num_workers=8,
        data_dir="./data",
        split=True,
        split_random_seed=None,
        valid_n_per_cat=10,  # randomly select 10 images for each category as valid_data
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_batch_size = train_batch_size
        self.inference_batch_size = inference_batch_size
        self.num_workers = num_workers
        self.data_dir = os.path.expanduser(data_dir)
        self.train_annFile = os.path.join(self.data_dir, "train.json")
        self.train_images_dir = os.path.join(self.data_dir, "images")
        self.val_csv = os.path.join(self.data_dir, "valid.csv")
        self.test_csv = os.path.join(self.data_dir, "test.csv")
        self.split = split
        self.split_random_seed = split_random_seed
        self.valid_n_per_cat = valid_n_per_cat

        coco = COCO(self.train_annFile)
        self.label_to_idx = {k: idx + 1 for idx, k in enumerate(coco.cats.keys())}
        self.label_to_name = {k: v["name"] for k, v in coco.cats.items()}
        self.idx_to_name = {idx + 1: v["name"] for idx, (_, v) in enumerate(coco.cats.items())}
        # self.category_weights = {k: 10000 / len(coco.getAnnIds(catIds=k)) for k in coco.cats.keys()}
        # self.category_weights = {87: 1600, 1034: 5, 131: 20, 318: 20, 588: 4.5}

        if self.split:
            random.seed(self.split_random_seed)
            self.validation_images_id = []
            for category in sorted(coco.cats.keys()):
                self.validation_images_id += random.sample(coco.catToImgs[category], self.valid_n_per_cat)
            if self.split_random_seed == 42:
                assert self.validation_images_id == VALIDATION_IMAGES_ID_42

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
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])  # type: ignore
        image = transform(image)
        target = self.target_transform(target)
        return image, target

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_dataset = CocoDetection(
                self.train_images_dir, self.train_annFile, transforms=self.train_transforms
            )
            if self.split:
                train_dataset.ids = [x for x in train_dataset.ids if x not in self.validation_images_id]
            self.train_data = wrap_dataset_for_transforms_v2(train_dataset)

            val_dataset = CocoDetection(
                self.train_images_dir, self.train_annFile, transforms=self.val_transforms
            )
            if self.split:
                val_dataset.ids = self.validation_images_id
            self.val_data = wrap_dataset_for_transforms_v2(val_dataset)

    def train_dataloader(self):
        # targets = [self.train_data._load_target(id) for id in self.train_data.ids]
        # sample_weights = [
        #     sum([self.category_weights[info["category_id"]] for info in target]) for target in targets
        # ]
        # sampler = WeightedRandomSampler(
        #     weights=sample_weights, num_samples=len(self.train_data), replacement=True
        # )
        return DataLoader(
            self.train_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            # sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: tuple(zip(*batch)),
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: tuple(zip(*batch)),
        )
