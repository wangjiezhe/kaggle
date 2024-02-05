import os.path

import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import tv_tensors
from torchvision.datasets import CocoDetection
from torchvision.ops import box_convert
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

__all__ = [
    "train_transform",
    "target_transform",
    "train_transforms",
    "CowboyDataset",
    "CowboyData",
]


class CowboyDataset(CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super().__init__(root, annFile, transform, target_transform, transforms)
        self.name_to_label = {info["name"]: info["id"] for info in self.coco.loadCats(self.coco.getCatIds())}
        self.label_to_name = {info["id"]: info["name"] for info in self.coco.loadCats(self.coco.getCatIds())}


train_transform = v2.Compose([v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True)])  # type: ignore


def target_transform(target):
    boxes = torch.as_tensor([ann["bbox"] for ann in target], dtype=torch.float32)  # type: ignore
    boxes = box_convert(boxes, in_fmt="xywh", out_fmt="xyxy")
    labels = torch.as_tensor([ann["category_id"] for ann in target])  # type: ignore
    image_id = target[0]["image_id"]
    area = torch.as_tensor([ann["area"] for ann in target], dtype=torch.float32)  # type: ignore
    iscrowd = torch.as_tensor([ann["iscrowd"] for ann in target], dtype=torch.uint8)  # type: ignore
    return {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}


def train_transforms(image, target):
    image = tv_tensors.Image(image)
    image = F.to_dtype(image, scale=True)  # type: ignore
    boxes = tv_tensors.BoundingBoxes(
        [ann["bbox"] for ann in target],
        format="xywh",
        canvas_size=F.get_size(image),  # type: ignore
    )
    boxes = F.convert_bounding_box_format(boxes, new_format=tv_tensors.BoundingBoxFormat.XYXY)
    labels = torch.as_tensor([ann["category_id"] for ann in target])  # type: ignore
    image_id = target[0]["image_id"]
    area = torch.as_tensor([ann["area"] for ann in target], dtype=torch.float32)  # type: ignore
    iscrowd = torch.as_tensor([ann["iscrowd"] for ann in target], dtype=torch.uint8)  # type: ignore
    target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}
    return image, target


class CowboyData(L.LightningDataModule):
    def __init__(
        self,
        batch_size=1,
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

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_dataset = CowboyDataset(
                root=self.train_images_dir,
                annFile=self.train_annFile,
                transforms=train_transforms,
            )
            if self.split:
                assert 0.0 < self.valid_ratio < 1.0
                self.train_data, self.val_data = random_split(
                    train_dataset,
                    [1.0 - self.valid_ratio, self.valid_ratio],
                    generator=torch.Generator().manual_seed(self.split_random_seed),  # type: ignore
                )
            else:
                self.train_data = train_dataset
                self.val_data = train_dataset

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
