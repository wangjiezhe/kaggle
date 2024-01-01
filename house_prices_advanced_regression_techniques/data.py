#!/usr/bin/env python3

import os.path

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset


class KaggleData(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=64,
        data_dir="./data",
        train_csv="train.csv",
        predict_csv="test.csv",
        num_workers=8,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_data_path = os.path.join(data_dir, train_csv)
        self.predict_data_path = os.path.join(data_dir, predict_csv)
        self.current_fold = 0
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # 读取数据
        train_data = pd.read_csv(self.train_data_path)
        predict_data = pd.read_csv(self.predict_data_path)
        self.predict_data_id = predict_data.iloc[:, 0]

        # 标准化数据 x <- (x - μ) / σ
        all_features = pd.concat((train_data.iloc[:, 1:-1], predict_data.iloc[:, 1:]))
        numeric_features = all_features.dtypes[all_features.dtypes != "object"].index
        all_features[numeric_features] = all_features[numeric_features].apply(
            lambda x: (x - x.mean()) / (x.std())
        )
        all_features[numeric_features] = all_features[numeric_features].fillna(0)

        # 对离散值使用 One-Hot Encoding
        all_features = pd.get_dummies(all_features, dummy_na=True, dtype=int)

        # 构造数据
        n_train = train_data.shape[0]
        if stage == "fit" or stage is None:
            train_features = all_features[:n_train].values
            train_labels = train_data.SalePrice.values
            dataset = TensorDataset(
                torch.tensor(train_features, dtype=torch.float32),
                torch.tensor(train_labels, dtype=torch.float32).reshape(-1, 1),
            )
            self.train_data, self.val_data = random_split(dataset, [0.9, 0.1])
        if stage == "test" or stage is None:
            train_features = all_features[:n_train].values
            train_labels = train_data.SalePrice.values
            self.test_data = TensorDataset(
                torch.tensor(train_features, dtype=torch.float32),
                torch.tensor(train_labels, dtype=torch.float32).reshape(-1, 1),
            )
        if stage == "predict" or stage is None:
            predict_features = all_features[n_train:].values
            self.predict_data = TensorDataset(torch.tensor(predict_features, dtype=torch.float32))

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_data,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )
