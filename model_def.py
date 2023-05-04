import argparse
import os
import time
from collections import OrderedDict
from typing import Any, Dict, Sequence, Union

import determined
import medmnist
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import wget
from determined.experimental import client
from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext
from medmnist import INFO, Evaluator
from tensorboardX import SummaryWriter
from torchvision.models import resnet18, resnet50
from tqdm import trange

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]
DATASET_ROOT = "datasets"


class MyMEDMnistTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context

        self.info = INFO[self.context.get_hparam("data_flag")]
        task = self.info["task"]
        n_classes = len(self.info["label"])

        self.context = context
        if self.context.get_hparam("model_flag") == "resnet18":
            model = resnet18(pretrained=False, num_classes=n_classes)
        elif self.context.get_hparam("model_flag") == "resnet50":
            model = resnet50(pretrained=False, num_classes=n_classes)
        else:
            raise NotImplementedError

        self.model = self.context.wrap_model(model)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.context.get_hparam("lr")
        )
        self.optimizer = self.context.wrap_optimizer(optimizer)
        
        if self.context.get_hparam("task") == "multi-label, binary-class":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        os.makedirs(DATASET_ROOT, exist_ok=True)
        wget.download(
            context.get_data_config()["url"],
            out=os.path.join(DATASET_ROOT, "pathmnist.npz"),
        )

    def build_training_data_loader(self) -> DataLoader:
        DataClass = getattr(medmnist, self.info["python_class"])

        if self.context.get_hparam("resize"):
            data_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            )
        else:
            data_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
            )

        train_dataset = DataClass(
            split="train",
            transform=data_transform,
            download=False,
            as_rgb=True,
            root=DATASET_ROOT,
        )
        train_loader = determined.pytorch.DataLoader(
            dataset=train_dataset,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=True,
        )

        return train_loader

    def build_validation_data_loader(self) -> DataLoader:
        DataClass = getattr(medmnist, self.info["python_class"])

        if self.context.get_hparam("resize"):
            data_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            )
        else:
            data_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
            )

        val_dataset = DataClass(
            split="val",
            transform=data_transform,
            download=False,
            as_rgb=True,
            root=DATASET_ROOT,
        )
        val_loader = determined.pytorch.DataLoader(
            dataset=val_dataset,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=False,
        )

        return val_loader

    def train_batch(
        self, batch: TorchData, epoch_idx: int, batch_idx: int
    ) -> Dict[str, Any]:
        inputs, targets = batch
        outputs = self.model(inputs)

        if self.context.get_hparam("task") == "multi-label, binary-class":
            targets = targets.to(torch.float32)
            loss = self.criterion(outputs, targets)
        else:
            targets = torch.squeeze(targets, 1).long()
            loss = self.criterion(outputs, targets)

        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)

        return {"loss": loss}

    def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
        inputs, targets = batch
        outputs = self.model(inputs)

        if self.context.get_hparam("task") == "multi-label, binary-class":
            targets = targets.to(torch.float32)
            loss = self.criterion(outputs, targets)
            m = nn.Sigmoid()
            outputs = m(outputs)
        else:
            targets = torch.squeeze(targets, 1).long()
            loss = self.criterion(outputs, targets)
            m = nn.Softmax(dim=1)
            outputs = m(outputs)
            targets = targets.float().resize_(len(targets), 1)

        return {"test_loss": loss}
