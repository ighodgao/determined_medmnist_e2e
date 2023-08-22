import os
from typing import Any, Dict, Sequence, Union

import determined
import medmnist
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import wget
from determined.pytorch import (
    DataLoader,
    LRScheduler,
    PyTorchTrial,
    PyTorchTrialContext,
)
from medmnist import INFO

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]
DATASET_ROOT = "datasets"


class MyMEDMnistTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context

        self.info = INFO[self.context.get_hparam("data_flag")]
        n_channels = self.info["n_channels"]
        n_classes = len(self.info["label"])
        task = self.info["task"]

        model = Net(n_channels, n_classes)
        self.model = self.context.wrap_model(model)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.context.get_hparam("lr"),
            weight_decay=self.context.get_hparam("weight_decay"),
        )
        self.optimizer = self.context.wrap_optimizer(optimizer)

        if task == "multi-label, binary-class":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        num_epochs = self.context.get_experiment_config()["searcher"]["max_length"][
            "epochs"
        ]
        milestones = [0.5 * num_epochs, 0.75 * num_epochs]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=milestones,
            gamma=self.context.get_hparam("gamma"),
        )
        self.lr_sch = self.context.wrap_lr_scheduler(
            scheduler, step_mode=LRScheduler.StepMode.STEP_EVERY_EPOCH
        )

        os.makedirs(DATASET_ROOT, exist_ok=True)

        data_url = context.get_data_config().get("url")
        if data_url:
            wget.download(
                data_url,
                out=os.path.join(
                    DATASET_ROOT, f"{self.context.get_hparam('data_flag')}.npz"
                ),
            )

    def build_training_data_loader(self) -> DataLoader:
        DataClass = getattr(medmnist, self.info["python_class"])
        data_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
        )

        train_dataset = DataClass(
            split="train",
            transform=data_transform,
            download=True,
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
        data_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
        )

        val_dataset = DataClass(
            split="val",
            transform=data_transform,
            download=True,
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


# from https://github.com/MedMNIST/MedMNIST/blob/main/examples/getting_started.ipynb
class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3), nn.BatchNorm2d(16), nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
