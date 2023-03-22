import os
import argparse
import time
from tqdm import trange
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50
from tensorboardX import SummaryWriter
from collections import OrderedDict
from models import ResNet18, ResNet50
import determined
import medmnist
from medmnist import INFO, Evaluator

from determined.experimental import client

from typing import Any, Dict, Union, Sequence
from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]

class MyMEDMnistTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context

        info = INFO[self.context.get_hparam("data_flag")]
        task = info['task']
        n_channels = 3 if self.context.get_hparam("as_rgb") else info['n_channels']
        n_classes = len(info['label'])

        self.context = context
        resize = self.context.get_hparam("resize")
        if self.context.get_hparam("model_flag") == 'resnet18':
            model =  resnet18(pretrained=False, num_classes=n_classes) if resize else ResNet18(in_channels=n_channels, num_classes=n_classes)
        elif self.context.get_hparam("model_flag") == 'resnet50':
            model =  resnet50(pretrained=False, num_classes=n_classes) if resize else ResNet50(in_channels=n_channels, num_classes=n_classes)
        else:
            raise NotImplementedError

        self.model = self.context.wrap_model(model)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.context.get_hparam("lr"))
        self.optimizer = self.context.wrap_optimizer(optimizer)

        if self.context.get_hparam("task") == "multi-label, binary-class":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        
    def build_training_data_loader(self) -> DataLoader:

        info = INFO[self.context.get_hparam("data_flag")]

        DataClass = getattr(medmnist, info['python_class'])        
        resize = self.context.get_hparam("resize")

        if resize:
            data_transform = transforms.Compose(
                [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST), 
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5])])
        else:
            data_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5])])
        
        train_dataset = DataClass(split='train', transform=data_transform, download=True, as_rgb=self.context.get_hparam("as_rgb"))
        val_dataset = DataClass(split='val', transform=data_transform, download=True, as_rgb=self.context.get_hparam("as_rgb"))
        test_dataset = DataClass(split='test', transform=data_transform, download=True, as_rgb=self.context.get_hparam("as_rgb"))
        
        train_loader = determined.pytorch.DataLoader(dataset=train_dataset,
                                    batch_size=self.context.get_per_slot_batch_size(),
                                    shuffle=True)

        return train_loader

    def build_validation_data_loader(self) -> DataLoader:

        info = INFO[self.context.get_hparam("data_flag")]

        DataClass = getattr(medmnist, info['python_class'])        
        resize = self.context.get_hparam("resize")

        if resize:
            data_transform = transforms.Compose(
                [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST), 
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5])])
        else:
            data_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5])])
        
        train_dataset = DataClass(split='train', transform=data_transform, download=True, as_rgb=self.context.get_hparam("as_rgb"))
        val_dataset = DataClass(split='val', transform=data_transform, download=True, as_rgb=self.context.get_hparam("as_rgb"))
        test_dataset = DataClass(split='test', transform=data_transform, download=True, as_rgb=self.context.get_hparam("as_rgb"))

        val_loader = determined.pytorch.DataLoader(dataset=val_dataset,
                                    batch_size=self.context.get_per_slot_batch_size(),
                                    shuffle=False)

        return val_loader

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int)  -> Dict[str, Any]:
        inputs, targets = batch
        outputs = self.model(inputs)

        if self.context.get_hparam("task") == 'multi-label, binary-class':
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
        
        if self.context.get_hparam("task") == 'multi-label, binary-class':
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
