from determined.experimental import client
from determined import pytorch
import torch
import torch.nn as nn

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator


data_flag = 'pathmnist'
# data_flag = 'breastmnist'
download = True

NUM_EPOCHS = 3
BATCH_SIZE = 128
lr = 0.001

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

# preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# load the data
train_dataset = DataClass(split='train', transform=data_transform, download=download)
test_dataset = DataClass(split='test', transform=data_transform, download=download)

pil_dataset = DataClass(split='train', download=download)

# encapsulate data into dataloader form
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)


checkpoint = client.get_experiment("641").top_checkpoint()
path = checkpoint.download()
trial = pytorch.load_trial_from_checkpoint_path(path)
model = trial.model

model.eval()

label_names = ["adipose", "background", "debris", "lymphocytes", "mucus","smooth muscle", "normal colon mucosa", "cancer-associated stroma", "colorectal adenocarcinoma epithelium"]


img_idx = 1
for batch_idx, (inputs, targets) in enumerate(test_loader):
    print(img_idx)
    img_idx +=1

    predictions = model(inputs[0].unsqueeze(0))
    gt_labels = [label_names[i] for i in targets[0].numpy()]

    # predictions = torch.squeeze(predictions, 1).long()
    m = nn.Softmax(dim=1)
    predictions = m(predictions)

    # Get the index of the maximum value in each row
    predicted_labels = torch.max(predictions, dim=1)[1]

    # Convert the index to label names using the label_names list
    predicted_labels = [label_names[i] for i in predicted_labels]

    print('Ground truth labels:', gt_labels)
    print('Predicted labels:', predicted_labels)
