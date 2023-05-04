from determined.experimental import client
from determined import pytorch
import torch
import torch.nn as nn

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator
import torch
from PIL import Image

# global variables
data_flag = 'pathmnist'
download = True

NUM_EPOCHS = 3
BATCH_SIZE = 128

info = INFO[data_flag]
task = info['task']

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

# download checkpoint from Determined experiment
checkpoint = client.get_experiment("641").top_checkpoint()
path = checkpoint.download()
trial = pytorch.load_trial_from_checkpoint_path(path)
model = trial.model
model.eval()

# list label names
label_names = ["adipose", "background", "debris", "lymphocytes", "mucus","smooth muscle", "normal colon mucosa", "cancer-associated stroma", "colorectal adenocarcinoma epithelium"]

# save first image of each batch (for label variety) 
for batch_idx, (inputs, targets) in enumerate(test_loader):

    # inputs is a tensor of shape (256, 3, 28, 28)
    img_array = inputs.permute(0, 2, 3, 1).numpy()  # shape (256, 28, 28, 3)
    img_array = img_array * 0.5 + 0.5  # reverse the normalization
    img_array = np.clip(img_array, 0.0, 1.0)  # clip the values to the range [0, 1]
    img_array = (img_array * 255).astype(np.uint8)  # convert to uint8

    # convert the first image of the batch to a PIL Image
    img = Image.fromarray(img_array[0]) 

    # get the image label
    gt_label = [label_names[i] for i in targets[0].numpy()]

    # save the image to disk with its label name
    img.save(f'sample_images/{gt_label[0]}_{batch_idx}.jpg')
