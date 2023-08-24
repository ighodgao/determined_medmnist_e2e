import os

import medmnist
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from dotenv import load_dotenv
from medmnist import INFO
from PIL import Image

load_dotenv()

# global variables
data_flag = os.getenv("DATA_FLAG")
info = INFO[data_flag]
download = True
root = "datasets"
BATCH_SIZE = 128

DataClass = getattr(medmnist, info["python_class"])

# preprocessing
data_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
)

# load the data
os.makedirs(root, exist_ok=True)
train_dataset = DataClass(
    root=root, split="train", transform=data_transform, download=download
)
test_dataset = DataClass(
    root=root, split="test", transform=data_transform, download=download
)

# encapsulate data into dataloader form
test_loader = data.DataLoader(
    dataset=test_dataset, batch_size=2 * BATCH_SIZE, shuffle=False
)

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
    gt_label = [info["label"][str(i)] for i in targets[0].numpy()]

    # save the image to disk with its label name
    img.save(f"sample_images/{gt_label[0]}_{batch_idx}.jpg")
