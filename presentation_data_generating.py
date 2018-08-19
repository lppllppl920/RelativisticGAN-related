import models

import random
import os
import cv2
import numpy as np
import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD

from pathlib import Path
from torchsummary import summary

from dataset import ColorizationDataset
from dataset import get_split
from transforms import (DualCompose,
                        Resize,
                        ImageOnly,
                        HorizontalFlip,
                        VerticalFlip,
                        ColorizationNormalize )
import utils


img_size = 300
valid_transform = DualCompose([
    Resize(size=img_size)
])

fold = 0
train_file_names, val_file_names = get_split(fold=fold)

batch_size = 6
num_workers = 4
valid_loader = DataLoader(dataset=ColorizationDataset(file_names=val_file_names, transform=valid_transform, to_augment=True),
            shuffle=False,
            num_workers=num_workers,
            batch_size=batch_size)

root = Path('/home/xingtong/ToolTrackingData/runs/materials_generation')
try:
    root.mkdir()
except OSError:
    print("directory exists!")

for i, (color_inputs, gray_inputs) in enumerate(valid_loader):

    gray = gray_inputs.data.cpu().numpy()[0]
    color = color_inputs.data.cpu().numpy()[0]

    gray = np.moveaxis(gray, source=[0, 1, 2], destination=[2, 0, 1])
    color = np.moveaxis(color, source=[0, 1, 2], destination=[2, 0, 1])

    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    cv2.imwrite(str(root / 'gray_{}.png').format(i), gray)
    cv2.imwrite(str(root / 'color_{}.png').format(i), color)
    cv2.waitKey()
