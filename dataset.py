import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from pathlib import Path
import prepare_data
import random

data_path =Path('/home/xingtong/ToolTrackingData')


class ColorizationDataset(Dataset):
    def __init__(self, file_names, to_augment=True, transform=None):
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):

        img_file_name = self.file_names[idx]
        img = load_image(img_file_name)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.expand_dims(gray, -1)
        gray = np.repeat(gray, 3, axis=2)

        if(self.to_augment == True):
            img, gray = self.transform(img, gray)

        return to_float_tensor(img), to_float_tensor(gray)

class RoboticsDataset(Dataset):
    def __init__(self, file_names, to_augment=True, transform=None, mode='train', problem_type=None):
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode
        self.problem_type = problem_type

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):

        img_file_name = self.file_names[idx]
        img = load_image(img_file_name)

        if(self.mode=='train'):
            rand_idx = random.randint(0, len(self.file_names) - 1)
            mask_file_name = self.file_names[rand_idx]
            mask = load_mask(mask_file_name, self.problem_type)
        elif(self.mode=='valid'):
            mask = load_mask(img_file_name, self.problem_type)

        if(self.to_augment == True):
            img, mask = self.transform(img, mask)

        if self.mode == 'train' or self.mode == 'valid':
            if self.problem_type == 'binary':
                return to_float_tensor(img), torch.from_numpy(np.expand_dims(mask, 0)).float()
            else:
                return to_float_tensor(img), torch.from_numpy(mask).long()
        else:
            return to_float_tensor(img), str(img_file_name)

class CADDataset(Dataset):
    def __init__(self, color_file_names, mask_file_names, transform=None):
        self.color_file_names = color_file_names
        self.mask_file_names = mask_file_names
        self.transform = transform
    def __len__(self):
        return len(self.color_file_names)

    def __getitem__(self, idx):
        color_file_name = self.color_file_names[idx]
        img = load_image(color_file_name)

        rand_idx = random.randint(0, len(self.mask_file_names) - 1)
        mask_file_name = self.mask_file_names[rand_idx]
        mask = load_mask_image(mask_file_name)
        img, mask = self.transform(img, mask)
        return to_float_tensor(img), torch.from_numpy(np.expand_dims(mask, 0)).float()

def to_float_tensor(img):
    return torch.from_numpy(np.moveaxis(img, -1, 0)).float()


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def load_mask_image(path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

def load_mask(path, problem_type):
    if problem_type == 'binary':
        mask_folder = 'binary_masks'
        factor = prepare_data.binary_factor
    elif problem_type == 'parts':
        mask_folder = 'parts_masks'
        factor = prepare_data.parts_factor
    elif problem_type == 'instruments':
        factor = prepare_data.instrument_factor
        mask_folder = 'instruments_masks'

    mask = cv2.imread(str(path).replace('images', mask_folder).replace('jpg', 'png'), 0)

    return (mask / factor).astype(np.uint8)

def get_file_names(root, prefix):
    path = Path(root)
    return list(path.glob(prefix + '*'))

def get_split(fold):
    folds = {0: [1, 3],
             1: [2, 5],
             2: [4, 8],
             3: [6, 7],
             4: [5, 8]}

    train_path = data_path / 'cropped_train'

    train_file_names = []
    val_file_names = []

    print(folds[fold])
    for instrument_id in range(1, 9):
        if instrument_id in folds[fold]:
            val_file_names += list((train_path / ('instrument_dataset_' + str(instrument_id)) / 'images').glob('*'))
        else:
            train_file_names += list((train_path / ('instrument_dataset_' + str(instrument_id)) / 'images').glob('*'))

    train_file_names.sort()
    val_file_names.sort()
    return train_file_names, val_file_names