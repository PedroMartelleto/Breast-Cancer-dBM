import torch
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import albumentations as A
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler, ConcatDataset, DataLoader
import globals

# Helper class for data augmentation
class AugDataset(datasets.ImageFolder):
    def __init__(self, root, aug, *args, **kwargs):
        super(AugDataset, self).__init__(root, *args, **kwargs)
        self.aug = aug

    def __getitem__(self, idx):
        image, label = super(AugDataset, self).__getitem__(idx)

        if self.aug:
            augmented = self.aug(image=np.array(image.convert('RGB')))
            image = augmented['image']
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            image = torch.tensor(image, dtype=torch.float)
        
        return image, label

#### DATA AUGMENTATION ####

# Defines a transform to run over the images loaded from the kaggle dataset
# Also computes the size of the training and validation set

class DatasetWrapper:
    def __init__(self, path, exp_config):
        transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=globals.NORM_MEAN, std=globals.NORM_STD) ])

        self.path = os.path.join(globals.BASE_PATH, path)
        self.dataset = AugDataset(self.path, aug=None, transform=transform)
        self.class_names = list(self.dataset.class_to_idx.keys())

        ds_size = len(self.dataset)
        self.fold_size = int(ds_size / 5)

        self.train_size = self.fold_size * 4
        self.val_size = self.fold_size

        splits = KFold(n_splits=5, shuffle=True, random_state=exp_config.seed)

        self.aug_dataset = AugDataset(self.path, aug=self.get_aug())
        self.train_loaders = []
        self.test_loaders = []

        for fold, (train_idx, val_idx) in enumerate(splits.split(self.dataset)):
            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(val_idx)
            self.train_loaders.append(DataLoader(self.aug_dataset, batch_size=exp_config.batch_size, sampler=train_sampler))
            self.test_loaders.append(DataLoader(self.dataset, batch_size=exp_config.batch_size, sampler=test_sampler))

        self.dataloaders = {'train': self.train_loaders, 'val': self.test_loaders}
        self.sizes = {'train': self.train_size, 'val': self.val_size}

        # We are not using the masks currently, so it is important to ensure no masks were left in the dataset
        for (sample_name, _) in self.dataset.samples:
            assert 'mask' not in os.path.basename(sample_name)
            # assert file ends with .png
            assert sample_name.endswith('.png')
    
    def get_aug(self):        
        aug = A.Compose([
                        A.Resize(256, 256),
                        A.ShiftScaleRotate(shift_limit=0.008, scale_limit=0.2, rotate_limit=30, p=0.7),
                        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, p=0.5),
                        A.Sharpen(alpha=(0.05, 0.1), lightness=(0.9, 1.1), p=0.5),
                        A.Blur(blur_limit=3, p=0.5),
                        A.PadIfNeeded(224, 224),
                        A.RandomCrop(width=224, height=224),
                        A.Normalize(mean=globals.NORM_MEAN, std=globals.NORM_STD)
                    ])
        return aug

    def preview_train(self):
        # Preview of the dataset

        def imshow(input, title=None):
            plt.figure(figsize=(10, 10))
            plt.imshow(np.clip(input.numpy().transpose((1, 2, 0)), 0, 1))
            if title is not None: plt.title(title)
            plt.pause(0.001)

        inputs, classes = next(iter(self.train_loader))
        out = torchvision.utils.make_grid(inputs[0:16])
        imshow(out)

    def preview_val(self):
        # Preview of the dataset

        def imshow(input, title=None):
            plt.figure(figsize=(10, 10))
            plt.imshow(np.clip(input.numpy().transpose((1, 2, 0)), 0, 1))
            if title is not None: plt.title(title)
            plt.pause(0.001)

        inputs, classes = next(iter(self.train_loader))
        out = torchvision.utils.make_grid(inputs[0:16])
        imshow(out)