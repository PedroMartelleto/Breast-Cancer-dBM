import torch
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
import glob
import os
import time
import copy
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Helper class that encapsulates training logic

class OptimContextFactory:
    # Defines loss, optimizer, scheduler...
    @staticmethod
    def create_optimizer(model, train_config):
        loss_fn = nn.CrossEntropyLoss()
        optimizer_ft = torch.optim.Adam(
            model.parameters(), lr=train_config.learning_rate, betas=train_config.betas, weight_decay=train_config.weight_decay)
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer_ft, step_size=train_config.step_size, gamma=train_config.gamma)
        return loss_fn, optimizer_ft, exp_lr_scheduler

class TrainHelper:
    def __init__(self, device, ds, experiment, train_config, model, loss_fn, optimizer, scheduler):
        self.device = device
        self.ds = ds
        self.model = model
        self.experiment = experiment
        self.train_config = train_config
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

    # One optimization/validation step that takes in batches from the data
    def optimizer_step(self, phase, inputs, labels):
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # Resets gradients
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(phase == 'train'):
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.loss_fn(outputs, labels)

            # Backprop only in the training stage
            if phase == 'train':
                loss.backward()
                self.optimizer.step()

        # Computes useful statistics
        self.running_loss += loss.item() * inputs.size(0)
        self.running_corrects += torch.sum(preds == labels.data)

    # The main function of this class. Call this to train self.model with the
    # specified optimizer, loss_fn, num_epochs and learning rate scheduler
    def train(self):
        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.train_config.num_epochs):
            print(f'Epoch {epoch+1}/{self.train_config.num_epochs}')
            print('-' * 10)

            for phase in ['train', 'val']:
                # Saves model's state at the beginning of each epoch to /netscratch/martelleto/
                torch.save(self.model.state_dict(), self.experiment.filepath(f"model_{phase}_{epoch}.pth"))

                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                self.running_loss = 0.0
                self.running_corrects = 0

                # Iterate over data
                for inputs, labels in self.ds.dataloaders[phase][self.train_config.cv_fold]:
                    self.optimizer_step(phase, inputs, labels)
                
                if phase == 'train':
                    # We only want to update the learning rate scheduler during training
                    self.scheduler.step()

                # Computes statistics
                epoch_loss = self.running_loss / self.ds.sizes[phase]
                epoch_acc = self.running_corrects.double() / self.ds.sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # The best weights are the ones if the biggest validation accuracy
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        # More statistics
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # Load the best weights before returning
        # self.model.load_state_dict(best_model_wts)

        return self.model