import torch
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import confusion_matrix
import seaborn as sn
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
from ray import tune

# Helper class that encapsulates training logic

profiler_start = None
profiler_name = None

def profile(name=None):
    return

    global profiler_start, profiler_name
    
    if profiler_name is not None:
        elapsed_ms = (time.time() - profiler_start) * 1000
        print(f'Profiler: {profiler_name} took {elapsed_ms:.2f} ms')

    profiler_name = None
    profiler_start = None

    if name is not None:
        profiler_name = name
        profiler_start = time.time()

class OptimContextFactory:
    # Defines loss, optimizer, scheduler...
    @staticmethod
    def create_optimizer(model, exp_config):
        loss_fn = nn.CrossEntropyLoss()
        optimizer_ft = torch.optim.Adam(
            model.parameters(), lr=exp_config.learning_rate,
            betas=exp_config.betas, weight_decay=exp_config.weight_decay)
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer_ft, step_size=exp_config.step_size, gamma=exp_config.gamma)
        return loss_fn, optimizer_ft, exp_lr_scheduler

class TrainHelper:
    def __init__(self, fold, device, ds, exp_config, model, optim_context):
        self.fold = fold
        self.device = device
        self.ds = ds
        self.model = model
        self.exp_config = exp_config
        self.loss_fn = optim_context[0]
        self.optimizer = optim_context[1]
        self.scheduler = optim_context[2]

    # One optimization/validation step that takes in batches from the data
    def optimizer_step(self, phase, inputs, labels):
        profile("Send to device")
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        profile("Zero grad")
        # Resets gradients
        self.optimizer.zero_grad()
        profile()
        with torch.set_grad_enabled(phase == 'train'):
            profile("Forward")
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            profile("Loss")
            loss = self.loss_fn(outputs, labels)
            profile()

            # Backprop only in the training stage
            if phase == 'train':
                profile("Backward")
                loss.backward()
                profile("Step")
                self.optimizer.step()
            
            profile()

        # Computes useful statistics
        self.running_loss += loss.item() * inputs.size(0)
        self.running_corrects += torch.sum(preds == labels.data)

    def save_confusion_matrix(self):
        y_true = []
        y_pred = []

        for inputs, labels in self.ds.dataloaders['val'][self.exp_config.cv_fold]:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(list(labels.cpu().numpy()))
            y_pred.extend(list(preds.cpu().numpy()))
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        cf_matrix = confusion_matrix(y_true, y_pred, labels=self.ds.class_names, normalize='true')
        df_cm = pd.DataFrame(cf_matrix, index = [i for i in self.ds.class_names],
                            columns = [i for i in self.ds.class_names])
        plt.figure(figsize = (7,7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig(self.exp_config.filepath('conf_matrix_cv_' + str(self.fold) + '.png'))

    # The main function of this class. Call this to train self.model with the
    # specified optimizer, loss_fn, num_epochs and learning rate scheduler
    def train_and_validate(self):
        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        best_loss = 0.0

        for epoch in range(self.exp_config.num_epochs):
            print(f'Epoch {epoch+1}/{self.exp_config.num_epochs}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if (epoch+1) % 10 == 0:
                    profile("Checkpoint save")
                    # Saves model's state at the beginning of each epoch to /netscratch/martelleto/...
                    torch.save(self.model.state_dict(), self.exp_config.filepath(f"model_{phase}_{epoch}.pth"))
                    profile()

                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                self.running_loss = 0.0
                self.running_corrects = 0

                # Iterate over data
                for inputs, labels in self.ds.dataloaders[phase][self.exp_config.cv_fold]:
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
                    profile("Update best model weights")
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    profile()

            print()

        # More statistics
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val acc: {best_acc:4f}')
    
        # Save the best weights before returning
        self.model.load_state_dict(best_model_wts)
        self.model.eval()
        torch.save(self.model.state_dict(), self.exp_config.filepath(f"best_model.pth"))

        tune.report(loss=best_loss, accuracy=best_acc)

        return self.model