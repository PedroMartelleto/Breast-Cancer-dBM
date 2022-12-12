from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch
import globals
import os

class ModelFactory:
    @staticmethod
    def create_model(device, num_classes=2):
        # Actually defines the model for transfer learning
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)
        return model
    
    @staticmethod
    def create_model_from_checkpoint(exp_name, device, num_classes=2):
        # Loads a model from a checkpoint
        model = ModelFactory.create_model(device, num_classes)
        model.load_state_dict(torch.load(os.path.join(globals.BASE_PATH, "experiments", exp_name, "best_model.pth")))
        return model