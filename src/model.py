from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch

class ModelFactory:
    @staticmethod
    def create_model(device, num_classes=2):
        # Actually defines the model for transfer learning
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        model = model.to(device)
        return model