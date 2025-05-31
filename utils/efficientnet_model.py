# utils/efficientnet_model.py

import torch
import torch.nn as nn
from torchvision import models

def get_efficientnet_model(pretrained=True):
    """
    Loads EfficientNet (b0) and adapts it for binary classification (real vs deepfake).
    """
    # Load EfficientNet-B0 with pretrained ImageNet weights
    model = models.efficientnet_b0(pretrained=pretrained)

    # Replace the classification layer to output a single probability
    model.classifier[1] = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 1),
        nn.Sigmoid()
    )

    return model
