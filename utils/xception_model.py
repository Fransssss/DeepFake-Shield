# utils/efficientnet_model.py

import torch
import torch.nn as nn
from torchvision import models

def get_efficientnet_model(pretrained=True):
    """
    Loads EfficientNet (e.g. b0) and adapts it for binary classification.
    """
    model = models.efficientnet_b0(pretrained=pretrained)
    model.classifier[1] = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 1),
        nn.Sigmoid()
    )
    return model
