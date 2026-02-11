"""
Document classification model - ResNet-based for Brazilian document types
"""
import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes: int, pretrained: bool = True):
    """Create a ResNet18-based document classifier."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


class DocumentClassifier(nn.Module):
    """Wrapper for document classification model."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone = create_model(num_classes, pretrained)

    def forward(self, x):
        return self.backbone(x)
