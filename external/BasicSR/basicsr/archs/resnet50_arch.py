import torch
import torch.nn as nn
from torchvision import models

from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class ResNet50(nn.Module):
    """ResNet50 for classification
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use ImageNet pretrained weights.
    """

    def __init__(self, num_classes = 4, pretrained = True):
        super(ResNet50, self).__init__()

        self.backbone = models.resnet50(pretrained=pretrained)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)