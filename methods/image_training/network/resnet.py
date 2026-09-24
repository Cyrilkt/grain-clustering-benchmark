"""RGB ResNet-50 feature extractor adapted from ProPos (Zhizhong Huang)."""
import torch.nn as nn
from torchvision.models.resnet import ResNet as TorchvisionResNet, Bottleneck

class ResNet:
    def __init__(self, net_name):
        if net_name != 'resnet50':
            raise ValueError('The Grain benchmark uses ResNet-50')

    def __call__(self):
        # Construct the original 1000-class layer before replacing it, preserving RNG.
        model = TorchvisionResNet(Bottleneck, [3, 4, 6, 3])
        return nn.Sequential(*[
            nn.Flatten(1) if isinstance(module, nn.Linear) else module
            for module in model.children()
        ])
