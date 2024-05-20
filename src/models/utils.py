import os

import torch
from torchvision.models import DenseNet201_Weights, densenet201

from ..utils import root

os.environ['TORCH_HOME'] = str(root / 'cache')


def get_densenet201_weight() -> torch.Tensor:
    """Fetches weights for a DenseNet201 instance pretrained on ImageNet and returns the first convolutional layer's weight.

    DenseNet201 does not include a bias in this convolutional layer.

    Returns:
        torch.Tensor: The weight tensor of the first convolutional layer in DenseNet201.
    """

    model = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
    weight = model.features.conv0.weight

    return weight
