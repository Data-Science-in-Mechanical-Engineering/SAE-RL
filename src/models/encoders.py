from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as f

from .layers import BasicConvBlock, KeyNetConvBlock, SpatialSoftargmax
from .utils import get_densenet201_weight


class EncoderDSAE(nn.Module):
    """SAE encoder architecture from "[Deep Spatial Autoencoders for Visuomotor Learning](https://ieeexplore.ieee.org/document/7487173)", Finn et al. 2016.

    Note:
    Changes from the original `goal_prediction_model` repository:
    - added biases for all convolutional layers
    - added learnable center and scale parameters for all batchnorm layers
    """

    def __init__(self, n_coordinates: int = 16, out_channels: Tuple[int, int] = (64, 32), kernel_sizes: Tuple[int, int, int] = (7, 5, 5), batch_norm: bool = True, softmax_temperature: float = 1.0, train_softmax_temperature: bool = True, imagenet_weights: bool = True):
        """Initializes the DSAE encoder.

        Args:
            n_coordinates (int, optional): Number of 2D coordinates in the latent space. Defaults to 16.
            out_channels (Tuple[int, int], optional): Number of output channels for first two convolutional layers. Defaults to (64, 32).
            kernel_sizes (Tuple[int, int, int], optional): Kernel sizes for all three convolutional layers. Defaults to (7, 5, 5).
            batch_norm (bool, optional): Whether to apply batch normalization after each convolutional layer. Defaults to True.
            softmax_temperature (float, optional): The softmax temperature parameter to use. Defaults to 1.0.
            train_softmax_temperature (bool, optional): Whether the softmax temperature parameter should be optimized during the training process. Defaults to True.
            imagenet_weights (bool, optional): Whether to initialize the filters of the first convolutional layer with a network trained on ImageNet. Defaults to True.
        """

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=out_channels[0],
                               kernel_size=kernel_sizes[0], bias=(not batch_norm), stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels[0]) \
            if batch_norm else (lambda x: x)

        self.conv2 = nn.Conv2d(in_channels=out_channels[0], out_channels=out_channels[1],
                               kernel_size=kernel_sizes[1], bias=(not batch_norm))
        self.bn2 = nn.BatchNorm2d(num_features=out_channels[1]) \
            if batch_norm else (lambda x: x)

        self.conv3 = nn.Conv2d(in_channels=out_channels[1], out_channels=n_coordinates,
                               kernel_size=kernel_sizes[2], bias=(not batch_norm))
        self.bn3 = nn.BatchNorm2d(num_features=n_coordinates) \
            if batch_norm else (lambda x: x)

        self.spatial_softargmax = SpatialSoftargmax(temperature=softmax_temperature,
                                                    train_temperature=train_softmax_temperature)

        # set pretrained weights for first layer
        if imagenet_weights:
            with torch.no_grad():
                self.conv1.weight = get_densenet201_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv1(x)
        x = self.bn1(x)
        x = f.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = f.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = f.relu(x)
        x = self.spatial_softargmax(x)

        return x


class EncoderKeyNet(nn.Module):
    """SAE encoder architecture from "[Unsupervised Learning of Object Landmarks through Conditional Image Generation](https://papers.nips.cc/paper/2018/hash/1f36c15d6a3d18d52e8d493bc8187cb9-Abstract.html)", Jakab and Gupta et al. 2018.

    Note:
    Changes from the original `goal_prediction_model` repository:
    - renamed to KeyNet, since Transporter is a different architecture
    - added blocks of two convolutional layers each, according to the paper
    """

    def __init__(self, n_coordinates: int = 30, n_filters: int = 32, kernel_sizes: Tuple[int, int] = (7, 3), batch_norm: bool = True, softmax_temperature: float = 1.0, train_softmax_temperature: bool = False):
        """Initializes the KeyNet encoder.

        Args:
            n_coordinates (int, optional): Number of 2D coordinates in the latent space. Defaults to 30.
            n_filters (int, optional): Number of output filters for convolutions in the first block. Every following blocks doubles this number. Defaults to 32.
            kernel_sizes (Tuple[int, int], optional): Kernel sizes for the first and for all other convolutional layers. Defaults to (7, 3).
            batch_norm (bool, optional): Whether to apply batch normalization after each convolutional layer. Defaults to True.
            softmax_temperature (float, optional): The softmax temperature parameter to use. Defaults to 1.0.
            train_softmax_temperature (bool, optional): Whether the softmax temperature parameter should be optimized during the training process. Defaults to False.
        """

        super().__init__()

        self.block1 = KeyNetConvBlock(in_channels=3, out_channels=n_filters,
                                      kernel_sizes=kernel_sizes, stride=1, batch_norm=batch_norm)
        self.block2 = KeyNetConvBlock(in_channels=n_filters, out_channels=(n_filters * 2),
                                      kernel_sizes=kernel_sizes[1], stride=2, batch_norm=batch_norm)
        self.block3 = KeyNetConvBlock(in_channels=(n_filters * 2), out_channels=(n_filters * 4),
                                      kernel_sizes=kernel_sizes[1], stride=2, batch_norm=batch_norm)
        self.block4 = KeyNetConvBlock(in_channels=(n_filters * 4), out_channels=(n_filters * 8),
                                      kernel_sizes=kernel_sizes[1], stride=2, batch_norm=batch_norm)

        self.aggregate_conv = nn.Conv2d(in_channels=(n_filters * 8), out_channels=n_coordinates, kernel_size=1)

        self.spatial_softargmax = SpatialSoftargmax(temperature=softmax_temperature,
                                                    train_temperature=train_softmax_temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.aggregate_conv(x)
        x = self.spatial_softargmax(x)

        return x


class EncoderBasic(nn.Module):
    """Basic SAE encoder architecture to try out architecture variations against.
    """

    def __init__(self, n_coordinates: int = 16, out_channels: Tuple[int, int, int] = (32, 64, 128), kernel_sizes: Tuple[int, int, int] = (3, 3, 3), batch_norm: bool = True, softmax_temperature: float = 1.0, train_softmax_temperature: bool = False):
        """Initializes basic encoder.

        Args:
            n_coordinates (int, optional): Number of 2D coordinates in the latent space. Defaults to 16.
            out_channels (Tuple[int, int, int], optional): Number of output channels for both convolutional layers in all blocks. Defaults to (32, 64, 128).
            kernel_sizes (Tuple[int, int, int], optional): Kernel sizes for both convolutional layers in all blocks. Defaults to (3, 3, 3).
            batch_norm (bool, optional): Whether to apply batch normalization after each convolutional layer. Defaults to True.
            softmax_temperature (float, optional): The softmax temperature parameter to use. Defaults to 1.0.
            train_softmax_temperature (bool, optional): Whether the softmax temperature parameter should be optimized during the training process. Defaults to False.
        """

        super().__init__()

        self.block1 = BasicConvBlock(in_channels=3, out_channels=out_channels[0],
                                     kernel_size=kernel_sizes[0], batch_norm=batch_norm)
        self.block2 = BasicConvBlock(in_channels=out_channels[0], out_channels=out_channels[1],
                                     kernel_size=kernel_sizes[1], batch_norm=batch_norm)
        self.block3 = BasicConvBlock(in_channels=out_channels[1], out_channels=out_channels[2],
                                     kernel_size=kernel_sizes[2], batch_norm=batch_norm)

        self.aggregate_conv = nn.Conv2d(in_channels=out_channels[2], out_channels=n_coordinates, kernel_size=1)

        self.spatial_softargmax = SpatialSoftargmax(temperature=softmax_temperature,
                                                    train_temperature=train_softmax_temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.aggregate_conv(x)
        x = self.spatial_softargmax(x)

        return x
