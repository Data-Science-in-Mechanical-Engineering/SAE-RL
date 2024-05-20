from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

from .layers import BackgroundBias, GaussianSampling, KeyNetConvBlock


class DecoderDSAE(nn.Module):
    """SAE decoder architecture from "[Deep Spatial Autoencoders for Visuomotor Learning](https://ieeexplore.ieee.org/document/7487173)", Finn et al. 2016.
    """

    def __init__(self, n_coordinates: int, target_shape: Tuple[int, int, int], background_bias: bool = False):
        """Initializes the DSAE decoder.

        Args:
            n_coordinates (int): Number of 2D coordinates in the latent space.
            target_shape (Tuple[int, int, int]): Desired image dimensions to be reconstructed with shape (CHANNELS, HEIGHT, WIDTH).
            background_bias (bool, optional): Whether to include a final layer for background reconstruction. Defaults to False.
        """

        super().__init__()

        self.target_shape = target_shape

        self.dense = nn.Linear(in_features=(n_coordinates * 2), out_features=np.prod(target_shape))

        self.background_bias = BackgroundBias(target_shape) \
            if background_bias else (lambda x: x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.flatten(start_dim=1)
        x = self.dense(x)
        x = x.reshape([-1, *self.target_shape])
        x = self.background_bias(x)

        return x


class DecoderKeyNet(nn.Module):
    """SAE decoder architecture from "[Unsupervised Learning of Object Landmarks through Conditional Image Generation](https://papers.nips.cc/paper/2018/hash/1f36c15d6a3d18d52e8d493bc8187cb9-Abstract.html)", Jakab and Gupta et al. 2018.

    Note:
    Changes from the original `goal_prediction_model` repository:
    - renamed to KeyNet, since Transporter is a different architecture
    - added blocks of two convolutional layers each, according to the paper
    """

    def __init__(self, n_coordinates: int, target_shape: Tuple[int, int, int], gaussian_std: float = 0.1, limit_gaussian_std: float = None, train_gaussian_std: bool = False, share_gaussian_std: bool = True, n_filters: int = 256, kernel_size: int = 3, batch_norm: bool = True, background_bias: bool = False):
        """Initializes the KeyNet decoder.

        Args:
            n_coordinates (int): Number of 2D coordinates in the latent space.
            target_shape (Tuple[int, int, int]): Desired image dimensions to be reconstructed with shape (CHANNELS, HEIGHT, WIDTH).
            gaussian_std (float, optional): Standard deviation for isotropic Gaussian activation. Defaults to 0.1.
            limit_gaussian_std (float, optional): Number of standard deviations from the mean outside of which Gaussian maps will be 0.0, limiting the support. If None, support will be unlimited. Defaults to None.
            train_gaussian_std (bool, optional): Whether the standard deviation should be optimized during the training process. Defaults to False.
            share_gaussian_std (bool, optional): Whether to share the same standard deviation across feature maps. If False, a different standard deviation can be learned for every feature map. Only has an effect if `train_gaussian_std` is True. Defaults to True.
            n_filters (int, optional): Number of output filters for convolutions in the first block. Every following blocks halves this number. Defaults to 256.
            kernel_size (int, optional): Kernel size for all convolutional layers. Defaults to 3.
            batch_norm (bool, optional): Whether to apply batch normalization after each convolutional layer. Defaults to True.
            background_bias (bool, optional): Whether to include a final layer for background reconstruction. Defaults to False.
        """

        super().__init__()

        assert ((target_shape[1] % 4) == 0) and ((target_shape[2] % 4) == 0)
        # quarter of the target size due to two 2× interpolations later
        maps_shape = (n_coordinates, target_shape[1] // 4, target_shape[2] // 4)

        self.gaussian_sampling = GaussianSampling(out_shape=maps_shape, std=gaussian_std, limit_std=limit_gaussian_std,
                                                  train_std=train_gaussian_std, share_std=share_gaussian_std)

        self.block1 = KeyNetConvBlock(in_channels=n_coordinates, out_channels=n_filters,
                                      kernel_sizes=kernel_size, stride=1, batch_norm=batch_norm)
        self.block2 = KeyNetConvBlock(in_channels=n_filters, out_channels=(n_filters // 2),
                                      kernel_sizes=kernel_size, stride=1, batch_norm=batch_norm)
        self.block3 = KeyNetConvBlock(in_channels=(n_filters // 2), out_channels=(n_filters // 4),
                                      kernel_sizes=kernel_size, stride=1, batch_norm=batch_norm)

        self.aggregate_conv = nn.Conv2d(in_channels=(n_filters // 4), out_channels=target_shape[0], kernel_size=1)

        self.background_bias = BackgroundBias(target_shape) \
            if background_bias else (lambda x: x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.gaussian_sampling(x)
        x = self.block1(x)
        x = f.interpolate(x, scale_factor=(2, 2), mode='bilinear')
        x = self.block2(x)
        x = f.interpolate(x, scale_factor=(2, 2), mode='bilinear')
        x = self.block3(x)
        x = self.aggregate_conv(x)
        x = self.background_bias(x)

        return x


class DecoderBasic(nn.Module):
    """Basic SAE decoder architecture to try out architecture variations against.
    """

    def __init__(self, n_coordinates: int, target_shape: Tuple[int, int, int], gaussian_std: float = 0.1, limit_gaussian_std: float = None, train_gaussian_std: bool = False, share_gaussian_std: bool = True, out_channels: Tuple[int, int, int] = (64, 32, 16), kernel_sizes: Tuple[int, int, int] = (3, 3, 3), batch_norm: bool = True, background_bias: bool = False):
        """Initializes basic decoder.

        Args:
            n_coordinates (int): Number of 2D coordinates in the latent space.
            target_shape (Tuple[int, int, int]): Desired image dimensions to be reconstructed with shape (CHANNELS, HEIGHT, WIDTH).
            gaussian_std (float, optional): Standard deviation for isotropic Gaussian activation. Defaults to 0.1.
            limit_gaussian_std (float, optional): Number of standard deviations from the mean outside of which Gaussian maps will be 0.0, limiting the support. If None, support will be unlimited. Defaults to None.
            train_gaussian_std (bool, optional): Whether the standard deviation should be optimized during the training process. Defaults to False.
            share_gaussian_std (bool, optional): Whether to share the same standard deviation across feature maps. If False, a different standard deviation can be learned for every feature map. Only has an effect if `train_gaussian_std` is True. Defaults to True.
            out_channels (Tuple[int, int, int], optional): Number of output channels for all convolutional layers. Defaults to (64, 32, 16).
            kernel_sizes (Tuple[int, int, int], optional): Kernel sizes for all convolutional layers. Defaults to (3, 3, 3).
            batch_norm (bool, optional): Whether to apply batch normalization after each convolutional layer. Defaults to True.
            background_bias (bool, optional): Whether to include a final layer for background reconstruction. Defaults to False.
        """

        super().__init__()

        maps_shape = (n_coordinates, target_shape[1], target_shape[2])

        self.gaussian_sampling = GaussianSampling(out_shape=maps_shape, std=gaussian_std, limit_std=limit_gaussian_std,
                                                  train_std=train_gaussian_std, share_std=share_gaussian_std)

        self.conv1 = nn.Conv2d(in_channels=n_coordinates, out_channels=out_channels[0], kernel_size=kernel_sizes[0],
                               bias=(not batch_norm), padding=(kernel_sizes[0] // 2))
        self.bn1 = nn.BatchNorm2d(num_features=out_channels[0]) \
            if batch_norm else (lambda x: x)

        self.conv2 = nn.Conv2d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=kernel_sizes[1],
                               bias=(not batch_norm), padding=(kernel_sizes[0] // 2))
        self.bn2 = nn.BatchNorm2d(num_features=out_channels[1]) \
            if batch_norm else (lambda x: x)

        self.conv3 = nn.Conv2d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=kernel_sizes[2],
                               bias=(not batch_norm), padding=(kernel_sizes[0] // 2))
        self.bn3 = nn.BatchNorm2d(num_features=out_channels[2]) \
            if batch_norm else (lambda x: x)

        self.aggregate_conv = nn.Conv2d(in_channels=out_channels[2], out_channels=target_shape[0], kernel_size=1)

        self.background_bias = BackgroundBias(target_shape) \
            if background_bias else (lambda x: x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.gaussian_sampling(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = f.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = f.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = f.relu(x)
        x = self.aggregate_conv(x)
        x = self.background_bias(x)

        return x
