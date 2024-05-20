from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as f

from ..utils import Bunch, deflate, gl, inflate


class SpatialSoftmax(nn.Module):
    """Network layer performing softmax normalization on 2D input activation maps.
    """

    def __init__(self, temperature: float = 1.0, train_temperature: bool = True, **_):
        """Initializes a Spatial Softargmax layer.

        Args:
            temperature (float, optional): The softmax temperature parameter to use. Defaults to 1.0.
            train_temperature (bool, optional): Whether the softmax temperature parameter should be optimized during the training process. Defaults to True.
        """

        super().__init__()

        # set softmax temperature, optionally as a trainable parameter
        self.temperature = temperature
        if train_temperature:
            self.temperature = nn.Parameter(torch.tensor([self.temperature]))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies softmax normalization to 2D input activation maps.

        Args:
            input (torch.Tensor): Output activation maps of a convolutional layer.

        Returns:
            torch.Tensor: Softmax normalized attention maps.
        """

        _, channels, height, width = input.shape

        # apply softmax to each input activation map
        output = f.softmax(
            input.view(-1, channels, height * width) / self.temperature, dim=-1
        ).view(-1, channels, height, width)

        return output


class SpatialSoftargmax(nn.Module):
    """Network layer performing softmax normalization on each activation map, weighting image coordinates with these softmax scores and returning the averaged feature point coordinates.
    """

    def __init__(self, temperature: float = 1.0, train_temperature: bool = True, **_):
        """Initializes a Spatial Softargmax layer.

        Args:
            temperature (float, optional): The softmax temperature parameter to use. Defaults to 1.0.
            train_temperature (bool, optional): Whether the softmax temperature parameter should be optimized during the training process. Defaults to True.
        """

        super().__init__()

        # spatial softmax to apply to each activation map
        self.spatial_softmax = SpatialSoftmax(temperature=temperature, train_temperature=train_temperature)

        # set up hooks for attention and feature point retrieval
        self._setup_output_hooks()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Retrieves feature point image coordinates via spatial softargmax.

        Args:
            input (torch.Tensor): Output activation maps of a convolutional layer.

        Returns:
            torch.Tensor: Feature points for each activation map.
        """

        # apply softmax to each input activation map
        # attention shape (BATCH, CHANNELS, HEIGHT, WIDTH)
        attention = self.spatial_softmax(input)

        _, _, height, width = attention.shape

        # build 2D meshs with size of input and x and y coordinates, respectively
        # pos_x/pos_y shape (BATCH, CHANNELS, HEIGHT, WIDTH)
        ys = torch.linspace(-1, 1, height, device=gl.device)
        xs = torch.linspace(-1, 1, width, device=gl.device)
        pos_x, pos_y = torch.meshgrid(xs, ys, indexing='xy')

        # weight coordinate grids by softmax attention and combine
        # expected_yx shape (BATCH, CHANNELS, YX)
        expected_y = (attention * pos_y).sum(dim=(-1, -2))
        expected_x = (attention * pos_x).sum(dim=(-1, -2))
        expected_yx = torch.stack([expected_y, expected_x], dim=-1)

        return expected_yx

    def _setup_output_hooks(self):
        """Sets up hooks to save the attention and feature point outputs of the softmax and argsoftmax layers respectively during a forward pass.
        """

        # container for all hook handles
        self.handles = Bunch()

        # define hook to save layer output when in eval mode
        def _output_hook(self, _, output):
            if not self.training:
                self.output = output.detach()

        # register attention and feature points hooks
        self.handles.attention = self.spatial_softmax.register_forward_hook(_output_hook)
        self.handles.feature_points = self.register_forward_hook(_output_hook)

    def get_presence(self) -> torch.Tensor:
        """Retrieves the softmax value at the softargmax location, i.e., the presence value from "[Deep Spatial Autoencoders for Visuomotor Learning](https://ieeexplore.ieee.org/document/7487173)", Finn et al. 2016.

        Returns:
            torch.Tensor: A batch with presence values for each feature point with shape (BATCH, FEATURE_POINTS).

        Raises:
            Exception: Raised if presence is requested during training, where it's unavailable.
        """

        if self.training:
            raise Exception('Presence is unavailable when model is in training mode.')

        # access softmax and softargmax outputs
        attention = self.spatial_softmax.output
        feature_points = self.output

        batch_size, channels, height, width = attention.shape

        # get y/x indices from coordinates
        idcs_y = deflate(torch.round((feature_points[:, :, 0] + 1) / 2 * (height-1)).long())
        idcs_x = deflate(torch.round((feature_points[:, :, 1] + 1) / 2 * (width-1)).long())
        idcs_samples = torch.arange(batch_size * channels)

        # obtain presence as softmax value at softargmax location
        presence = inflate(deflate(attention)[idcs_samples, idcs_y, idcs_x], batch_size)

        return presence


class BackgroundBias(nn.Module):
    """Background bias layer with values for every RGB or grayscale value in every pixel.
    """

    def __init__(self, shape: Tuple[int, int, int]):
        """Initializes background bias layer with zeros.

        Args:
            shape (Tuple[int, int, int]): Shape of the image input and output of this layer.
        """

        super().__init__()

        # bias with image dimensions (CHANNELS, HEIGHT, WIDTH)
        self.bias = nn.Parameter(torch.zeros(list(shape)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Adds background bias to input images.

        Args:
            input (torch.Tensor): Batch of input images.

        Returns:
            torch.Tensor: Batch of input images plus background bias.
        """

        output = input + self.bias

        return output


class GaussianSampling(nn.Module):
    """Gaussian sampling layer returning a activation map for each coordinate with a small isotropic Gaussian activation at the coordinate position.
    """

    def __init__(self, out_shape: Tuple[int, int], std: float = 0.1, limit_std: float = None, train_std: bool = False, share_std: bool = True):
        """Initializes Gaussian sampling layer.

        Args:
            out_shape (Tuple[int, int, int]): Output shape of the activation maps generated by this layer with shape (COORDS, HEIGHT, WIDTH).
            std (float, optional): Standard deviation for isotropic Gaussian activation. Defaults to 0.1.
            limit_std (float, optional): Number of standard deviations from the mean outside of which Gaussian maps will be 0.0, limiting the support. If None, support will be unlimited. Defaults to None.
            train_std (bool, optional): Whether the standard deviation should be optimized during the training process. Defaults to False.
            share_std (bool, optional): Whether to share the same standard deviation across feature maps. If False, a different standard deviation can be learned for every feature map. Only has an effect if `train_std` is True. Defaults to True.
        """

        super().__init__()

        # set sampling standard deviation, optionally as a trainable parameter and shared
        if share_std:
            self.std = torch.tensor([std], device=gl.device)
        else:
            self.std = torch.tensor([std] * out_shape[0], device=gl.device).unsqueeze(dim=1).unsqueeze(dim=2)
        if train_std:
            self.std = nn.Parameter(self.std)

        self.limit_std = limit_std

        # set up y/x mesh for Gaussian sampling
        ys = torch.linspace(-1, 1, out_shape[1], device=gl.device)
        xs = torch.linspace(-1, 1, out_shape[2], device=gl.device)
        pos_x, pos_y = torch.meshgrid(xs, ys, indexing='xy')
        self.pos_y = pos_y.view(1, 1, *pos_y.shape)
        self.pos_x = pos_x.view(1, 1, *pos_x.shape)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Generates activation maps with one Gaussian activation each for each input coordinate.

        Args:
            input (torch.Tensor): Batch with multiple y/x coordinate values for each sample.

        Returns:
            torch.Tensor: Batch with multiple activation maps for each sample.
        """

        # y/x mean positions for Gaussian activations
        mu_y = input[:, :, 0].view(*input.shape[:2], 1, 1)
        mu_x = input[:, :, 1].view(*input.shape[:2], 1, 1)

        # fill maps with Gaussian activations
        output = torch.exp(
            -((self.pos_x - mu_x)**2 + (self.pos_y - mu_y)**2) / self.std**2
        )  # shape (BATCH, COORDS, HEIGHT, WIDTH)

        if self.limit_std is not None:
            # set maps outside of limit_std standard deviations to 0.0
            self.limit_std_value = torch.exp(-(self.std * self.limit_std)**2 / self.std**2)
            output = output - self.limit_std_value
            output = f.relu(output)

        return output


class KeyNetConvBlock(nn.Module):
    """Convolutional block for KeyNet, consisting of two convolutional layers with optional batch normalization.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: Union[Tuple[int, int], int], stride: int, batch_norm: bool):
        """Initilizes a convolutional block.

        Args:
            in_channels (int): Number of input channels to the first convolutional layer.
            out_channels (int): Number of output channels of both convolutional layers.
            kernel_sizes (Union[Tuple[int, int], int]): Kernel sizes for the first and second convolutional layer.
            stride (int): Stride of the first convolutional layer, used for downsampling.
            batch_norm (bool): Whether to apply batch normalization after each convolutional layer.
        """

        super().__init__()

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * 2

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_sizes[0],
                               bias=(not batch_norm), stride=stride, padding=(kernel_sizes[0] // 2))
        self.bn1 = nn.BatchNorm2d(num_features=out_channels) \
            if batch_norm else (lambda x: x)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_sizes[1],
                               bias=(not batch_norm), padding=(kernel_sizes[1] // 2))
        self.bn2 = nn.BatchNorm2d(num_features=out_channels) \
            if batch_norm else (lambda x: x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv1(x)
        x = self.bn1(x)
        x = f.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = f.relu(x)

        return x


class BasicConvBlock(nn.Module):
    """Convolutional block for Basic, consisting of two convolutional layers with pooling and optional batch normalization.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, batch_norm: bool):
        """Initializes a convolutional block.

        Args:
            in_channels (int): Number of input channels to the first convolutional layer.
            out_channels (int): Number of output channels of both convolutional layers.
            kernel_size (int): Kernel size for both convolutional layers.
            batch_norm (bool): Whether to apply batch normalization after each convolutional layer.
        """

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               bias=(not batch_norm), padding=(kernel_size // 2))
        self.bn1 = nn.BatchNorm2d(num_features=out_channels) \
            if batch_norm else (lambda x: x)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               bias=(not batch_norm), padding=(kernel_size // 2))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels) \
            if batch_norm else (lambda x: x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv1(x)
        x = self.bn1(x)
        x = f.relu(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.bn2(x)
        x = f.relu(x)

        return x
