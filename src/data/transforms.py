from typing import Tuple

import numpy as np
import torch
from torchvision.transforms import Compose, GaussianBlur, Grayscale, Resize


class IdentityTransform():
    """Callable identity transform class for returning the input."""

    def __call__(self, x):

        return x


class ResizeTransform():
    """Callable transformation class for resizing and or grayscale conversion.
    """

    def __init__(self, output_shape: Tuple[int, int, int] = (1, 60, 60)):
        """Initializes the transformation between input and output images.
        The first output shape dimension can be either 3 for RGB images or 1 for grayscale images.

        Args:
            output_shape (Tuple[int, int, int], optional): Output dimensions for a single image with shape (CHANNELS, HEIGHT, WIDTH). Defaults to (1, 60, 60).
        """

        self.output_shape = output_shape

        # compose full transformation from resize transformation and/or grayscale conversion
        operations = []
        operations.append(Resize(output_shape[-2:]))
        if output_shape[0] == 1:  # if output image has just one (grayscale) dimension
            operations.append(Grayscale())
        self.transform = Compose(operations)

    def __call__(self, input_images: torch.Tensor) -> torch.Tensor:

        # transform images
        output_images = self.transform(input_images)

        return output_images


class NoiseTransform():
    """Callable transformation class for adding Gaussian sampled noise.
    """

    def __init__(self, stddev: float = 0.001):
        """Initializes the transformation adding Gaussian noise.

        Args:
            stddev (float): Standard deviation for the added noise. Defaults to 0.001.
        """

        self.stddev = stddev

    def __call__(self, input_images: torch.Tensor) -> torch.Tensor:

        # add N(0,1)-sampled noise, scaled with standard deviation
        output_images = input_images + self.stddev * torch.randn(*input_images.shape)

        return output_images


class LocalContrastNormalization():
    """Callable transformation class to perform local contrast normalization to images, following "[What is the Best Multi-Stage Architecture for Object Recognition?](https://ieeexplore.ieee.org/document/5459469)", Jarrett et al. 2009.
    """

    def __init__(self, kernel_size: int = 9, stddev: float = 1.0):
        """Initializes the local contrast normalization transformation.

        Args:
            kernel_size (int, optional): Kernel size for Gaussian smoothing kernel. Defaults to 9.
            stddev: Standard deviation for Gaussian smoothing kernel in pixels. Defaults to 1.0.
        """

        self.gaussian = GaussianBlur(kernel_size, stddev)

    def __call__(self, input_images: torch.Tensor) -> torch.Tensor:

        # apply LCN to every image in batch
        if input_images.dim() == 4:
            output_images = torch.stack([self.__apply(image) for image in input_images])
        else:
            output_images = self.__apply(input_images)

        return output_images

    def __apply(self, input: torch.Tensor) -> torch.Tensor:
        """Applies local contrast normalization to a single image tensor.

        Args:
            input (torch.Tensor): Image tensor to normalize.

        Returns:
            torch.Tensor: Normalized image tensor.
        """

        v = input - self.gaussian(input).mean(dim=-3)
        sigma = torch.sqrt(self.gaussian(v**2).mean(dim=-3))
        c = sigma.mean(dim=[-1, -2])
        output = v / torch.max(c, sigma)

        o_min, o_max = output.min(), output.max()
        output = (output - o_min) / (o_max - o_min)

        return output


class BackgroundSegmentation():
    """Callable transformation class to subtract the median training image from images, threshold the absolute difference to get a segmentation mask and then mask the original image with this.
    """

    def __init__(self, median_image: torch.Tensor, segmentation_threshold: float = 0.01, segmentation_output: str = 'masked-image'):
        """Initializes the background segmentation transformation.

        Args:
            median_image (torch.Tensor): The median training image tensor.
            segmentation_threshold (float, optional): Threshold above which (in absolute numbers) to count a deviation from the median image. Defaults to 0.01.
            segmentation_output (str, optional): What image to output. Either 'masked-image', 'segmentation' or 'difference'. Defaults to 'masked-image'.

        Raises:
            ValueError: Raised if `segmentation_output` is none of the valid options.
        """

        self.median_image = median_image
        self.segmentation_threshold = segmentation_threshold
        if segmentation_output in ['masked-image', 'segmentation', 'difference']:
            self.segmentation_output = segmentation_output
        else:
            raise ValueError(
                f'Invalid segmentation output format {segmentation_output}. Expected "masked-image", "segmentation" or "difference".')

    def __call__(self, input_images):

        abs_diff = (input_images - self.median_image).abs()
        if self.segmentation_output == 'difference':
            output_images = abs_diff
        else:
            input_images[torch.where(abs_diff < self.segmentation_threshold)] = 0.0
            if self.segmentation_output == 'segmentation':
                input_images[torch.where(abs_diff >= self.segmentation_threshold)] = 1.0
            output_images = input_images

        return output_images
