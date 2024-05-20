import torch
import torch.nn.functional as f


def reconstruction_loss(input: torch.Tensor, target: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """Loss function for reconstruction objective of an autoencoder architecture.

    A batch of video snippets with multiple frames is expected for both input and target.

    Args:
        input (torch.Tensor): A batch of predicted video snippets with shape (BATCH, FRAMES, CHANNELS, HEIGHT, WIDTH).
        target (torch.Tensor): A batch of target video snippets with shape (BATCH, FRAMES, CHANNELS, HEIGHT, WIDTH).
        reduction (str, optional): Specifies the reduction to apply to the output, 'none', 'mean' or 'sum'. Defaults to 'mean'.

    Returns:
        torch.Tensor: The computed loss.
    """

    loss = f.mse_loss(
        input, target, reduction=reduction
    )

    return loss


def velocity_loss(input: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """Loss function for consistent feature point velocities in a spatial autoencoder latent space.

    A batch of feature point snippets exactly three frames is expected for both input and target, with the frame dimension being 1. The loss is computed as an MSE loss between the feature point position changes from frame 0 to 1 and from frame 1 to 2.

    Args:
        input (torch.Tensor): A batch of predicted feature point snippets with shape (BATCH, FRAMES, FEATURE_POINTS, YX).
        reduction (str, optional): Specifies the reduction to apply to the output, 'none', 'mean' or 'sum'. Defaults to 'mean'.

    Returns:
        torch.Tensor: The computed loss.
    """

    loss = f.mse_loss(
        input[:, 2] - input[:, 1], input[:, 1] - input[:, 0],
        reduction=reduction
    )

    return loss
