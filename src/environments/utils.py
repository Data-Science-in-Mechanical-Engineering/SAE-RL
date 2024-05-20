from typing import List

import pybullet as p
import torch
import torch.nn.functional as f

from ..utils import deflate, dehomogenize, gl, homogenize, inflate


def antialias(image: torch.Tensor, factor: int) -> torch.Tensor:
    """Applies antialiasing for a given image by downsampling it via average-pooling.

    Note: This reduces the image resolution by the specified factor.

    Args:
        image (torch.Tensor): High-resolution input image to use for antialiasing.
        factor (int): Antialiasing factor for average-pooling kernel size. Image resolution is reduced by this factor.

    Returns:
        torch.Tensor: The antialiased output image.
    """

    if factor > 1:
        image = torch.movedim(image, -1, -3)
        image = f.avg_pool2d(image, kernel_size=factor, stride=factor)
        image = torch.movedim(image, -3, -1)

    return image


def project_2D(point: torch.Tensor, view_matrix: torch.Tensor, projection_matrix: torch.Tensor) -> torch.Tensor:
    """Projects a given 3D world coordinate onto its 2D location on the image plane, using the precomputed projection matrices.

    Args:
        point (torch.Tensor): 3D world coordinate to project.
        view_matrix (torch.Tensor): View matrix as 4×4 tensor.
        projection_matrix (torch.Tensor): Projection matrix as 4×4 tensor.

    Returns:
        torch.Tensor: Projected 2D coordinate in [-1, 1] for both dimensions.
    """

    # transform to camera coordinate system and then to image space via homogeneous coordinates
    point = dehomogenize(
        projection_matrix @ view_matrix.T @ homogenize(point)
    )

    # drop z-dimension and scale to [-1, 1] in both dimensions
    point = point[:2] / torch.tensor([5, -5], device=gl.device)

    return point


def compute_view_matrix(target_position: List[int], distance: float, yaw: float, pitch: float, roll: float) -> torch.Tensor:
    """Computes view matrix for specified camera parameters.

    Args:
        target_position (List[int]): Position targeted by the camera, as (x, y, z).
        distance (float): Distance of the camera.
        yaw (float): Yaw of the camera.
        pitch (float): Pitch of the camera.
        roll (float): Roll of the camera.

    Returns:
        torch.Tensor: View matrix as 4×4 tensor.
    """

    view_matrix = torch.tensor(
        p.computeViewMatrixFromYawPitchRoll(target_position, distance, yaw, pitch, roll, 2),
        device=gl.device).reshape([4, 4])

    return view_matrix


def compute_projection_matrix(aspect_ratio: float) -> torch.Tensor:
    """Computes projection matrix for specified aspect ratio.

    Args:
        aspect_ratio (float): Image aspect ratio as width/height.

    Returns:
        torch.Tensor: Projection matrix as 4×4 tensor.
    """

    projection_matrix = torch.tensor(
        p.computeProjectionMatrixFOV(fov=60, aspect=aspect_ratio, nearVal=0.1, farVal=100.0),
        device=gl.device).reshape([4, 4])
    
    return projection_matrix
