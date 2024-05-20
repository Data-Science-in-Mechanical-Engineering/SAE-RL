from typing import List

import torch

from ..utils import Bunch
from .transforms import BaseTransform2D, all_transforms


def all_tracking_errors(feature_points: torch.Tensor, keypoints: torch.Tensor) -> Bunch:
    """Compute tracking errors between transformed feature points and ground truth keypoints for all transformations.

    Args:
        feature_points (torch.Tensor): Feature points to transform.
        keypoints (torch.Tensor): Ground truth keypoint locations.

    Returns:
        Bunch: Bunch of transforms with corresponding lists of tracking errors for all objects.
    """

    results = Bunch()

    for name, transform in all_transforms.items():
        results[name] = tracking_errors(feature_points, keypoints, transform)

    return results


def tracking_errors(feature_points: torch.Tensor, keypoints: torch.Tensor, transform: BaseTransform2D) -> List[float]:
    """Compute tracking errors between transformed feature points and ground truth keypoints, fitting a given transformation.

    Note: The minimal error for each keypoint is returned, corresponding to the feature point best tracking it.

    Args:
        feature_points (torch.Tensor): Feature points to transform.
        keypoints (torch.Tensor): Ground truth keypoint locations.
        transform (BaseTransform): Transformation to fit and use for evaluation.

    Returns:
        List[float]: Tracking errors for all keypoints.
    """

    n_sites = keypoints.shape[1]
    n_fps = feature_points.shape[1]

    pairwise_errors = torch.empty((n_sites, n_fps))

    # compute error for each pair of site and feature point
    for site in range(n_sites):
        for fp in range(n_fps):
            regr = transform()
            regr.fit(feature_points[:, fp], keypoints[:, site])  # fit transformation
            pairwise_errors[site, fp] = regr.mse(feature_points[:, fp], keypoints[:, site])

    # find tracking errors as minimum error for every keypoint
    errors = pairwise_errors.min(dim=1).values

    return errors
