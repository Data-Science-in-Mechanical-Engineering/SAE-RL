from abc import ABC, abstractmethod

import torch
import torch.nn.functional as f
from torch.linalg import lstsq

from ..utils import Bunch, dehomogenize, homogenize


class BaseTransform2D(ABC):
    """Abstract base class for all linear 2D transformations.

    Transformation matrix is assumed to be 3×3 and is multiplied from the left side with a homogeneous row vector `[y, x, 1]`, with y being the leading dimension.
    """

    def __init__(self):
        """Initializes 2D transformation matrix to identity.
        """

        self.x = torch.eye(3)  # identity transform for homogeneous coordinates

    @abstractmethod
    def fit(self, fps: torch.Tensor, kps: torch.Tensor):
        """Fits transformation matrix for given feature points and keypoints.

        Args:
            fps (torch.Tensor): Sequence of feature points.
            kps (torch.Tensor): Sequence of ground truth keypoints.

        Raises:
            NotImplementedError: Raised if this abstract method is not overridden.
        """

        raise NotImplementedError

    def mse(self, fps: torch.Tensor, kps: torch.Tensor) -> torch.Tensor:
        """Computes mean squared error between transformed feature points and keypoints.

        Args:
            fps (torch.Tensor): Sequence of feature points.
            kps (torch.Tensor): Sequence of ground truth keypoints.

        Returns:
            torch.Tensor: Mean squared tracking error.
        """

        pred = self.transform(fps)
        squared_error = f.mse_loss(pred, kps, reduction='none')
        mse = torch.mean(torch.sum(squared_error, dim=1), dim=0)

        return mse

    def transform(self, fps: torch.Tensor) -> torch.Tensor:
        """Transforms feature points according to this transform's transformation matrix.

        Args:
            fps (torch.Tensor): Sequence of feature points.

        Returns:
            torch.Tensor: Sequence of transformed feature points.
        """

        pred = dehomogenize(
            homogenize(fps) @ self.x
        )

        return pred


class IdentityTransform(BaseTransform2D):
    """Identity transformation in 2D.

    This corresponds to an identity transformation matrix.
    """

    def fit(self, fps: torch.Tensor, kps: torch.Tensor):

        pass

    def mse(self, fps: torch.Tensor, kps: torch.Tensor) -> torch.Tensor:
        """Computes mean squared error between feature points and keypoints, omitting the identity transformation.

        Args:
            fps (torch.Tensor): Sequence of feature points.
            kps (torch.Tensor): Sequence of ground truth keypoints.

        Returns:
            torch.Tensor: Mean squared tracking error.
        """

        squared_error = f.mse_loss(fps, kps, reduction='none')
        mse = torch.mean(torch.sum(squared_error, dim=1), dim=0)

        return mse

    def transform(self, fps: torch.Tensor) -> torch.Tensor:
        """Returns the input feature points directly, omitting the identity tranformation.
        """

        return fps


class ScaleTranslateTransform(BaseTransform2D):
    """Scale and translate transformation in 2D.

    This corresponds to the following transformation matrix `x` with scale parameters `s_y`, `s_x` and translation parameters `t_y` and `t_x`.
    ```
    x =
    [[s_y, 0  , 0],
     [0  , s_x, 0],
     [t_y, t_x, 1]]
    ```
    """

    def fit(self, fps: torch.Tensor, kps: torch.Tensor):

        # fit variable parameters in first two columns
        self.x[[0, 2], [0, 0]] = lstsq(homogenize(fps[:, [0]]), kps[:, 0]).solution.squeeze(-1)
        self.x[[1, 2], [1, 1]] = lstsq(homogenize(fps[:, [1]]), kps[:, 1]).solution.squeeze(-1)


class AffineTransform(BaseTransform2D):
    """Affine transformation in 2D.

    This corresponds to the following transformation matrix `x` with scale parameters `a_yy`, `a_xx`, skew parameters `a_yx`, `a_xy` and translation parameters `t_y` and `t_x`.
    ```
    x =
    [[a_yy, a_yx, 0],
     [a_xy, a_xx, 0],
     [t_y,  t_x,  1]]
    ```
    """

    def fit(self, fps: torch.Tensor, kps: torch.Tensor):

        # fit variable parameters in first two columns
        self.x[[0, 1, 2], [0, 0, 0]] = lstsq(homogenize(fps), kps[:, 0]).solution.squeeze(-1)
        self.x[[0, 1, 2], [1, 1, 1]] = lstsq(homogenize(fps), kps[:, 1]).solution.squeeze(-1)


# collection of all transformations
all_transforms = Bunch(
    original=IdentityTransform,
    scale_translate=ScaleTranslateTransform,
    affine=AffineTransform
)
