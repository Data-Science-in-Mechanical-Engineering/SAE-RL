import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.tensorboard.summary import hparams
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import draw_keypoints, make_grid

from ..utils import Bunch, get_hydra_run_dir, gl


class Logger():
    """Logger class handling logging of training statistics to TensorBoard and saving model checkpoints.
    """

    def setup(self, cfg: DictConfig, epoch: int = 0):
        """Initializes logger class with the passed configuration.

        Args:
            cfg (DictConfig): Hydra configuration object.
        """

        # store configurations
        self.cfg = cfg

        # store logging directory reference
        self.log_dir = get_hydra_run_dir() / 'tb'

        # set up TensorBoard writer
        self.writer = self.setup_summary_writer()
        self.global_step = epoch

    def log_losses(self, tag: str, loss: float, loss_rec: float, loss_vel: float):
        """Logs the reconstruction, velocity and total loss in TensorBoard.

        Args:
            tag (str): Label to prefix loss with, usually 'train' or 'valid.
            loss (float): The mean epoch loss.
            loss_rec (float): The mean epoch reconstruction loss.
            loss_vel (float): The mean epoch velocity loss.
        """

        # log all three losses as scalars for every epoch
        self.writer.add_scalar(f'{tag}/loss', loss, self.global_step)
        self.writer.add_scalar(f'{tag}/loss_rec', loss_rec, self.global_step)
        self.writer.add_scalar(f'{tag}/loss_vel', loss_vel, self.global_step)

    def log_images(self, inputs: torch.Tensor, targets: torch.Tensor, reconstructions: torch.Tensor, sites: torch.Tensor, feature_points: torch.Tensor, max_images: int = 9):
        """Logs sample images for inputs, targets, reconstructions and feature points over time on TensorBoard.

        Args:
            inputs (torch.Tensor): Batch of input images with shape (BATCH, FRAMES, CHANNELS, HEIGHT, WIDTH).
            targets (torch.Tensor): Batch of target images with shape (BATCH, FRAMES, CHANNELS, HEIGHT, WIDTH).
            reconstructions (torch.Tensor): Batch of reconstructed images with shape (BATCH, FRAMES, CHANNELS, HEIGHT, WIDTH).
            sites (torch.Tensor): Batch of ground truth feature point sites with shape (BATCH, FRAMES, SITES, YX).
            feature_points (torch.Tensor): Batch of feature point locations with shape (BATCH, FRAMES, COORDS, YX).
            max_images (int, optional): How many sample images to log. Defaults to 9.
        """

        # make a random but constant selection of image samples
        np.random.seed(self.cfg.dataset.seed)
        selection = np.random.choice(len(inputs), min(max_images, len(inputs)), replace=False)
        nrow = int(np.ceil(np.sqrt(len(selection))))  # number of images per row

        # log input and target images once
        if self.global_step == 0:

            # input images
            input_grid = make_grid(inputs[selection, 0], nrow=nrow, padding=0)
            self.writer.add_image('inputs', input_grid)

            # target images
            target_grid = make_grid(targets[selection, 0], nrow=nrow, padding=0)
            self.writer.add_image('targets', target_grid)

        # log reconstruction images for every epoch
        reconstruction_grid = make_grid(
            torch.clip(reconstructions[selection, 0], min=0.0, max=1.0), nrow=nrow, padding=0)
        self.writer.add_image('reconstructions', reconstruction_grid, self.global_step)

        # log feature point images for every epoch
        hw = torch.tensor(inputs.shape[-2:], device=gl.device) - 1  # height and width
        fp_images = (inputs[selection, 0] * 255).to(torch.uint8)
        plot_sites = ((sites[selection, 0] + 1) * (hw / 2)).unsqueeze(dim=-2).flip(dims=[-1])
        plot_fps = ((feature_points[selection, 0] + 1) * (hw / 2)).unsqueeze(dim=-3).flip(dims=[-1])
        for i in range(len(fp_images)):
            fp_images[i] = draw_keypoints(fp_images[i], plot_sites[i], colors='white', radius=3)
            fp_images[i] = draw_keypoints(fp_images[i], plot_fps[i], colors='red', radius=2)
        feature_point_grid = make_grid(fp_images, nrow=nrow, padding=0)
        self.writer.add_image('feature points', feature_point_grid, self.global_step)

    def log_tracking_errors(self, tracking_errors: Bunch):
        """Logs tracking errors for each keypoint and different transformations.

        Args:
            tracking_errors (Bunch): Bunch of transforms with corresponding lists of tracking errors for all objects.
        """

        # log tracking errors
        for transform, errors in tracking_errors.items():
            for i, e in enumerate(errors):
                # use global step as keypoint identifier instead
                self.writer.add_scalar(f'tracking_errors/{transform}/{i}', e, self.global_step)

    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, tag: str = None):
        """Saves a training checkpoint dictionary with the current model state, optimizer state, and epoch.

        Args:
            model (nn.Module): Model to save the current state of.
            optimizer (torch.optim.Optimizer): Optimizer to save the current state of
            epoch (int): Current training epoch.
            tag (str, optional): Additional text label for the saved checkpoint. Defaults to None.
        """

        # assemble filename
        filename = 'checkpoint'
        if tag is not None:
            filename += '_' + tag
        filename += '.pth'

        # save checkpoint dictionary
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'cfg': self.cfg
        }, self.log_dir.parent / filename)

    def load_checkpoint(self, cfg: DictConfig, checkpoint_file: Path) -> Bunch:
        """Loads a stored checkpoint file and sets the logger up to resume training from this point.

        Args:
            cfg (DictConfig): Hydra configuration object
            checkpoint_file (Path): The checkpoint file to load the stored state from.

        Returns:
            Bunch: A bunch with all saved checkpoint parameters.
        """

        # load checkpoint dictionary
        checkpoint = Bunch(**torch.load(checkpoint_file, map_location=gl.device))

        # set up logger
        self.setup(cfg, epoch=(checkpoint.epoch + 1))

        return checkpoint

    def setup_summary_writer(self) -> SummaryWriter:
        """Sets up and returns a TensorBoard summary writer.

        Note: The `add_hparams(...)` routine is overwritten following [this proposal](https://discuss.pytorch.org/t/how-to-add-graphs-to-hparams-in-tensorboard/109349/2) so that metrics can be logged continuously.

        Returns:
            SummaryWriter: The set up TensorBoard summary writer.
        """

        # instantiate default summary writer
        writer = SummaryWriter(log_dir=self.log_dir)

        # modified add_hparams routine to enable continuous metric logging
        def add_hparams(self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None):
            torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
            if type(hparam_dict) is not dict or type(metric_dict) is not dict:
                raise TypeError("hparam_dict and metric_dict should be dictionary.")
            exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

            self.file_writer.add_summary(exp)
            self.file_writer.add_summary(ssi)
            self.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                if v is not None:
                    self.add_scalar(k, v)

        # replace original add_hparams routine with the modified one
        writer.add_hparams = types.MethodType(add_hparams, writer)

        return writer

    def close(self):
        """Closes TensorBoard summary writer.
        """

        self.writer.close()
