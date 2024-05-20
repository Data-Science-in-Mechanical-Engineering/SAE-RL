import logging

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

import __init__
from src.data import load
from src.data.loading import ConstantRandomSampler
from src.logging import Logger
from src.metrics.losses import reconstruction_loss, velocity_loss
from src.metrics.tracking import all_tracking_errors
from src.models.sae import SAE, assemble_sae
from src.utils import check_gpu, deflate, get_hydra_run_dir, gl, inflate, is_rerun, print_cfg, setup_wandb

log = logging.getLogger(__name__)


def train_sae(cfg: DictConfig, model: SAE, optimizer: torch.optim.Optimizer, dataset_train: Dataset, dataset_valid: Dataset, logger: Logger, start_epoch: int = 0):
    """Runs main training loop over epochs, including validation.

    Args:
        cfg (DictConfig): Hydra configuration object.
        model (SAE): The model instance to train.
        optimizer (torch.optim.Optimizer): The optimizer instance to use.
        dataset_train (Dataset): The training dataset.
        dataset_valid (Dataset): The validation dataset.
        logger (Logger): The logger instance to use for TensorBoard logging and checkpoint saving.
        start_epoch (int, optional): Epoch to start or resume training at. Defaults to 0.

    Raises:
        e: KeyboardInterrupt or Exception raised after checkpoint saving if any error occurs.
    """

    # initialize data batch loaders
    loader_train = DataLoader(dataset_train, cfg.training.batch_size,
                              shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    loader_valid = DataLoader(dataset_valid, cfg.training.batch_size,
                              sampler=ConstantRandomSampler(dataset_valid, cfg.dataset.seed),
                              shuffle=False, drop_last=True, num_workers=8, pin_memory=True)
    min_valid_loss = np.inf

    try:
        # loop over training epochs
        for epoch in trange(start_epoch, (start_epoch + cfg.training.epochs), leave=False):

            logger.global_step = epoch

            train_epoch(cfg, model, loader_train, optimizer, logger)

            if cfg.training.evaluation.frequency > 0:
                if ((epoch + 1) % cfg.training.evaluation.frequency) == 0:
                    valid_loss = valid_epoch(cfg, model, loader_valid, logger)
                    if valid_loss <= min_valid_loss:
                        min_valid_loss = valid_loss
                        log.info('Saving best checkpoint')
                        logger.save_checkpoint(model, optimizer, epoch, tag='best')

        # save trained model checkpoint
        log.info('Saving checkpoint')
        logger.save_checkpoint(model, optimizer, epoch, tag='final')
        logger.close()

    except (KeyboardInterrupt, Exception) as e:
        # save current model checkpoint before exiting
        log.error('Encountered error')
        log.info('Saving checkpoint')
        logger.save_checkpoint(model, optimizer, epoch, tag=e.__class__.__name__)
        logger.close()

        raise e


def train_epoch(cfg: DictConfig, model: SAE, data_loader: DataLoader, optimizer: torch.optim.Optimizer, logger: Logger):
    """Runs training loop over batches for one epoch.

    Args:
        cfg (DictConfig): Hydra configuration object.
        model (SAE): The model instance to train.
        data_loader (DataLoader): The training dataset batch loader.
        optimizer (torch.optim.Optimizer): The optimizer instance to use.
        logger (Logger): The logger instance to use for TensorBoard logging.
    """

    model.train()  # put model into training state
    epoch_losses = torch.zeros(3, device=gl.device)  # total loss, reconstruction loss, velocity loss

    # loop over training batches
    for batch, (inputs, targets) in enumerate(tqdm(data_loader, leave=False)):

        # move all data to GPU
        inputs = inputs.to(gl.device)
        targets = targets.to(gl.device)

        optimizer.zero_grad(set_to_none=True)

        # encoder pass to obtain feature points
        fps = model.encoder(deflate(inputs))
        feature_points = inflate(fps, len(inputs))

        # decoder pass to obtain reconstructions
        recs = model.decoder(fps)
        reconstructions = inflate(recs, len(inputs))

        # loss computation
        loss_rec = reconstruction_loss(reconstructions, targets)
        loss_rec *= cfg.training.loss_weights.reconstruction

        loss_vel = velocity_loss(feature_points)
        loss_vel *= cfg.training.loss_weights.velocity

        loss = loss_rec + loss_vel

        # gradient backpropagation and optimizer step
        loss.backward()
        optimizer.step()

        # cumulate epoch losses
        epoch_losses += torch.stack([loss, loss_rec, loss_vel])

    epoch_losses /= (batch + 1)
    logger.log_losses('train', *epoch_losses)


@torch.no_grad()
def valid_epoch(cfg: DictConfig, model: SAE, data_loader: DataLoader, logger: Logger) -> float:
    """Runs validation loop over batches for one epoch.

    Args:
        cfg (DictConfig): Hydra configuration object.
        model (SAE): The model instance to validate with. 
        data_loader (DataLoader): The validation dataset batch loader.
        logger (Logger): The logger instance to use for TensorBoard logging.

    Returns:
        float: Validation loss.
    """

    model.eval()  # put model into evaluation state
    epoch_losses = torch.zeros(3, device=gl.device)  # total loss, reconstruction loss, velocity loss
    track_fps = []  # feature points for tracking error computation
    track_sites = []  # site coordinates for tracking error computation

    # loop over validation batches
    for batch, (inputs, targets, sites) in enumerate(tqdm(data_loader, leave=False)):

        # move all data to GPU
        inputs = inputs.to(gl.device)
        targets = targets.to(gl.device)
        sites = sites.to(gl.device)

        # encoder pass to obtain feature points
        fps = model.encoder(deflate(inputs))
        feature_points = inflate(fps, len(inputs))

        # decoder pass to obtain reconstructions
        recs = model.decoder(fps)
        reconstructions = inflate(recs, len(inputs))

        # storing fps and sites for first image of each snippet (avoiding duplicates)
        track_fps.append(feature_points[:, 0])
        track_sites.append(sites[:, 0])

        # loss computation
        loss_rec = reconstruction_loss(reconstructions, targets)
        loss_rec *= cfg.training.loss_weights.reconstruction

        loss_vel = velocity_loss(feature_points)
        loss_vel *= cfg.training.loss_weights.velocity

        loss = loss_rec + loss_vel

        # cumulate epoch losses
        epoch_losses += torch.stack([loss, loss_rec, loss_vel])

    epoch_losses /= (batch + 1)
    logger.log_losses('valid', *epoch_losses)

    logger.log_images(inputs, targets, reconstructions, sites, feature_points)

    track_fps = torch.cat(track_fps)
    track_sites = torch.cat(track_sites)
    tracking_errors = all_tracking_errors(track_fps.cpu(), track_sites.cpu())
    logger.log_tracking_errors(tracking_errors)

    return epoch_losses[0].item()


@hydra.main(version_base=None, config_path='../configs', config_name='train_sae')
def main(cfg: DictConfig):
    """Main program entry point.

    Args:
        cfg (DictConfig): Hydra configuration object.
    """

    print_cfg(cfg)
    check_gpu(cfg.gpu)
    rerun = is_rerun()
    run = setup_wandb(cfg, rerun)

    if not rerun:

        # load datasets
        log.info('Loading training and validation datasets')
        dataset_train, dataset_valid = load(cfg, train=True, valid=True)

        # set up logging
        logger = Logger()
        logger.setup(cfg)

        # initialize model and optimizer
        log.info('Setting up model')
        model = assemble_sae(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

        # begin training
        log.info('Training starts')
        train_sae(cfg, model, optimizer, dataset_train, dataset_valid, logger)

    else:  # resume run
        log.info('Resuming previous run')

        # set up logger and load checkpoint
        logger = Logger()
        log.info('Loading checkpoint file')
        checkpoint = logger.load_checkpoint(cfg, get_hydra_run_dir() / 'checkpoint_final.pth')

        # load datasets
        log.info('Loading training and validation datasets')
        dataset_train, dataset_valid = load(cfg, train=True, valid=True)

        # reinstantiate model and optimizer
        log.info('Setting up model')
        model = assemble_sae(cfg)
        model.load_state_dict(checkpoint.model_state_dict)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)

        # resume training
        log.info('Training starts')
        train_sae(cfg, model, optimizer, dataset_train, dataset_valid, logger, start_epoch=(checkpoint.epoch + 1))


if __name__ == '__main__':
    main()
