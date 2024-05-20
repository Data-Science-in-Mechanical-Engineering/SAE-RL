import logging
from pathlib import Path
from typing import Tuple

import gymnasium as gym
import h5py
import hydra
import numpy as np
import panda_gym
import torch
from omegaconf import DictConfig, ListConfig
from rich.progress import Progress

import __init__
from src import environments
from src.environments.utils import antialias
from src.environments.wrappers import KeypointObservationWrapper
from src.utils import Bunch, check_gpu, get_display, gl, print_cfg, root

log = logging.getLogger(__name__)


def record_dataset(cfg: DictConfig, env: gym.Env) -> Tuple[torch.Tensor, torch.Tensor]:
    """Records a dataset of image sequences and corresponding keypoints for a simulated roboter task.

    Args:
        cfg (DictConfig): Hydra configuration object.
        env (gym.Env): Environment to record.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensor with image sequences with shape (SEQUENCES, FRAMES, HEIGHT, WIDTH, RGB) and tensor with keypoints with shape (SEQUENCES, FRAMES, KEYPOINTS, XY)
    """

    # empty array for all images
    images = torch.empty(
        [cfg.recording.n_sequences, cfg.recording.n_frames, cfg.environment.camera.height, cfg.environment.camera.width, 3],
        device=gl.device)

    # empty array for all keypoints
    n_keypoints = len(cfg.environment.keypoints.objects) + len(cfg.environment.keypoints.links)
    keypoints = torch.empty(
        [cfg.recording.n_sequences, cfg.recording.n_frames, n_keypoints, 2],
        device=gl.device)

    # set up flags and counters
    terminated, truncated = True, True
    i_episode = 0
    i_sequence = 0
    i_frame = 0

    # set up progress bars
    progress = Progress()
    bar = Bunch(
        sequences=progress.add_task('Sequences', total=cfg.recording.n_sequences),
        frames=progress.add_task('Frames', total=cfg.recording.n_frames)
    )

    with progress:

        # start recording
        while i_sequence < cfg.recording.n_sequences:

            # reset environment and progress bars if necessary
            if terminated or truncated:
                i_frame = 0
                progress.reset(bar.frames)
                i_episode += 1
                env.reset(seed=(cfg.recording.seed + i_episode))
                action = env.action_space.sample()

            # perform a random action
            action += np.random.normal(scale=cfg.recording.action_std, size=env.action_space.shape)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            _, _, terminated, truncated, _ = env.step(action)

            # record image and keypoints
            image = torch.tensor(env.render() / 255)
            images[i_sequence, i_frame] = antialias(image, cfg.environment.camera.antialias_factor)
            keypoints[i_sequence, i_frame] = env.get_keypoints(**cfg.environment.keypoints)

            # advance counts and progress bars
            i_frame += 1
            progress.advance(bar.frames)
            if i_frame == cfg.recording.n_frames:
                i_sequence += 1
                progress.advance(bar.sequences)
                i_frame = 0
                progress.reset(bar.frames)

    return (images, keypoints)


def setup_environment(cfg: DictConfig) -> gym.Env:
    """Sets up the environment with all necessary configuration.

    Args:
        cfg (DictConfig): Hydra configuration object.

    Returns:
        gym.Env: The set up environment.
    """

    env = gym.make(cfg.environment.id,
                   render_mode='rgb_array',
                   max_episode_steps=cfg.recording.episode_time_limit,
                   renderer=cfg.environment.camera.renderer,
                   render_height=(cfg.environment.camera.height * cfg.environment.camera.antialias_factor),
                   render_width=(cfg.environment.camera.width * cfg.environment.camera.antialias_factor),
                   render_target_position=cfg.environment.camera.target_position,
                   render_distance=cfg.environment.camera.distance,
                   render_yaw=cfg.environment.camera.yaw,
                   render_pitch=cfg.environment.camera.pitch,
                   render_roll=cfg.environment.camera.roll)

    env = KeypointObservationWrapper(env,
                                     objects=cfg.environment.keypoints.objects,
                                     links=cfg.environment.keypoints.links,
                                     **cfg.environment.camera)

    env.action_space.seed(cfg.recording.seed)

    return env


def store_dataset(cfg: DictConfig, file: Path, images: torch.Tensor, keypoints: torch.Tensor):
    """Stores images and keypoints as well as Hydra configuration items in a `.hdf5` file.

    Note: Formatting options in configuration determine whether images are saved as floats or integers, in HWC or CHW format and whether keypoints are saved in XY or YX order.

    Args:
        cfg (DictConfig): Hydra configuration object.
        file (Path): File path to write dataset to.
        images (torch.Tensor): Array with all images with shape (SEQUENCES, FRAMES, HEIGHT, WIDTH, RGB).
        keypoints (torch.Tensor): Array with all keypoints with shape (SEQUENCES, FRAMES, KEYPOINTS, XY).
    """

    if cfg.save.format.integer:  # convert float32 in [0, 1] to uint8 in [0, 255]
        images = (images * 255).to(torch.uint8)

    if cfg.save.format.channels_first:  # convert HWC to CHW
        images = torch.movedim(images, -1, -3)

    if cfg.save.format.y_first:  # convert XY to YX
        keypoints = torch.flip(keypoints, -1)

    with h5py.File(file, 'w') as f:

        # store images and keypoints as datasets
        f.create_dataset('images', data=images.cpu(), compression='gzip', compression_opts=9)
        f.create_dataset('keypoints', data=keypoints.cpu(), compression='gzip', compression_opts=9)

        # add configuration as file attributes
        store_cfg_attributes(f, cfg)


def store_cfg_attributes(f: h5py.File, cfg: DictConfig, prefix: str = None):
    """Stores all Hydra configuration items as attributes of an `.hdf5` file.

    Note: Nested configurations are stored by prefixing keys, e.g. 'env/camera/yaw'.

    Args:
        f (h5py.File): The `.hdf5` file to write attributes to.
        cfg (DictConfig): Hydra configuration object to store.
        prefix (str, optional): Prefix for all configuration keys. Defaults to None.
    """

    for key, value in cfg.items():
        # combine key with prefix for nested configs
        prefixed_key = f'{prefix}/{key}' if prefix is not None else key

        if isinstance(value, DictConfig):
            # prefix the nested config items
            store_cfg_attributes(f, value, prefix=prefixed_key)
        elif isinstance(value, ListConfig):
            # convert config lists
            f.attrs[prefixed_key] = list(value)
        else:
            # store file attribute
            f.attrs[prefixed_key] = value


@hydra.main(version_base=None, config_path='../configs', config_name='record')
def main(cfg: DictConfig):
    """Main program entry point.

    Args:
        cfg (DictConfig): Hydra configuration object.

    Raises:
        Exception: Raised if an error occurs during simulation.
    """

    print_cfg(cfg)
    check_gpu(cfg.gpu)

    # check if file already exists and cannot be replaced
    file = root / cfg.save.file
    file.parent.mkdir(parents=True, exist_ok=True)
    if not cfg.save.replace and file.exists():
        raise FileExistsError(f'{str(file)} already exists and cfg.save.replace is False.')

    # perform recording
    with get_display(cfg.display):
        log.info('Setting up environment')
        env = setup_environment(cfg)
        try:
            log.info('Recording starts')
            images, keypoints = record_dataset(cfg, env)
        except Exception as e:
            log.error('Encountered error')
            raise e
        finally:
            env.close()

    # store recording
    log.info('Storing dataset')
    store_dataset(cfg, file, images, keypoints)


if __name__ == '__main__':
    main()
