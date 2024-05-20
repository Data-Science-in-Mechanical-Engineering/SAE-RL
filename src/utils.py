import os
from contextlib import nullcontext
from datetime import datetime as dt
from pathlib import Path
from typing import ContextManager

import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pyvirtualdisplay import Display
from wandb.sdk.wandb_run import Run

import wandb

root = Path(__file__).parents[1]  # root directory of the repository


class Bunch(dict):
    """Container object exposing dictionary keys as attributes.
    This allows values to be accessible by key, `bunch['value_key']`, as well as by an attribute, `bunch.value_key`.

    Note: The Bunch class is a copy from the scikit-learn implementation.
    """

    def __init__(self, **kwargs):

        super().__init__(kwargs)

    def __setattr__(self, key, value):

        self[key] = value

    def __dir__(self):

        return self.keys()

    def __getattr__(self, key):

        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


Config = Bunch


gl = Bunch(
    device='cuda'
)


def get_display(use_real_display: bool) -> ContextManager:
    """Set up a context manager for using the real display or a virtual display.

    Args:
        use_real_display (bool): Whether to use a real connected display.

    Raises:
        ValueError: Raised if real display is requested but none is available.

    Returns:
        ContextManager: Context manager to run display-dependent code in.
    """

    if use_real_display:
        if os.environ.get('DISPLAY') is not None:
            display = nullcontext()  # use the actual display
        else:
            raise ValueError('Use of real display requested but no real display is available.')
    else:
        display = Display()  # create a virtual display

    return display


def print_cfg(cfg: DictConfig):
    """Prints a `.yaml` representation of the Hydra configuration object.

    Args:
        cfg (DictConfig): Hydra configuration object.
    """

    print(OmegaConf.to_yaml(cfg))


def is_rerun() -> bool:
    """Determines whether the current run is a rerun and logs the run start time to a file.

    Reruns are started with Hydra's `--experimental-rerun path/to/run/.hydra/config.pickle` option.

    Returns:
        bool: Whether the current run is a rerun of a previous one.
    """

    hydra_cfg = HydraConfig.get()
    run_dir = hydra_cfg.runtime.output_dir
    output_subdir = hydra_cfg.output_subdir
    runs_logfile = root / run_dir / output_subdir / 'runs.log'

    rerun = True if runs_logfile.exists() else False

    with open(runs_logfile, 'a+') as f:
        f.write(f'{dt.now()}\n')

    return rerun


def setup_wandb(cfg: DictConfig, rerun: bool) -> Run:
    """Sets up Weights and Biases logging.

    Args:
        cfg (DictConfig): Hydra configuration object.
        rerun (bool): Whether this is a resumed run to be continued.

    Returns:
        Run: The Weights and Biases run object.
    """

    if cfg.wandb.enabled:
        if not rerun:  # normal run
            run = wandb.init(
                entity=cfg.wandb.entity, project=cfg.wandb.project,
                config=OmegaConf.to_container(cfg),
                sync_tensorboard=True
            )
            store_wandb_id(run)

        else:  # run is resumed
            wandb_resume_id = get_wandb_resume_id()
            run = wandb.init(
                entity=cfg.wandb.entity, project=cfg.wandb.project,
                id=wandb_resume_id, resume='allow',
                sync_tensorboard=True
            )
    else:
        run = None

    return run


def get_hydra_run_dir() -> Path:
    """Determines the current Hydra run's output directory.

    Returns:
        Path: The run directory.
    """

    hydra_cfg = HydraConfig.get()
    dir = root / hydra_cfg.runtime.output_dir

    return dir


def store_wandb_id(run: Run):
    """Store the name and id of the current Weights and Biases run in the run directory.

    Stored file is called `random-name-42.wandb` and contains the run id only.

    Args:
        run (Run): The current Weights and Biases run object.
    """

    id_file = get_hydra_run_dir() / f'{run.name}.wandb'
    with open(id_file, 'w') as f:
        f.write(run.id)


def get_wandb_resume_id() -> str:
    """Reads the id of the Weights and Biases run to be resumed from the `.wandb` file in the run directory.

    Returns:
        str: The resumed Weights and Biases run's id.
    """

    id_files = [f for f in get_hydra_run_dir().iterdir() if f.suffix == '.wandb']
    if len(id_files) == 0:
        raise FileNotFoundError('Could not find a .wandb file for resuming.')

    with open(id_files[0], 'r') as f:
        resume_id = f.readline()

    return resume_id


def deflate(t: torch.Tensor) -> torch.Tensor:
    """Deflates a tensor, i.e. returns a view on the same tensor with the first two dimensions flattened into one dimension.

    E.g. a tensor with shape (2, 3, 4, 5) gets deflated to shape (6, 4, 5).

    Args:
        t (torch.Tensor): Tensor to be deflated.

    Returns:
        torch.Tensor: The deflated tensor view.
    """

    _, _, *dims = t.shape

    return t.view(-1, *dims)


def inflate(t: torch.Tensor, slices: int) -> torch.Tensor:
    """Inflates a tensor, i.e. returns a view on the same tensor with the first dimension stacked into two dimensions.

    E.g. a tensor with shape (6, 4, 5) gets inflated to shape (2, 3, 4, 5) with `slices=2`.

    Args:
        t (torch.Tensor): Tensor to be inflated.
        slices (int): Size of new first dimension. Previous first dimension must be divisible by this number.

    Returns:
        torch.Tensor: The inflated tensor view.
    """

    _, *dims = t.shape

    return t.view(slices, -1, *dims)


def homogenize(t: torch.Tensor) -> torch.Tensor:
    """Transforms given vectors with normal coordinates to homogenous coordinates by appending `1.0` to every vector along the last tensor dimension.

    Args:
        t (torch.Tensor): Tensor of inhomogeneous input vectors to homogenize.

    Returns:
        torch.Tensor: Tensor of homogeneous vectors.
    """

    device = t.device

    output = torch.cat([t, torch.ones((*t.shape[:-1], 1), device=device)], dim=-1)

    return output


def dehomogenize(t: torch.Tensor) -> torch.Tensor:
    """Tansforms given vectors with homogeneous coordinates to normal coordinates by dividing by the final entry of every vector along the last tensor dimension.

    Args:
        t (torch.Tensor): Tensor of homogeneous input vectors to dehomogenize.

    Returns:
        torch.Tensor: Tensor of inhomogeneous vectors.
    """

    output = t[..., :-1] / t[..., [-1]]

    return output


def check_gpu(enable_gpu: bool = True):
    """Sets the global device to either 'cuda' or 'mps' if a GPU is requested and available, or 'cpu'.

    Args:
        enable_gpu (bool, optional): Whether to enable GPU support. Defaults to True.

    Raises:
        Exception: Raised if GPU is requested but no GPU device can be found.
    """

    global gl
    if enable_gpu:
        if torch.cuda.is_available():
            gl.device = 'cuda'
        elif torch.has_mps:
            gl.device = 'mps'
        else:
            raise Exception("GPU not found, pass gpu=False to run on CPU only")
    else:
        gl.device = 'cpu'


def current_datetime_str() -> str:
    """Formats the current date and time as a string with format YYYY-MM-DD_HH-MM-SS.

    Returns:
        str: The formatted date and time.
    """

    time = dt.now().strftime('%Y-%m-%d_%H-%M-%S')

    return time
