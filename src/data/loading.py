from typing import Callable, Iterator, Optional, Sized, Tuple

import h5py
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision.transforms import Compose

from ..utils import deflate, inflate, root
from .transforms import (BackgroundSegmentation, IdentityTransform, LocalContrastNormalization, NoiseTransform,
                         ResizeTransform)


class SnippetDataset(Dataset):
    """Dataset class holding a set of image sequences and returning short snippets from them.
    """

    def __init__(self, input_sequences: torch.Tensor, sites: torch.Tensor = None, snippet_len: int = 3, input_transform: Callable = IdentityTransform(), target_transform: Callable = IdentityTransform(), preprocess_target_transform: bool = True):
        """Initializes snippet dataset by calculating all dimensions, storing input images and transforming and storing target images.

        Args:
            input_sequences (torch.Tensor): Tensor of input images with shape (SEQUENCES, FRAMES, RGB, HEIGHT, WIDTH).
            sites (torch.Tensor, optional): Tensor of coordinates of sites of interest with shape (SEQUENCES, FRAMES, SITES, YX). Defaults to None.
            snippet_len (int, optional): Number of frames for a video snippet training sample. Defaults to 3.
            input_transform (Callable, optional): Transformation to apply to input snippets when loading them. Defaults to IdentityTransform().
            target_transform (Callable, optional): Transformation to apply to target snippets, either preprocessed or when loading them. Defaults to IdentityTransform().
            preprocess_target_transform (bool, optional): Whether to preprocess target transformation on dataset creation or later when loading snippets. Defaults to True.
        """

        super().__init__()

        # store attributes
        self.input_sequences = input_sequences
        self.sites = sites
        self.snippet_len = snippet_len
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.preprocess_target_transform = preprocess_target_transform

        # get sizes
        self.n_sequences, self.n_frames, *_ = self.input_sequences.shape
        self.n_snippets = self.n_frames - self.snippet_len + 1

        # preprocess target images if desired
        if self.preprocess_target_transform:
            transformed = self.target_transform(deflate(self.input_sequences))
            self.target_sequences = inflate(transformed, self.n_sequences)

    def __len__(self) -> int:
        """Calculates the dataset length as the total number of video snippet samples.

        Returns:
            int: Dataset length.
        """

        return self.n_sequences * self.n_snippets

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Accesses an input snippet with the corresponding target snippets, including the site coordinates if available.

        Args:
            idx (int): Index of sample to access

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: Input and target snippet, including site coordinates if available.
        """

        # get sequence and frame indices, prepare return object
        i_sequence, i_frame = divmod(idx, self.n_snippets)
        sample = ()

        # get input snippet and transform it
        input_snippet = self.input_transform(
            self.input_sequences[i_sequence, i_frame:(i_frame+self.snippet_len)]
        )
        sample += (input_snippet, )

        # get target snippet or transform it from input snippet if needed
        if self.preprocess_target_transform:
            target_snippet = self.target_sequences[i_sequence, i_frame:(i_frame+self.snippet_len)]
        else:
            target_snippet = self.target_transform(
                self.input_sequences[i_sequence, i_frame:(i_frame+self.snippet_len)]
            )
        sample += (target_snippet, )

        if self.sites is not None:
            sites_snippet = self.sites[i_sequence, i_frame:(i_frame+self.snippet_len)]
            sample += (sites_snippet, )

        return sample


class ConstantRandomSampler(Sampler[int]):
    """Random sampler shuffling data once and then always yielding it in the same order.
    """

    def __init__(self, data_source: Sized, seed: int) -> None:
        """Initializes constant random sampler.

        Args:
            data_source (Sized): Dataset to sample from.
            seed (int): Random seed for initial shuffling permutation.
        """

        self.data_source = data_source
        self.num_samples = len(self.data_source)

        np.random.seed(seed)
        self.shuffled_list = np.random.permutation(self.num_samples).tolist()

    def __iter__(self) -> Iterator[int]:

        yield from self.shuffled_list

    def __len__(self) -> int:

        return self.num_samples


def load(cfg: DictConfig, train=False, valid=False, test=False, shuffle=True) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
    """Loads dataset instances for training, validation and test sets based on the given config.

    Args:
        cfg (DictConfig): Hydra configuration object.
        train (bool, optional): Whether to load the training set. Defaults to False.
        valid (bool, optional): Whether to load the validation set. Defaults to False.
        test (bool, optional): Whether to load the test set. Defaults to False.
        shuffle (bool, optional): Whether to shuffle the dataset before splitting into train, valid and test sets. Defaults to True.

    Returns:
        Tuple[Dataset, Optional[Dataset], Optional[Dataset]]: The requested datasets.
    """

    datasets = ()

    if train or valid or test:

        if '.hdf5' in cfg.dataset.file:
            with h5py.File(root / cfg.dataset.directory / cfg.dataset.file, 'r') as data:
                # load hdf5 data arrays
                # PyTorch expects RGB dimension before height and width
                sequences = np.moveaxis(data['images'][:], -1, -3)
                # PyTorch expects image height in Y to precede image width in X
                sites = np.flip(data['keypoints'][:], axis=-1)
        else:
            with np.load(root / cfg.dataset.directory / cfg.dataset.file) as data:
                # load numpy data arrays
                # PyTorch expects RGB dimension before height and width
                sequences = np.moveaxis(data['frames'], -1, -3)
                # PyTorch expects image height in Y to precede image width in X
                sites = np.flip(data['sites'], axis=-1)

        # randomly distribute dataset samples
        n_sequences = len(sequences)
        split = cfg.dataset.split
        if shuffle:
            np.random.seed(cfg.dataset.seed)
            idcs = np.random.choice(n_sequences, sum(split), replace=False)
        else:
            idcs = np.arange(sum(split))
        idcs_train = idcs[:split[0]]
        idcs_valid = idcs[split[0]:sum(split[:2])]
        idcs_test = idcs[sum(split[:2]):sum(split)]

        # prepare input and target transformations
        input_transform = NoiseTransform(cfg.training.noise_std)
        target_transforms = []
        if cfg.training.bg_segmentation_threshold is not None:
            median_image = torch.tensor(sequences).view([-1, *sequences.shape[-3:]]).median(dim=0).values
            target_transforms.append(BackgroundSegmentation(
                median_image, cfg.training.bg_segmentation_threshold, cfg.training.bg_segmentation_output))
        target_transforms.append(ResizeTransform(cfg.dataset.target_shape))
        if cfg.training.contrast_norm_std is not None:
            target_transforms.append(LocalContrastNormalization(stddev=cfg.training.contrast_norm_std))
        target_transform = Compose(target_transforms)

        if train:  # initialize training dataset
            train_dataset = SnippetDataset(
                torch.tensor(sequences[idcs_train]),
                snippet_len=cfg.dataset.snippet_length,
                input_transform=input_transform,
                target_transform=target_transform
            )
            datasets += (train_dataset, )

        if valid:  # initialize validation dataset
            valid_dataset = SnippetDataset(
                torch.tensor(sequences[idcs_valid]),
                torch.tensor(sites[idcs_valid]),
                snippet_len=cfg.dataset.snippet_length,
                target_transform=target_transform
            )
            datasets += (valid_dataset, )

        if test:  # initialize test dataset
            test_dataset = SnippetDataset(
                torch.tensor(sequences[idcs_test]),
                torch.tensor(sites[idcs_test]),
                snippet_len=cfg.dataset.snippet_length,
                target_transform=target_transform
            )
            datasets += (test_dataset, )

    return datasets
