import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..utils import gl
from . import decoders, encoders


class SAE(nn.Module):
    """Spatial Autoencoder (SAE) class, aggregating an encoder model and a decoder model.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        """Initializes an SAE by storing initialized encoder and decoder parts.

        Args:
            encoder (nn.Module): Encoder model.
            decoder (nn.Module): Decoder model.
        """

        super().__init__()

        # store models as attributes
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.encoder(x)
        x = self.decoder(x)

        return x


def assemble_sae(cfg: DictConfig) -> SAE:
    """Assembles SAE by configuring an encoder, decoder, and spatial softargmax layer according to passed model and data configurations.

    Args:
        cfg (DictConfig): Hydra configuration object.

    Returns:
        SAE: The assembled SAE model.
    """

    # build encoder
    encoder = getattr(encoders, cfg.model.encoder.class_name)
    encoder = encoder(**cfg.model.encoder.settings)

    # build decoder
    decoder = getattr(decoders, cfg.model.decoder.class_name)
    decoder = decoder(n_coordinates=cfg.model.encoder.settings.n_coordinates,
                      target_shape=cfg.dataset.target_shape,
                      **cfg.model.decoder.settings)

    # assemble complete SAE
    model = SAE(encoder, decoder).to(gl.device)

    return model
