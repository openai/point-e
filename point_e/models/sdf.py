from abc import abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn

from .perceiver import SimplePerceiver
from .transformer import Transformer


class PointCloudSDFModel(nn.Module):
    @property
    @abstractmethod
    def device(self) -> torch.device:
        """
        Get the device that should be used for input tensors.
        """

    @property
    @abstractmethod
    def default_batch_size(self) -> int:
        """
        Get a reasonable default number of query points for the model.
        In some cases, this might be the only supported size.
        """

    @abstractmethod
    def encode_point_clouds(self, point_clouds: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of point clouds to cache part of the SDF calculation
        done by forward().

        :param point_clouds: a batch of [batch x 3 x N] points.
        :return: a state representing the encoded point cloud batch.
        """

    def forward(
        self,
        x: torch.Tensor,
        point_clouds: Optional[torch.Tensor] = None,
        encoded: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Predict the SDF at the coordinates x, given a batch of point clouds.

        Either point_clouds or encoded should be passed. Only exactly one of
        these arguments should be None.

        :param x: a [batch x 3 x N'] tensor of query points.
        :param point_clouds: a [batch x 3 x N] batch of point clouds.
        :param encoded: the result of calling encode_point_clouds().
        :return: a [batch x N'] tensor of SDF predictions.
        """
        assert point_clouds is not None or encoded is not None
        assert point_clouds is None or encoded is None
        if point_clouds is not None:
            encoded = self.encode_point_clouds(point_clouds)
        return self.predict_sdf(x, encoded)

    @abstractmethod
    def predict_sdf(
        self, x: torch.Tensor, encoded: Optional[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Predict the SDF at the query points given the encoded point clouds.

        Each query point should be treated independently, only conditioning on
        the point clouds themselves.
        """


class CrossAttentionPointCloudSDFModel(PointCloudSDFModel):
    """
    Encode point clouds using a transformer, and query points using cross
    attention to the encoded latents.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int = 4096,
        width: int = 512,
        encoder_layers: int = 12,
        encoder_heads: int = 8,
        decoder_layers: int = 4,
        decoder_heads: int = 8,
        init_scale: float = 0.25,
    ):
        super().__init__()
        self._device = device
        self.n_ctx = n_ctx

        self.encoder_input_proj = nn.Linear(3, width, device=device, dtype=dtype)
        self.encoder = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            layers=encoder_layers,
            heads=encoder_heads,
            init_scale=init_scale,
        )
        self.decoder_input_proj = nn.Linear(3, width, device=device, dtype=dtype)
        self.decoder = SimplePerceiver(
            device=device,
            dtype=dtype,
            n_data=n_ctx,
            width=width,
            layers=decoder_layers,
            heads=decoder_heads,
            init_scale=init_scale,
        )
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, 1, device=device, dtype=dtype)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def default_batch_size(self) -> int:
        return self.n_query

    def encode_point_clouds(self, point_clouds: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder_input_proj(point_clouds.permute(0, 2, 1))
        h = self.encoder(h)
        return dict(latents=h)

    def predict_sdf(
        self, x: torch.Tensor, encoded: Optional[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        data = encoded["latents"]
        x = self.decoder_input_proj(x.permute(0, 2, 1))
        x = self.decoder(x, data)
        x = self.ln_post(x)
        x = self.output_proj(x)
        return x[..., 0]
