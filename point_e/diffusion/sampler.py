"""
Helpers for sampling from a single- or multi-stage point cloud diffusion model.
"""

from typing import Any, Callable, Dict, Iterator, List, Sequence, Tuple

import torch
import torch.nn as nn

from point_e.util.point_cloud import PointCloud

from .gaussian_diffusion import GaussianDiffusion
from .k_diffusion import karras_sample_progressive


class PointCloudSampler:
    """
    A wrapper around a model or stack of models that produces conditional or
    unconditional sample tensors.

    By default, this will load models and configs from files.
    If you want to modify the sampler arguments of an existing sampler, call
    with_options() or with_args().
    """

    def __init__(
        self,
        device: torch.device,
        models: Sequence[nn.Module],
        diffusions: Sequence[GaussianDiffusion],
        num_points: Sequence[int],
        aux_channels: Sequence[str],
        model_kwargs_key_filter: Sequence[str] = ("*",),
        guidance_scale: Sequence[float] = (3.0, 3.0),
        clip_denoised: bool = True,
        use_karras: Sequence[bool] = (True, True),
        karras_steps: Sequence[int] = (64, 64),
        sigma_min: Sequence[float] = (1e-3, 1e-3),
        sigma_max: Sequence[float] = (120, 160),
        s_churn: Sequence[float] = (3, 0),
    ):
        n = len(models)
        assert n > 0

        if n > 1:
            if len(guidance_scale) == 1:
                # Don't guide the upsamplers by default.
                guidance_scale = list(guidance_scale) + [1.0] * (n - 1)
            if len(use_karras) == 1:
                use_karras = use_karras * n
            if len(karras_steps) == 1:
                karras_steps = karras_steps * n
            if len(sigma_min) == 1:
                sigma_min = sigma_min * n
            if len(sigma_max) == 1:
                sigma_max = sigma_max * n
            if len(s_churn) == 1:
                s_churn = s_churn * n
            if len(model_kwargs_key_filter) == 1:
                model_kwargs_key_filter = model_kwargs_key_filter * n
        if len(model_kwargs_key_filter) == 0:
            model_kwargs_key_filter = ["*"] * n
        assert len(guidance_scale) == n
        assert len(use_karras) == n
        assert len(karras_steps) == n
        assert len(sigma_min) == n
        assert len(sigma_max) == n
        assert len(s_churn) == n
        assert len(model_kwargs_key_filter) == n

        self.device = device
        self.num_points = num_points
        self.aux_channels = aux_channels
        self.model_kwargs_key_filter = model_kwargs_key_filter
        self.guidance_scale = guidance_scale
        self.clip_denoised = clip_denoised
        self.use_karras = use_karras
        self.karras_steps = karras_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.s_churn = s_churn

        self.models = models
        self.diffusions = diffusions

    @property
    def num_stages(self) -> int:
        return len(self.models)

    def sample_batch(self, batch_size: int, model_kwargs: Dict[str, Any]) -> torch.Tensor:
        samples = None
        for x in self.sample_batch_progressive(batch_size, model_kwargs):
            samples = x
        return samples

    def sample_batch_progressive(
        self, batch_size: int, model_kwargs: Dict[str, Any]
    ) -> Iterator[torch.Tensor]:
        samples = None
        for (
            model,
            diffusion,
            stage_num_points,
            stage_guidance_scale,
            stage_use_karras,
            stage_karras_steps,
            stage_sigma_min,
            stage_sigma_max,
            stage_s_churn,
            stage_key_filter,
        ) in zip(
            self.models,
            self.diffusions,
            self.num_points,
            self.guidance_scale,
            self.use_karras,
            self.karras_steps,
            self.sigma_min,
            self.sigma_max,
            self.s_churn,
            self.model_kwargs_key_filter,
        ):
            stage_model_kwargs = model_kwargs.copy()
            if stage_key_filter != "*":
                use_keys = set(stage_key_filter.split(","))
                stage_model_kwargs = {k: v for k, v in stage_model_kwargs.items() if k in use_keys}
            if samples is not None:
                stage_model_kwargs["low_res"] = samples
            if hasattr(model, "cached_model_kwargs"):
                stage_model_kwargs = model.cached_model_kwargs(batch_size, stage_model_kwargs)
            sample_shape = (batch_size, 3 + len(self.aux_channels), stage_num_points)

            if stage_guidance_scale != 1 and stage_guidance_scale != 0:
                for k, v in stage_model_kwargs.copy().items():
                    stage_model_kwargs[k] = torch.cat([v, torch.zeros_like(v)], dim=0)

            if stage_use_karras:
                samples_it = karras_sample_progressive(
                    diffusion=diffusion,
                    model=model,
                    shape=sample_shape,
                    steps=stage_karras_steps,
                    clip_denoised=self.clip_denoised,
                    model_kwargs=stage_model_kwargs,
                    device=self.device,
                    sigma_min=stage_sigma_min,
                    sigma_max=stage_sigma_max,
                    s_churn=stage_s_churn,
                    guidance_scale=stage_guidance_scale,
                )
            else:
                internal_batch_size = batch_size
                if stage_guidance_scale:
                    model = self._uncond_guide_model(model, stage_guidance_scale)
                    internal_batch_size *= 2
                samples_it = diffusion.p_sample_loop_progressive(
                    model,
                    shape=(internal_batch_size, *sample_shape[1:]),
                    model_kwargs=stage_model_kwargs,
                    device=self.device,
                    clip_denoised=self.clip_denoised,
                )
            for x in samples_it:
                samples = x["pred_xstart"][:batch_size]
                if "low_res" in stage_model_kwargs:
                    samples = torch.cat(
                        [stage_model_kwargs["low_res"][: len(samples)], samples], dim=-1
                    )
                yield samples

    @classmethod
    def combine(cls, *samplers: "PointCloudSampler") -> "PointCloudSampler":
        assert all(x.device == samplers[0].device for x in samplers[1:])
        assert all(x.aux_channels == samplers[0].aux_channels for x in samplers[1:])
        assert all(x.clip_denoised == samplers[0].clip_denoised for x in samplers[1:])
        return cls(
            device=samplers[0].device,
            models=[x for y in samplers for x in y.models],
            diffusions=[x for y in samplers for x in y.diffusions],
            num_points=[x for y in samplers for x in y.num_points],
            aux_channels=samplers[0].aux_channels,
            model_kwargs_key_filter=[x for y in samplers for x in y.model_kwargs_key_filter],
            guidance_scale=[x for y in samplers for x in y.guidance_scale],
            clip_denoised=samplers[0].clip_denoised,
            use_karras=[x for y in samplers for x in y.use_karras],
            karras_steps=[x for y in samplers for x in y.karras_steps],
            sigma_min=[x for y in samplers for x in y.sigma_min],
            sigma_max=[x for y in samplers for x in y.sigma_max],
            s_churn=[x for y in samplers for x in y.s_churn],
        )

    def _uncond_guide_model(
        self, model: Callable[..., torch.Tensor], scale: float
    ) -> Callable[..., torch.Tensor]:
        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.chunk(eps, 2, dim=0)
            half_eps = uncond_eps + scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)

        return model_fn

    def split_model_output(
        self,
        output: torch.Tensor,
        rescale_colors: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert (
            len(self.aux_channels) + 3 == output.shape[1]
        ), "there must be three spatial channels before aux"
        pos, joined_aux = output[:, :3], output[:, 3:]

        aux = {}
        for i, name in enumerate(self.aux_channels):
            v = joined_aux[:, i]
            if name in {"R", "G", "B", "A"}:
                v = v.clamp(0, 255).round()
                if rescale_colors:
                    v = v / 255.0
            aux[name] = v
        return pos, aux

    def output_to_point_clouds(self, output: torch.Tensor) -> List[PointCloud]:
        res = []
        for sample in output:
            xyz, aux = self.split_model_output(sample[None], rescale_colors=True)
            res.append(
                PointCloud(
                    coords=xyz[0].t().cpu().numpy(),
                    channels={k: v[0].cpu().numpy() for k, v in aux.items()},
                )
            )
        return res

    def with_options(
        self,
        guidance_scale: float,
        clip_denoised: bool,
        use_karras: Sequence[bool] = (True, True),
        karras_steps: Sequence[int] = (64, 64),
        sigma_min: Sequence[float] = (1e-3, 1e-3),
        sigma_max: Sequence[float] = (120, 160),
        s_churn: Sequence[float] = (3, 0),
    ) -> "PointCloudSampler":
        return PointCloudSampler(
            device=self.device,
            models=self.models,
            diffusions=self.diffusions,
            num_points=self.num_points,
            aux_channels=self.aux_channels,
            model_kwargs_key_filter=self.model_kwargs_key_filter,
            guidance_scale=guidance_scale,
            clip_denoised=clip_denoised,
            use_karras=use_karras,
            karras_steps=karras_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            s_churn=s_churn,
        )
