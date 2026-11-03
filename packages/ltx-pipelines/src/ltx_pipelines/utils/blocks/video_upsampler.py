from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.loader.registry import ModelRegistry, Registry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.model.upsampler import LatentUpsamplerConfigurator, upsample_video
from ltx_core.model.video_vae import VideoEncoderConfigurator
from ltx_pipelines.utils.blocks.image_conditioner import VAE_ENCODER_SD_OPS
from ltx_pipelines.utils.blocks.residency import ModelResidency, build_model, model_context

logger = logging.getLogger(__name__)

__all__ = ["VideoUpsampler"]


class VideoUpsampler:
    """Owns video encoder + spatial upsampler lifecycle."""

    def __init__(
        self,
        checkpoint_path: str,
        upsampler_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
    ) -> None:
        self._upsampler_path = upsampler_path
        self._dtype = dtype
        self._device = device
        self._encoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=VideoEncoderConfigurator,
            model_sd_ops=VAE_ENCODER_SD_OPS,
            registry=registry or ModelRegistry(cache_models=True, cache_weights=False),
        )
        self._upsampler_builder = Builder(
            model_path=upsampler_path,
            model_class_configurator=LatentUpsamplerConfigurator,
            registry=registry or ModelRegistry(cache_models=True, cache_weights=False),
        )
        self._alloc_trim_strategy = alloc_trim_strategy
        self._resident: ModelResidency | None = None

    @contextmanager
    def resident(self) -> Iterator[None]:
        """Reuse lazy builds until scope exit; nested scopes share the lifetime."""
        if self._resident is not None:
            yield
            return
        holder = ModelResidency()
        self._resident = holder
        try:
            yield
        finally:
            self._resident = None
            holder.close()

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        """Upsample video; free models per call or at explicit resident scope exit."""
        logger.info("Building video encoder + spatial upsampler from %s", self._upsampler_path)
        with (
            model_context(
                self._resident, ("encoder", id(self._encoder_builder), self._device, self._dtype),
                lambda: build_model(self._encoder_builder, self._resident, device=self._device, dtype=self._dtype),
                self._alloc_trim_strategy,
            ) as encoder,
            model_context(
                self._resident, ("upsampler", id(self._upsampler_builder), self._device, self._dtype),
                lambda: build_model(self._upsampler_builder, self._resident, device=self._device, dtype=self._dtype),
                self._alloc_trim_strategy,
            ) as upsampler,
        ):
            return upsample_video(latent=latent, video_encoder=encoder, upsampler=upsampler)
