"""Opt-in, lazy model lifetimes shared by auxiliary pipeline blocks."""
from collections.abc import Callable, Hashable
from contextlib import AbstractContextManager, ExitStack, nullcontext
from typing import Any

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.loader.registry import ModelRegistry
from ltx_pipelines.utils.gpu_model import gpu_model


class ModelResidency:
    """Own model builds until scope exit, including partial-build/error cleanup."""

    def __init__(self) -> None:
        self.models: dict[Hashable, Any] = {}
        self.lifetime = ExitStack()
        self.signatures: dict[Hashable, Hashable] = {}
        self.active = True

    def model_context(
        self, key: tuple[Hashable, ...], build: Callable[[], Any], trim: AllocatorTrimStrategy
    ) -> AbstractContextManager:
        if not self.active:
            raise RuntimeError("auxiliary model residency scope has ended")
        role = key[0]
        if role in self.signatures and self.signatures[role] != key:
            raise ValueError("cannot change auxiliary model dtype, builder, device or policy inside a resident scope")
        if key not in self.models:
            self.signatures[role] = key
            self.models[key] = self.lifetime.enter_context(gpu_model(build(), alloc_trim_strategy=trim))
        return nullcontext(self.models[key])

    def close(self) -> None:
        self.active = False
        self.models.clear()
        self.lifetime.close()


def model_context(
    residency: ModelResidency | None,
    key: tuple[Hashable, ...],
    build: Callable[[], Any],
    trim: AllocatorTrimStrategy,
) -> AbstractContextManager:
    if residency is not None:
        return residency.model_context(key, build, trim)
    return gpu_model(build(), alloc_trim_strategy=trim)


def build_model(builder, residency: ModelResidency | None, *, device, dtype):
    """A resident module must not alias another owner's structurally cached shell."""
    if residency is not None:
        builder = builder.with_registry(ModelRegistry(cache_models=False, cache_weights=False))
    return builder.build(device=device, dtype=dtype).eval()
