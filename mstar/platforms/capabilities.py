"""What a device can do, asked instead of which vendor it is.

Code that picks an implementation asks ``capabilities(device)`` a question (does
programmatic dependent launch exist here, can host memory be pinned) rather than
which vendor this is. Every field's default is the conservative answer, so a
platform M* doesn't know gets the safe path. A field is added with the first code
that reads it.

Answering for an NVIDIA GPU reads its compute capability, which initializes the
device runtime, so this is for worker-side code once the device is chosen. The
answer is cached per platform and device. A forward that ``torch.compile`` traces
may ask too: Dynamo calls this at trace time and keeps the answer.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import dataclass

import torch

from mstar import platforms
from mstar.platforms.interface import Platform


@dataclass(frozen=True)
class Capabilities:
    """What one device can do. Every default is the conservative answer."""

    nvidia_triton_extras: bool = False  # libdevice, inline PTX, triton.language.extra.cuda
    pdl: bool = False  # programmatic dependent launch (NVIDIA sm_90 and newer)
    pinned_host_memory: bool = False  # host tensors can be pinned for async copies to the device


DEFAULTS = Capabilities()

Builder = Callable[[Platform, "torch.device | None"], Capabilities]


def _cuda(platform: Platform, device) -> Capabilities:
    return Capabilities(
        nvidia_triton_extras=True, pdl=platform.has_cuda_capability(9, 0, device), pinned_host_memory=True,
    )


def _pins(platform: Platform, device) -> Capabilities:
    return Capabilities(pinned_host_memory=True)


_BUILDERS: dict[str, Builder] = {
    "cuda": _cuda,
    "rocm": _pins,
    "xpu": _pins,
    "cpu": lambda platform, device: DEFAULTS,
}
_CACHE: weakref.WeakKeyDictionary[Platform, dict[torch.device | None, Capabilities]] = (
    weakref.WeakKeyDictionary()
)


def register_capabilities(platform_name: str, builder: Builder) -> None:
    """Answer for platforms named ``platform_name`` that have no ``capabilities``
    method of their own. ``builder(platform, device)`` gets ``device=None`` for the
    platform's current device."""
    _BUILDERS[platform_name] = builder
    clear_capabilities_cache()


def clear_capabilities_cache() -> None:
    _CACHE.clear()


# Dynamo calls it at trace time and folds the answer into the graph. Traced, it would
# rebuild the platform from resolve_platform's body rather than its cache, and could
# not store that new object in the cache, so the whole frame would run eagerly.
@torch.compiler.assume_constant_result
def capabilities(device: torch.device | str | None = None, platform: Platform | None = None) -> Capabilities:
    """What ``device`` (by default the platform's current one) can do on
    ``platform`` (by default ``current_platform``).

    A CPU device gets the defaults on every platform. Otherwise a platform's own
    ``capabilities(device)`` method answers, then the builder registered for its
    name; a platform with neither gets the defaults.
    """
    if device is not None:
        if not isinstance(device, torch.device):
            device = torch.device(device)
        if device.type == "cpu":
            return DEFAULTS
    if platform is None or platform is platforms.current_platform:
        # the instance itself: the conftest fixture swaps it between tests
        platform = platforms.resolve_platform()
    per_device = _CACHE.get(platform)
    if per_device is None:
        per_device = _CACHE.setdefault(platform, {})
    caps = per_device.get(device)
    if caps is None:
        caps = per_device[device] = _build(platform, device)
    return caps


def _build(platform: Platform, device) -> Capabilities:
    own = getattr(platform, "capabilities", None)
    if callable(own):
        return own(device)
    builder = _BUILDERS.get(platform.name)
    return DEFAULTS if builder is None else builder(platform, device)
