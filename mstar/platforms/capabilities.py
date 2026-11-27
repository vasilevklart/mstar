"""What a device can do, asked before a resource is built.

Code that picks an implementation asks ``capabilities(device)`` a question (does
PDL exist here, which FP8 format does the GPU multiply) rather than which vendor
this is. Every field's default is the conservative answer, so a platform M*
doesn't know gets the safe path.

Answering for a GPU reads its compute capability or GFX architecture, which
initializes the device runtime, so this is for worker-side code once the device
is chosen; code that runs before the first device call asks the platform
registry's ``built_for`` instead. The answer is cached per platform and device,
but ask it where a resource or module is built rather than inside a forward
that ``torch.compile`` traces or a graph captures.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import torch

from mstar import platforms
from mstar.platforms.interface import Platform


class GraphMode(Enum):
    """How a device makes a step replayable."""

    NONE = "none"  # every step runs eagerly
    CAPTURE = "capture"  # stream capture and replay: CUDA graphs, hipGraphs, XPU graphs


@dataclass(frozen=True)
class Capabilities:
    """What one device can do. Every default is the conservative answer."""

    streams: bool = False  # torch.Stream / torch.Event order and overlap work
    graphs: GraphMode = GraphMode.NONE
    triton: bool = False  # Triton kernels launch here
    nvidia_triton_extras: bool = False  # libdevice, inline PTX, triton.language.extra.cuda
    pdl: bool = False  # programmatic dependent launch (NVIDIA sm_90 and newer)
    pinned_host_memory: bool = False
    host_register: bool = False  # page-lock an existing host mapping for the device
    device_ipc: bool = False  # device memory can be shared across processes
    fp8_dtype: torch.dtype | None = None  # the e4m3 format the GPU multiplies natively
    block_scaled: frozenset[str] = frozenset()  # block-scaled GEMM formats, of "mxfp8" and "nvfp4"


DEFAULTS = Capabilities()

Builder = Callable[[Platform, "torch.device | None"], Capabilities]

# AMD's FP8 differs by architecture, and hipBLASLt refuses the other one
_ROCM_FP8 = {"gfx942": torch.float8_e4m3fnuz, "gfx950": torch.float8_e4m3fn}


def _platform_fp8(platform: Platform, device) -> tuple[bool, torch.dtype | None]:
    # a platform that names its FP8 format answers for itself, so the two can't disagree
    own = getattr(platform, "fp8_dtype", None)
    return (True, own(device)) if callable(own) else (False, None)


def _cuda(platform: Platform, device) -> Capabilities:
    has_own, fp8 = _platform_fp8(platform, device)
    if not has_own:
        fp8 = torch.float8_e4m3fn if platform.has_cuda_capability(8, 9, device) else None
    return Capabilities(
        streams=True, graphs=GraphMode.CAPTURE, triton=True, nvidia_triton_extras=True,
        pdl=platform.has_cuda_capability(9, 0, device),
        pinned_host_memory=True, host_register=True, device_ipc=True, fp8_dtype=fp8,
        block_scaled=frozenset({"mxfp8", "nvfp4"}) if platform.has_block_scaled_gemm(device) else frozenset(),
    )


def _rocm(platform: Platform, device) -> Capabilities:
    has_own, fp8 = _platform_fp8(platform, device)
    if not has_own:
        fp8 = _ROCM_FP8.get(platform.arch(device))
    # host_register stays off until hipHostRegister has a caller; the KV handoff
    # already takes the IPC path on ROCm
    return Capabilities(
        streams=True, graphs=GraphMode.CAPTURE, triton=True,
        pinned_host_memory=True, device_ipc=True, fp8_dtype=fp8,
    )


def _xpu(platform: Platform, device) -> Capabilities:
    # triton stays off until Intel's backend is measured on M*'s kernels, so XPU keeps its eager paths
    return Capabilities(streams=True, graphs=GraphMode.CAPTURE, pinned_host_memory=True)


_BUILDERS: dict[str, Builder] = {
    "cuda": _cuda,
    "rocm": _rocm,
    "xpu": _xpu,
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
