"""The platforms M* can run on, in the tree or from installed plugins, and which one a process uses.

The four in-tree platforms are registered here. A package outside the tree adds one through the
``mstar.platforms`` entry-point group in its own ``pyproject.toml``::

    [project.entry-points."mstar.platforms"]
    tpu = "mstar_tpu:platform_entry"

The entry point's name is the platform's name, and it names a callable that returns a
:class:`PlatformEntry`, or None when the package can't serve this host. Importing this module
loads no plugin.

`select_platform`: ``MSTAR_PLATFORM`` names the platform, and only that plugin is loaded, so no
other vendor's SDK is imported. Unset, the in-tree accelerators come first, in
``resolve_platform``'s order, so an installed plugin can't take over a GPU host; then exactly
one available plugin; then the CPU.
"""

from __future__ import annotations

import functools
import importlib.metadata
import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from mstar.platforms.interface import (
    CpuPlatform,
    CudaPlatform,
    Platform,
    RocmPlatform,
    XpuPlatform,
)

logger = logging.getLogger(__name__)

# a plugin built against another version is refused; the contract may change before 1.0
PLATFORM_API_VERSION = 1
ENTRY_POINT_GROUP = "mstar.platforms"
ENV_VAR = "MSTAR_PLATFORM"
CPU = "cpu"


@dataclass(frozen=True)
class PlatformEntry:
    """How to detect and build one platform."""

    name: str
    factory: Callable[[], Platform]
    device_type: str  # torch's device type for its devices: "cuda" for AMD GPUs too
    built_for: Callable[[], bool]  # torch is built for it; makes no device call
    available: Callable[[], bool]  # this process can see a device; may initialize the runtime
    in_tree: bool = False
    api_version: int = PLATFORM_API_VERSION


_ENTRIES: dict[str, PlatformEntry] = {}
# entry points already loaded, so a scan after a named load doesn't load one twice
_LOADED: set[str] = set()


def register_platform(entry: PlatformEntry, *, replace: bool = False) -> None:
    if entry.api_version != PLATFORM_API_VERSION:
        raise ValueError(
            f"platform {entry.name!r} was written for platform API version {entry.api_version}, "
            f"and this M* has version {PLATFORM_API_VERSION}"
        )
    if entry.name in _ENTRIES and not replace:
        raise ValueError(f"platform {entry.name!r} is already registered")
    _ENTRIES[entry.name] = entry


def unregister_platform(name: str) -> None:
    _ENTRIES.pop(name, None)
    _LOADED.discard(name)


def registered_platforms() -> tuple[PlatformEntry, ...]:
    return tuple(_ENTRIES.values())


@functools.cache
def _entry_points() -> tuple[importlib.metadata.EntryPoint, ...]:
    # read once: listing the installed distributions takes milliseconds
    return tuple(importlib.metadata.entry_points(group=ENTRY_POINT_GROUP))


def load_plugins(only: str | None = None) -> None:
    """Register the platforms of the installed plugins, or of the one named ``only``.

    A named plugin that fails to load raises. In a scan of them all, one that fails is
    logged and skipped, so a broken package elsewhere doesn't stop a host it doesn't serve.
    """
    for ep in _entry_points():
        if (only is not None and ep.name != only) or ep.name in _LOADED:
            continue
        _LOADED.add(ep.name)
        try:
            entry = ep.load()()
            if entry is None:
                continue
            if not isinstance(entry, PlatformEntry) or entry.name != ep.name:
                raise TypeError(f"expected a PlatformEntry named {ep.name!r}, got {entry!r}")
            register_platform(entry)
        except Exception as e:
            if only is not None:
                raise
            logger.warning("platform plugin %s (%s) not loaded: %s", ep.name, ep.value, e)


def select_platform(environ: Mapping[str, str] = os.environ) -> PlatformEntry:
    """The platform this process runs on; detecting one may initialize its runtime."""
    name = environ.get(ENV_VAR, "").strip()
    if name:
        if name not in _ENTRIES:
            load_plugins(only=name)
        if name not in _ENTRIES:
            known = sorted({*_ENTRIES, *(ep.name for ep in _entry_points())})
            raise ValueError(f"{ENV_VAR}={name!r} names no platform M* knows; it knows {', '.join(known)}")
        return _ENTRIES[name]
    for entry in _ENTRIES.values():
        if entry.in_tree and entry.name != CPU and entry.available():
            return entry
    load_plugins()
    available = [entry for entry in _ENTRIES.values() if not entry.in_tree and _plugin_available(entry)]
    if len(available) > 1:
        names = ", ".join(sorted(entry.name for entry in available))
        raise RuntimeError(f"several platforms can run here ({names}); set {ENV_VAR} to one of them")
    return available[0] if available else _ENTRIES[CPU]


def _plugin_available(entry: PlatformEntry) -> bool:
    try:
        return entry.available()
    except Exception as e:
        logger.warning("platform plugin %s: its availability check raised, so it is skipped: %s", entry.name, e)
        return False


def resolve_registered() -> Platform:
    """Select the platform and build it."""
    entry = select_platform()
    if not entry.in_tree or os.environ.get(ENV_VAR):
        logger.info("platform: %s (%s)", entry.name, ENV_VAR if os.environ.get(ENV_VAR) else "plugin")
    return entry.factory()


def _xpu_available() -> bool:
    xpu = getattr(torch, "xpu", None)
    return xpu is not None and xpu.is_available()


# The in-tree accelerators in detection order: CUDA and ROCm share torch.cuda, told apart by
# torch.version.hip; then XPU. The CPU is the fallback, never detected.
for _entry in (
    PlatformEntry(
        "cuda", CudaPlatform, "cuda", in_tree=True,
        built_for=lambda: torch.version.hip is None and torch.version.cuda is not None,
        available=lambda: torch.version.hip is None and torch.cuda.is_available(),
    ),
    PlatformEntry(
        "rocm", RocmPlatform, "cuda", in_tree=True,
        built_for=lambda: torch.version.hip is not None,
        available=lambda: torch.version.hip is not None and torch.cuda.is_available(),
    ),
    PlatformEntry(
        "xpu", XpuPlatform, "xpu", in_tree=True,
        built_for=lambda: getattr(torch.version, "xpu", None) is not None,
        available=_xpu_available,
    ),
    PlatformEntry(CPU, CpuPlatform, "cpu", in_tree=True, built_for=lambda: True, available=lambda: True),
):
    register_platform(_entry)
