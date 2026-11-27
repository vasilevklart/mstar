"""A platform nobody ships, for tests of what an unknown platform reaches."""

from __future__ import annotations

import functools
import os
from collections.abc import Iterator
from contextlib import contextmanager
from unittest import mock

import torch

from mstar import platforms
from mstar.platforms.capabilities import Capabilities, clear_capabilities_cache
from mstar.platforms.interface import Platform
from mstar.platforms.registry import (
    ENV_VAR,
    PlatformEntry,
    register_platform,
    registered_platforms,
    unregister_platform,
)


class FakePlatform(Platform):
    """A platform M* has no code for. Every capability is at its default unless overridden."""

    def __init__(self, name: str = "fake", device_type: str = "cpu", **capability_overrides):
        self.name = name
        self.device_type = device_type
        self._capabilities = Capabilities(**capability_overrides)

    def capabilities(self, device: torch.device | None = None) -> Capabilities:
        return self._capabilities

    def __repr__(self) -> str:
        return f"FakePlatform({self.name!r})"


@contextmanager
def use_platform(platform: Platform) -> Iterator[Platform]:
    """Make ``platform`` the current one: registered under its name, named by
    ``MSTAR_PLATFORM``, and the resolve and capability caches cleared. All of it is undone on
    exit, and an entry it displaced comes back.

    ``mstar.platforms.resolve_platform`` is patched to return it, so code that reads it
    through the module (``current_platform`` among them) sees it; a module that imported the
    function by name keeps the original.
    """
    displaced = next((e for e in registered_platforms() if e.name == platform.name), None)
    register_platform(PlatformEntry(
        platform.name, lambda: platform, getattr(platform, "device_type", "cpu"),
        built_for=lambda: True, available=lambda: False,
    ), replace=True)
    env = os.environ.get(ENV_VAR)
    os.environ[ENV_VAR] = platform.name
    clear_capabilities_cache()
    try:
        with mock.patch.object(platforms, "resolve_platform", functools.cache(lambda: platform)):
            yield platform
    finally:
        if env is None:
            os.environ.pop(ENV_VAR, None)
        else:
            os.environ[ENV_VAR] = env
        if displaced is None:
            unregister_platform(platform.name)
        else:
            # in place, so detection order is unchanged
            register_platform(displaced, replace=True)
        clear_capabilities_cache()
