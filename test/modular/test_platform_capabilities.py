"""``mstar.platforms.capabilities``: what a device can do, asked instead of which vendor it is.
GPU queries are stubbed, so no accelerator is needed."""

import functools
import sys

import pytest
import torch

from mstar import platforms
from mstar.platforms import CpuPlatform, CudaPlatform, Platform, RocmPlatform, XpuPlatform
from mstar.platforms import capabilities as caps_module
from mstar.platforms.capabilities import (
    DEFAULTS,
    Capabilities,
    capabilities,
    clear_capabilities_cache,
    register_capabilities,
)
from mstar.platforms.testing import FakePlatform, use_platform

GPU = torch.device("cuda", 0)
CPU = torch.device("cpu")

# torch.compile imports Triton's modules wherever Triton is installed; a host without it
# has conftest's stub, which has no such modules
needs_compile = pytest.mark.skipif(
    "triton" in sys.modules and getattr(sys.modules["triton"], "__file__", None) is None,
    reason="torch.compile can't import Triton's modules past conftest's stub on this host",
)


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    """A fresh answer cache, and builders registered by a test go when it ends."""
    monkeypatch.setattr(caps_module, "_BUILDERS", dict(caps_module._BUILDERS))
    clear_capabilities_cache()
    yield
    clear_capabilities_cache()


@pytest.fixture
def compute_capability(monkeypatch):
    """Stub what a ``cuda`` device reports: NVIDIA's compute capability, or the GFX version on ROCm."""

    def set_capability(capability):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)

    return set_capability


@pytest.mark.parametrize(
    ("capability", "pdl"), [((8, 0), False), ((8, 9), False), ((9, 0), True), ((10, 0), True)],
)
def test_cuda(compute_capability, capability, pdl):
    compute_capability(capability)
    assert capabilities(GPU, CudaPlatform()) == Capabilities(
        nvidia_triton_extras=True, pdl=pdl, pinned_host_memory=True,
    )


def test_rocm_has_neither_nvidia_capability_though_an_mi300x_reports_9_4(compute_capability):
    """The fused MoE decode asked ``get_device_capability()[0] >= 9``, which an MI300X passes."""
    compute_capability((9, 4))
    assert capabilities(GPU, RocmPlatform()) == Capabilities(pinned_host_memory=True)


def test_xpu_pins_and_has_neither_nvidia_capability():
    assert capabilities(torch.device("xpu", 0), XpuPlatform()) == Capabilities(pinned_host_memory=True)


def test_cpu_platform_gets_the_defaults():
    assert capabilities(None, CpuPlatform()) == DEFAULTS == Capabilities()
    assert DEFAULTS == Capabilities(nvidia_triton_extras=False, pdl=False, pinned_host_memory=False)


@pytest.mark.parametrize(
    "platform",
    [CudaPlatform(), RocmPlatform(), XpuPlatform(), FakePlatform(pdl=True, pinned_host_memory=True)],
    ids=repr,
)
@pytest.mark.parametrize("device", [CPU, "cpu"])
def test_a_cpu_device_gets_the_defaults_on_every_platform(monkeypatch, platform, device):
    """A resource built with CPU stand-ins on a GPU host asks no GPU."""
    def fail(*args, **kwargs):
        pytest.fail("a CPU device queried the GPU")

    monkeypatch.setattr(torch.cuda, "get_device_capability", fail)
    assert capabilities(device, platform) is DEFAULTS


def test_a_platforms_own_method_wins_over_the_builtin_for_its_name(compute_capability):
    compute_capability((9, 0))

    class NoPdlCuda(CudaPlatform):
        def capabilities(self, device=None):
            return Capabilities(pinned_host_memory=True)

    assert capabilities(GPU, NoPdlCuda()) == Capabilities(pinned_host_memory=True)


def test_an_unknown_platform_gets_the_defaults():
    class Mystery(Platform):
        name = "mystery"

    assert capabilities(GPU, Mystery()) is DEFAULTS


def test_a_builder_registered_for_a_name_answers_for_it():
    class Plugin(Platform):
        name = "plugin-for-test"

    register_capabilities("plugin-for-test", lambda platform, device: Capabilities(pdl=True))
    assert capabilities(None, Plugin()) == Capabilities(pdl=True)


def test_answers_are_cached_per_platform_instance_and_device():
    calls = []

    class Counting(Platform):
        name = "counting"

    register_capabilities("counting", lambda platform, device: calls.append(device) or DEFAULTS)
    first = Counting()
    for _ in range(3):
        capabilities(GPU, first)
        capabilities(torch.device("cuda", 1), first)
    assert calls == [GPU, torch.device("cuda", 1)]
    # the conftest fixture builds a new platform per test: a new instance asks again
    capabilities(GPU, Counting())
    assert len(calls) == 3


def test_the_default_platform_is_the_current_one():
    with use_platform(FakePlatform(pdl=True)):
        assert capabilities().pdl
        assert capabilities(platform=platforms.current_platform).pdl
    assert platforms.current_platform.name != "fake"


@needs_compile
def test_a_compiled_forward_asks_without_a_graph_break(dynamo_counts):
    """Traced, the lookup would rebuild the platform as an object Dynamo can't store, and
    the whole frame would run eagerly; it folds to a constant instead."""

    def forward(x):
        return x * 2 if capabilities().pdl else x - 1

    with use_platform(FakePlatform(pdl=True)):
        assert dynamo_counts(forward, torch.ones(2))[:2] == (1, 1)


@needs_compile
def test_a_compiled_forward_asks_a_gpu_platform_without_a_graph_break(monkeypatch, compute_capability, dynamo_counts):
    compute_capability((9, 0))
    platform = CudaPlatform()
    monkeypatch.setattr(platforms, "resolve_platform", functools.cache(lambda: platform))

    def forward(x):
        c = capabilities(GPU)
        return x * 2 if c.pdl and c.nvidia_triton_extras else x - 1

    assert dynamo_counts(forward, torch.ones(2))[:2] == (1, 1)
