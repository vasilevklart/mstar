"""``mstar.platforms.capabilities``: what a device can do, asked instead of which vendor it is.
GPU queries are stubbed, so no accelerator is needed."""

import functools
import sys
from types import SimpleNamespace

import pytest
import torch

from mstar import platforms
from mstar.platforms import CpuPlatform, CudaPlatform, Platform, RocmPlatform, XpuPlatform
from mstar.platforms import capabilities as caps_module
from mstar.platforms.capabilities import (
    DEFAULTS,
    Capabilities,
    GraphMode,
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
def gpu(monkeypatch):
    """Stub the compute capability (CUDA's, or the GFX version ROCm reports) and the GFX arch."""

    def set_gpu(capability=(9, 0), arch="gfx942"):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
        monkeypatch.setattr(
            torch.cuda, "get_device_properties",
            lambda device=None: SimpleNamespace(gcnArchName=f"{arch}:sramecc+:xnack-"),
        )

    return set_gpu


@pytest.mark.parametrize(
    ("capability", "pdl", "fp8", "block_scaled"),
    [
        ((8, 0), False, None, frozenset()),
        ((8, 9), False, torch.float8_e4m3fn, frozenset()),
        ((9, 0), True, torch.float8_e4m3fn, frozenset()),
        ((10, 0), True, torch.float8_e4m3fn, frozenset({"mxfp8", "nvfp4"})),
    ],
)
def test_cuda(gpu, capability, pdl, fp8, block_scaled):
    gpu(capability)
    assert capabilities(GPU, CudaPlatform()) == Capabilities(
        streams=True, graphs=GraphMode.CAPTURE, triton=True, nvidia_triton_extras=True, pdl=pdl,
        pinned_host_memory=True, host_register=True, device_ipc=True, fp8_dtype=fp8,
        block_scaled=block_scaled,
    )


@pytest.mark.parametrize(
    ("arch", "fp8"),
    [("gfx942", torch.float8_e4m3fnuz), ("gfx950", torch.float8_e4m3fn), ("gfx90a", None)],
)
def test_rocm(gpu, arch, fp8):
    gpu((9, 4), arch)
    assert capabilities(GPU, RocmPlatform()) == Capabilities(
        streams=True, graphs=GraphMode.CAPTURE, triton=True, pinned_host_memory=True,
        device_ipc=True, fp8_dtype=fp8,
    )


def test_an_mi300x_gets_no_pdl_though_it_reports_capability_9_4(gpu):
    """The fused MoE decode asked ``get_device_capability()[0] >= 9``, which an MI300X passes."""
    gpu((9, 4), "gfx942")
    assert torch.cuda.get_device_capability(GPU)[0] >= 9
    assert not capabilities(GPU, RocmPlatform()).pdl
    assert not capabilities(GPU, RocmPlatform()).nvidia_triton_extras


def test_xpu_keeps_its_eager_paths():
    assert capabilities(torch.device("xpu", 0), XpuPlatform()) == Capabilities(
        streams=True, graphs=GraphMode.CAPTURE, pinned_host_memory=True,
    )


def test_cpu_platform_gets_the_defaults():
    assert capabilities(None, CpuPlatform()) == DEFAULTS == Capabilities()


@pytest.mark.parametrize(
    "platform",
    [CudaPlatform(), RocmPlatform(), XpuPlatform(), CpuPlatform(), FakePlatform(pdl=True, streams=True)],
    ids=repr,
)
@pytest.mark.parametrize("device", [CPU, "cpu"])
def test_a_cpu_device_gets_the_defaults_on_every_platform(monkeypatch, platform, device):
    """A resource built with CPU stand-ins on a GPU host asks no GPU."""
    def fail(*args, **kwargs):
        pytest.fail("a CPU device queried the GPU")

    monkeypatch.setattr(torch.cuda, "get_device_capability", fail)
    monkeypatch.setattr(torch.cuda, "get_device_properties", fail)
    assert capabilities(device, platform) is DEFAULTS


def test_a_platforms_own_method_wins_over_the_builtin_for_its_name(gpu):
    gpu((9, 0))

    class NoPdlCuda(CudaPlatform):
        def capabilities(self, device=None):
            return Capabilities(streams=True)

    assert capabilities(GPU, NoPdlCuda()) == Capabilities(streams=True)


def test_an_unknown_platform_gets_the_defaults():
    class Mystery(Platform):
        name = "mystery"

    assert capabilities(GPU, Mystery()) is DEFAULTS


def test_a_builder_registered_for_a_name_answers_for_it():
    class Plugin(Platform):
        name = "plugin-for-test"

    register_capabilities("plugin-for-test", lambda platform, device: Capabilities(streams=True))
    assert capabilities(None, Plugin()).streams


def test_a_platform_that_names_its_fp8_format_answers_for_itself(gpu):
    """The ROCm stack adds ``Platform.fp8_dtype``; the two must not disagree."""
    gpu((9, 4), "gfx942")

    class OwnFp8(RocmPlatform):
        def fp8_dtype(self, device=None):
            return torch.float8_e5m2

    assert capabilities(GPU, OwnFp8()).fp8_dtype is torch.float8_e5m2


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
    with use_platform(FakePlatform(streams=True, pdl=True)):
        assert capabilities().pdl
        assert capabilities(platform=platforms.current_platform).pdl
    assert platforms.current_platform.name != "fake"


@needs_compile
def test_a_compiled_forward_asks_without_a_graph_break(dynamo_counts):
    """Traced, the lookup would rebuild the platform as an object Dynamo can't store, and
    the whole frame would run eagerly; it folds to a constant instead."""

    def forward(x):
        return x * 2 if capabilities().streams else x - 1

    assert dynamo_counts(forward, torch.ones(2))[:2] == (1, 1)


@needs_compile
def test_a_compiled_forward_asks_a_gpu_platform_without_a_graph_break(monkeypatch, gpu, dynamo_counts):
    gpu((9, 4), "gfx942")
    platform = RocmPlatform()
    monkeypatch.setattr(platforms, "resolve_platform", functools.cache(lambda: platform))

    def forward(x):
        c = capabilities(GPU)
        return x * 2 if c.fp8_dtype is torch.float8_e4m3fnuz and not c.pdl else x - 1

    assert dynamo_counts(forward, torch.ones(2))[:2] == (1, 1)
