"""What a platform M* has no code for reaches: a reference path or a refusal that names it.

The start of a porting checklist. Each strict expected failure below is a path where an
unknown platform still gets NVIDIA's implementation; it passes once that path asks the
platform, and the marker must then go.
"""

from types import SimpleNamespace

import pytest
import torch

from mstar.engine.accelerator_graph_backend import (
    _GRAPH_BACKENDS,
    AcceleratorGraphBackend,
    get_accelerator_graph_backend,
    register_graph_backend,
)
from mstar.engine.resources.attn.base import resolve_attn_backend
from mstar.engine.resources.attn.config import AttnBackend
from mstar.engine.resources.kv import page_copier
from mstar.model.glm5_next import fused_decode
from mstar.platforms import CudaPlatform, RocmPlatform, current_platform
from mstar.platforms.capabilities import DEFAULTS, capabilities
from mstar.platforms.dispatch import PlatformDispatch
from mstar.platforms.testing import FakePlatform, use_platform
from mstar.utils.device import DeviceCapability


@pytest.fixture
def fake():
    with use_platform(FakePlatform()) as platform:
        yield platform


@pytest.fixture
def graph_backends():
    saved = dict(_GRAPH_BACKENDS)
    yield
    _GRAPH_BACKENDS.clear()
    _GRAPH_BACKENDS.update(saved)


def _graph_backend_class():
    """The smallest backend a plugin could register: every abstract method, doing nothing."""
    methods = {name: lambda self, *args, **kwargs: None for name in AcceleratorGraphBackend.__abstractmethods__}
    return type("PluginGraphs", (AcceleratorGraphBackend,), methods)


# --- what the fake platform reaches ---------------------------------------------------------------


def test_the_fake_platform_is_current(fake):
    assert current_platform.name == "fake"


def test_every_capability_is_at_its_default(fake):
    assert capabilities() == DEFAULTS
    assert capabilities(torch.device("privateuseone", 0)) == DEFAULTS


def test_kv_page_copies_take_the_reference(fake):
    resolved = page_copier.PAGE_COPIER.resolve()
    assert (resolved.name, resolved.skipped) == ("torch", ())
    assert isinstance(page_copier.make_page_copier.__wrapped__(), page_copier.TorchCopier)


def test_kv_page_copies_on_cuda_log_each_binding_that_fails_then_fall_back(monkeypatch, caplog):
    """The same three messages as before the slot, without a GPU: neither binding loads."""

    def unavailable(error):
        def build():
            raise error

        return build

    monkeypatch.setattr(page_copier, "CudartCopier", unavailable(OSError("expected one libcudart mapped")))
    monkeypatch.setattr(page_copier, "CudaBindingsCopier", unavailable(ImportError("no cuda.bindings")))
    with use_platform(CudaPlatform()), caplog.at_level("INFO", logger=page_copier.__name__):
        assert isinstance(page_copier.make_page_copier.__wrapped__(), page_copier.TorchCopier)
    assert [(r.levelname, r.getMessage()) for r in caplog.records] == [
        ("INFO", "KV page copies: libcudart unavailable (expected one libcudart mapped)"),
        ("INFO", "KV page copies: cuda.bindings unavailable (no cuda.bindings)"),
        ("WARNING", "KV page copies: no 2D async memcpy binding works here, so offload and reload fall back "
                    "to copy_, which blocks the host and runs several times slower"),
    ]


def test_a_cpu_device_has_no_graph_backend(fake):
    assert DeviceCapability.of("cpu").graph_backend is None


def test_a_graph_backend_registered_for_a_new_device_type_is_found(graph_backends):
    backend_cls = _graph_backend_class()
    register_graph_backend("privateuseone", backend_cls)
    device = torch.device("privateuseone", 0)
    assert isinstance(get_accelerator_graph_backend(device), backend_cls)
    assert isinstance(DeviceCapability(device).graph_backend, backend_cls)
    with pytest.raises(ValueError, match="already has graph backend PluginGraphs"):
        register_graph_backend("privateuseone", backend_cls)
    register_graph_backend("privateuseone", _graph_backend_class(), replace=True)


def test_a_platform_dispatch_with_no_factory_for_it_refuses_naming_it(fake):
    dispatch = PlatformDispatch("mstar::example", cuda=lambda: (lambda: "nvidia"))
    with pytest.raises(NotImplementedError, match="no implementation for platform fake"):
        dispatch()


@pytest.mark.parametrize(
    ("platform", "fused"),
    [
        (CudaPlatform(), True),
        (RocmPlatform(), False),
        (FakePlatform("third-vendor"), False),
        (FakePlatform("third-vendor", nvidia_triton_extras=True), True),
    ],
    ids=repr,
)
def test_glm5_next_fuses_only_where_nvidias_triton_extras_exist(monkeypatch, platform, fused):
    """A third vendor whose tensors are ``cuda`` too used to land on the NVIDIA kernels,
    since the gate was ``not is_rocm()``."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device=None: SimpleNamespace(gcnArchName="gfx942"))
    monkeypatch.setattr(fused_decode, "_HAS_TRITON", True)
    with use_platform(platform):
        assert fused_decode.fused_available(SimpleNamespace(is_cuda=True)) is fused
        assert not fused_decode.fused_available(SimpleNamespace(is_cuda=False))


# --- known gaps: paths an unknown platform still takes on NVIDIA's implementation ----------------


@pytest.mark.xfail(
    strict=True,
    reason="AUTO attention is FlashInfer on every platform (mstar/engine/resources/attn/base.py, "
    "resolve_attn_backend); a per-platform AUTO is the content of #1088",
)
def test_auto_attention_does_not_pick_flashinfer_for_an_unknown_platform(fake):
    assert resolve_attn_backend(AttnBackend.AUTO, torch.device("cpu")) is not AttnBackend.FLASHINFER
