"""What a platform M* has no code for reaches: a reference path or a refusal that names it.

The start of a porting checklist. Each strict expected failure below is a path where an
unknown platform still gets an in-tree platform's treatment; it passes once that path asks
the platform, and the marker must then go.
"""

import os
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
from mstar.engine.resources.attn.ragged.block_causal import RaggedBlockCausalWrapper
from mstar.engine.resources.attn.workspace import Workspace
from mstar.engine.resources.kv import page_copier
from mstar.model.glm5_next import fused_decode
from mstar.platforms import CudaPlatform, Platform, RocmPlatform, apply_process_env_defaults, current_platform
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


def _binding(name: str) -> SimpleNamespace:
    """A stand-in for a 2D memcpy binding that loaded."""
    return SimpleNamespace(name=name)


def _unavailable(error: Exception):
    def build():
        raise error

    return build


def _copier_log(caplog) -> list[tuple[str, str]]:
    return [(r.levelname, r.getMessage()) for r in caplog.records]


# --- what the fake platform reaches ---------------------------------------------------------------


def test_the_fake_platform_is_current(fake):
    assert current_platform.name == "fake"


def test_a_platform_that_says_nothing_gets_every_capability_at_its_default():
    """No ``capabilities`` of its own and a name M* has no answers for: the safe path, for
    a ``cuda`` device too."""

    class Silent(Platform):
        name = "silent"

    with use_platform(Silent()):
        assert capabilities() is DEFAULTS
        assert capabilities(torch.device("cuda", 0)) is DEFAULTS


def test_kv_page_copies_take_the_reference(fake):
    resolved = page_copier.PAGE_COPIER.resolve()
    assert (resolved.name, resolved.skipped) == ("torch", ())
    assert isinstance(page_copier.make_page_copier.__wrapped__(), page_copier.TorchCopier)


def test_a_device_type_with_no_graph_backend_runs_eagerly():
    capability = DeviceCapability(torch.device("privateuseone", 0))
    assert capability.graph_backend is None
    assert not capability.is_capturing()


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


# --- the in-tree platforms get what they got before -------------------------------------------------
# No GPU here, so the bindings and the device queries are stand-ins; the slot, the gates and
# the log lines are the real ones.


@pytest.mark.parametrize(
    ("platform", "copier", "name"),
    [(CudaPlatform(), "CudartCopier", "libcudart"), (RocmPlatform(), "HipCopier", "libamdhip64")],
    ids=repr,
)
def test_kv_page_copies_take_the_platforms_first_binding_that_round_trips(monkeypatch, caplog, platform, copier, name):
    def other_vendor():
        pytest.fail("built another platform's binding")

    binding = _binding(name)
    for cls in ("CudartCopier", "CudaBindingsCopier", "HipCopier"):
        monkeypatch.setattr(page_copier, cls, (lambda: binding) if cls == copier else other_vendor)
    monkeypatch.setattr(page_copier, "_round_trips", lambda copier: True)
    with use_platform(platform), caplog.at_level("INFO", logger=page_copier.__name__):
        assert page_copier.make_page_copier.__wrapped__() is binding
    assert _copier_log(caplog) == [("INFO", f"KV page copies: one 2D async memcpy a run of pages, via {name}")]


def test_kv_page_copies_pass_over_a_binding_that_copies_wrong(monkeypatch, caplog):
    wrong, right = _binding("libcudart"), _binding("cuda.bindings")
    monkeypatch.setattr(page_copier, "CudartCopier", lambda: wrong)
    monkeypatch.setattr(page_copier, "CudaBindingsCopier", lambda: right)
    monkeypatch.setattr(page_copier, "_round_trips", lambda copier: copier is right)
    with use_platform(CudaPlatform()), caplog.at_level("INFO", logger=page_copier.__name__):
        assert page_copier.make_page_copier.__wrapped__() is right
    assert _copier_log(caplog) == [
        ("WARNING", "KV page copies: libcudart returned wrong data; not using it"),
        ("INFO", "KV page copies: one 2D async memcpy a run of pages, via cuda.bindings"),
    ]


def test_kv_page_copies_on_cuda_log_each_binding_that_fails_then_fall_back(monkeypatch, caplog):
    monkeypatch.setattr(page_copier, "CudartCopier", _unavailable(OSError("expected one libcudart mapped")))
    monkeypatch.setattr(page_copier, "CudaBindingsCopier", _unavailable(ImportError("no cuda.bindings")))
    with use_platform(CudaPlatform()), caplog.at_level("INFO", logger=page_copier.__name__):
        assert isinstance(page_copier.make_page_copier.__wrapped__(), page_copier.TorchCopier)
    assert _copier_log(caplog) == [
        ("INFO", "KV page copies: libcudart unavailable (expected one libcudart mapped)"),
        ("INFO", "KV page copies: cuda.bindings unavailable (no cuda.bindings)"),
        ("WARNING", "KV page copies: no 2D async memcpy binding works here, so offload and reload fall back "
                    "to copy_, which blocks the host and runs several times slower"),
    ]


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
    monkeypatch.setattr(fused_decode, "_HAS_TRITON", True)
    with use_platform(platform):
        assert fused_decode.fused_available(SimpleNamespace(is_cuda=True)) is fused
        assert not fused_decode.fused_available(SimpleNamespace(is_cuda=False))


@pytest.mark.parametrize(
    ("platform", "pins"),
    [
        (CudaPlatform(), True),
        (RocmPlatform(), True),
        (FakePlatform(), False),
        (FakePlatform(pinned_host_memory=True), True),
    ],
    ids=repr,
)
def test_a_block_causal_plan_pins_its_host_tensors_only_where_the_platform_can(monkeypatch, platform, pins):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    pinned = []
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: pinned.append(self) or self)
    with use_platform(platform):
        staged = RaggedBlockCausalWrapper._pinned([0, 3, 5])
    assert staged.tolist() == [0, 3, 5]
    assert len(pinned) == pins


def test_a_workspace_on_an_accelerator_waits_for_it_before_it_grows(monkeypatch):
    """Through ``torch.accelerator``, so for a device of any type: a kernel may still be
    reading the buffer the workspace is about to free."""
    workspace = Workspace(torch.empty(1024, dtype=torch.uint8), "test")
    waited = []
    monkeypatch.setattr(DeviceCapability, "of", staticmethod(lambda device: SimpleNamespace(is_accelerator=True)))
    monkeypatch.setattr(
        torch.accelerator, "synchronize", lambda device=None: waited.append((device, workspace.tensor.numel())),
    )

    def plan():
        if workspace.tensor.numel() == 1024:
            raise RuntimeError(
                "Buffer overflow when allocating memory for batch_prefill_tmp_v with size 4096 "
                "and alignment 16, but only 1024 bytes available"
            )

    workspace.plan(plan)
    assert waited == [(torch.device("cpu"), 1024)]  # once, while it still held the old buffer
    assert workspace.tensor.numel() == 1024 * 1024


# --- known gaps: paths where an unknown platform still gets an in-tree one's treatment -------------


# Only the named assertion is expected to fail: any other error in one of these is a failure.


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="AUTO attention is FlashInfer on every platform (mstar/engine/resources/attn/base.py, "
    "resolve_attn_backend); a per-platform AUTO is the content of #1088",
)
def test_auto_attention_does_not_pick_flashinfer_for_an_unknown_platform(fake):
    assert resolve_attn_backend(AttnBackend.AUTO, torch.device("cpu")) is not AttnBackend.FLASHINFER


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="apply_process_env_defaults asks the platform torch was built for (mstar/platforms/__init__.py, "
    "_built_for), which is always an in-tree one",
)
def test_the_selected_platforms_process_env_defaults_are_applied(monkeypatch):
    class WithEnv(FakePlatform):
        def process_env_defaults(self):
            return {"FAKE_PLATFORM_RUNTIME_FLAG": "1"}

    # a copy, so the host platform's own defaults don't outlive the test
    monkeypatch.setattr(os, "environ", dict(os.environ))
    with use_platform(WithEnv()):
        if current_platform.process_env_defaults() != {"FAKE_PLATFORM_RUNTIME_FLAG": "1"}:
            raise RuntimeError("the fake platform is not the current one")
        apply_process_env_defaults()
        assert os.environ.get("FAKE_PLATFORM_RUNTIME_FLAG") == "1"
