"""``mstar.platforms.registry``: which platform a process runs on, in the tree or from a plugin.

Plugins here are real installed distributions: a module and a ``.dist-info`` with an
``mstar.platforms`` entry point, written to a directory put on ``sys.path``.
"""

import logging
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch

from mstar.platforms import CpuPlatform, CudaPlatform, Platform, RocmPlatform, XpuPlatform, _built_for, registry
from mstar.platforms.registry import (
    ENTRY_POINT_GROUP,
    PLATFORM_API_VERSION,
    PlatformEntry,
    load_plugins,
    register_platform,
    registered_platforms,
    resolve_registered,
    select_platform,
)

PLUGINS = ("mstar_test_plugin_a", "mstar_test_plugin_b")


@pytest.fixture(autouse=True)
def _own_registry(monkeypatch):
    """Each test registers into a copy, and imports its plugins afresh."""
    monkeypatch.setattr(registry, "_ENTRIES", dict(registry._ENTRIES))
    monkeypatch.setattr(registry, "_LOADED", set())
    for module in PLUGINS:
        monkeypatch.delitem(sys.modules, module, raising=False)
    yield
    for module in PLUGINS:
        sys.modules.pop(module, None)


@pytest.fixture(autouse=True)
def build(monkeypatch):
    """Stub a torch build and the devices it sees, as ``test_platforms.py`` does; by
    default a CUDA build that sees no device, so a GPU host runs these tests alike."""

    def set_build(*, cuda: bool = False, hip: str | None = None, xpu: bool = False, cuda_version="12.8"):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.version, "hip", hip)
        monkeypatch.setattr(torch.version, "cuda", None if hip else cuda_version)
        monkeypatch.setattr(torch, "xpu", SimpleNamespace(is_available=lambda: xpu))

    set_build()
    return set_build


@pytest.fixture
def install(tmp_path, monkeypatch):
    """Install a plugin distribution whose entry point is named ``name``."""

    def add(name: str, module: str = PLUGINS[0], *, available: bool = True, api_version: int = PLATFORM_API_VERSION,
            body: str | None = None):
        body = body or f"""
            return PlatformEntry({name!r}, Plugin, "privateuseone", built_for=lambda: True,
                                 available=lambda: {available}, api_version={api_version})
        """
        (tmp_path / f"{module}.py").write_text(textwrap.dedent(f"""
            from mstar.platforms import Platform
            from mstar.platforms.registry import PlatformEntry

            class Plugin(Platform):
                name = {name!r}

            def entry():
        """) + textwrap.indent(textwrap.dedent(body), "    "))
        dist = tmp_path / f"{module}-0.1.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {module}\nVersion: 0.1\n")
        (dist / "entry_points.txt").write_text(f"[{ENTRY_POINT_GROUP}]\n{name} = {module}:entry\n")
        monkeypatch.syspath_prepend(str(tmp_path))  # also invalidates the import caches

    return add


def _old_chain() -> type[Platform]:
    """``resolve_platform`` before the registry, which the in-tree detection must match."""
    if torch.cuda.is_available():
        return RocmPlatform if torch.version.hip is not None else CudaPlatform
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return XpuPlatform
    return CpuPlatform


# --- in-tree detection -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("devices", "expected"),
    [
        ({"cuda": True}, "cuda"),
        ({"cuda": True, "hip": "7.0"}, "rocm"),
        ({"xpu": True}, "xpu"),
        ({"cuda": True, "xpu": True}, "cuda"),
        ({}, "cpu"),
        ({"hip": "7.0"}, "cpu"),
    ],
)
def test_detection_agrees_with_the_old_if_chain(build, devices, expected):
    build(**devices)
    entry = select_platform({})
    assert entry.name == expected
    assert type(entry.factory()) is _old_chain()


@pytest.mark.parametrize(
    "kwargs", [{"hip": "7.0"}, {}, {"cuda_version": None}], ids=["hip", "cuda", "none"],
)
def test_built_for_agrees_with_the_build_detector(build, kwargs):
    build(**kwargs)
    built = [e.name for e in registered_platforms() if e.in_tree and e.name != "cpu" and e.built_for()]
    assert built == ([] if _built_for().is_cpu() else [_built_for().name])


# --- MSTAR_PLATFORM ----------------------------------------------------------------------------


def test_mstar_platform_names_a_registered_platform_even_on_a_gpu_host(build):
    build(cuda=True)
    assert select_platform({"MSTAR_PLATFORM": "cpu"}).name == "cpu"


def test_an_unknown_mstar_platform_is_an_error_that_lists_the_known_ones(install):
    install("dummy")
    with pytest.raises(ValueError, match=r"MSTAR_PLATFORM='tpu' names no platform") as err:
        select_platform({"MSTAR_PLATFORM": "tpu"})
    for name in ("cpu", "cuda", "dummy", "rocm", "xpu"):
        assert name in str(err.value)


def test_naming_a_plugin_loads_only_that_plugin(build, install):
    build(cuda=True)
    install("dummy", PLUGINS[0])
    install("other", PLUGINS[1])
    assert select_platform({"MSTAR_PLATFORM": "dummy"}).name == "dummy"
    assert PLUGINS[0] in sys.modules and PLUGINS[1] not in sys.modules


def test_naming_an_in_tree_platform_loads_no_plugin(install):
    install("dummy")
    select_platform({"MSTAR_PLATFORM": "cpu"})
    assert PLUGINS[0] not in sys.modules


# --- plugins under detection -------------------------------------------------------------------


def test_an_in_tree_accelerator_beats_an_available_plugin_and_loads_none(build, install):
    build(cuda=True, hip="7.0")
    install("dummy")
    assert select_platform({}).name == "rocm"
    assert PLUGINS[0] not in sys.modules


def test_with_no_in_tree_accelerator_the_available_plugin_wins(install):
    install("dummy")
    entry = select_platform({})
    assert entry.name == "dummy" and not entry.in_tree
    assert entry.factory().name == "dummy"


def test_an_unavailable_plugin_leaves_the_cpu(install):
    install("dummy", available=False)
    assert select_platform({}).name == "cpu"


def test_two_available_plugins_are_an_error(install):
    install("dummy", PLUGINS[0])
    install("other", PLUGINS[1])
    with pytest.raises(RuntimeError, match=r"several platforms can run here \(dummy, other\); set MSTAR_PLATFORM"):
        select_platform({})


def test_a_plugin_that_returns_none_registers_nothing(install):
    install("dummy", body="return None")
    load_plugins()
    assert "dummy" not in {e.name for e in registered_platforms()}
    with pytest.raises(ValueError, match="names no platform"):
        select_platform({"MSTAR_PLATFORM": "dummy"})


def test_a_broken_plugin_is_skipped_in_a_scan_and_raises_when_named(install, caplog):
    install("dummy", body="raise ImportError('no libtpu')")
    with caplog.at_level(logging.WARNING, logger=registry.__name__):
        assert select_platform({}).name == "cpu"
    assert "platform plugin dummy" in caplog.text and "no libtpu" in caplog.text
    registry._LOADED.clear()
    with pytest.raises(ImportError, match="no libtpu"):
        select_platform({"MSTAR_PLATFORM": "dummy"})


# --- the contract version ----------------------------------------------------------------------


def test_a_wrong_api_version_is_refused_with_both_numbers():
    entry = PlatformEntry("future", CpuPlatform, "cpu", built_for=lambda: True, available=lambda: True,
                          api_version=PLATFORM_API_VERSION + 1)
    both = rf"version {PLATFORM_API_VERSION + 1}, and this M\* has version {PLATFORM_API_VERSION}"
    with pytest.raises(ValueError, match=both):
        register_platform(entry)


def test_a_plugin_for_another_api_version_is_refused_when_named(install):
    install("dummy", api_version=PLATFORM_API_VERSION + 1)
    with pytest.raises(ValueError, match="platform API version"):
        select_platform({"MSTAR_PLATFORM": "dummy"})


# --- registration ------------------------------------------------------------------------------


def test_a_name_registers_once_unless_replaced():
    entry = PlatformEntry("cpu", CpuPlatform, "cpu", built_for=lambda: True, available=lambda: True)
    with pytest.raises(ValueError, match="already registered"):
        register_platform(entry)
    register_platform(entry, replace=True)
    assert [e.name for e in registered_platforms()] == ["cuda", "rocm", "xpu", "cpu"]


def test_resolve_registered_builds_the_selected_platform(monkeypatch):
    monkeypatch.setenv("MSTAR_PLATFORM", "cpu")
    assert isinstance(resolve_registered(), CpuPlatform)
