"""``StrategySlot``: one resource family's implementations, and which one runs here."""

from types import SimpleNamespace

import pytest
import torch

from mstar.platforms import CudaPlatform, RocmPlatform
from mstar.platforms.strategy import SkipKind, StrategySlot, StrategyUnavailableError
from mstar.platforms.testing import FakePlatform, use_platform

ROCM = RocmPlatform()
FAKE = FakePlatform()


def _slot(reference: str | None = "torch") -> StrategySlot:
    slot = StrategySlot("test.op", reference=reference)
    if reference is not None:
        slot.register(reference, lambda: reference)
    return slot


def _raises(error):
    def fail(*args):
        raise error

    return fail


def _gfx(monkeypatch, arch: str) -> None:
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda device=None: SimpleNamespace(gcnArchName=f"{arch}:sramecc+:xnack-"),
    )


# --- the order: a requested name, the platform's strategies, the reference, a refusal ---------


def test_a_requested_strategy_runs_though_an_earlier_one_could():
    slot = _slot()
    slot.register("first", lambda: "first", platforms={"rocm"})
    slot.register("second", lambda: "second", platforms={"rocm"})
    assert slot.resolve("second", platform=ROCM).impl == "second"


def test_without_a_request_the_first_strategy_for_the_platform_that_works_wins():
    slot = _slot()
    slot.register("nvidia", lambda: "nvidia", platforms={"cuda"})
    slot.register("amd", lambda: "amd", platforms={"rocm"})
    resolved = slot.resolve(platform=ROCM)
    assert (resolved.slot, resolved.name, resolved.impl) == ("test.op", "amd", "amd")
    assert resolved.skipped == ()  # another platform's strategy is not a skip


def test_the_reference_runs_when_nothing_else_does():
    slot = _slot()
    slot.register("amd", lambda: "amd", platforms={"rocm"})
    assert slot.resolve(platform=FAKE).name == "torch"
    assert slot.names(FAKE) == ("torch",)


def test_a_slot_with_no_reference_and_no_candidate_refuses_naming_the_platform():
    slot = _slot(reference=None)
    slot.register("amd", lambda: "amd", platforms={"rocm"})
    with pytest.raises(StrategyUnavailableError, match=r"test\.op: no strategy runs on platform 'fake'") as err:
        slot.resolve(platform=FAKE)
    assert err.value.platform == "fake" and err.value.slot == "test.op"


def test_a_reference_that_is_not_registered_is_a_refusal():
    slot = StrategySlot("test.op", reference="torch")
    with pytest.raises(StrategyUnavailableError, match="reference is not registered"):
        slot.resolve(platform=FAKE)


# --- a requested strategy never falls back ------------------------------------------------------


@pytest.mark.parametrize(
    ("requested", "options", "reason"),
    [
        ("missing", {}, "not registered; test.op has torch, amd"),
        ("amd", {"platforms": {"cuda"}}, "registered for cuda, not fake"),
        ("amd", {"check": lambda target: "needs head_dim 128"}, "needs head_dim 128"),
        ("amd", {"factory": _raises(ImportError("no module named aiter"))}, "no module named aiter"),
        ("amd", {"verify": lambda impl: False}, "failed its verify"),
    ],
    ids=["unregistered", "other-platform", "check", "factory", "verify"],
)
def test_a_requested_strategy_that_cant_run_raises_and_never_falls_back(requested, options, reason):
    slot = _slot()  # its reference could run
    options = dict(options)
    slot.register("amd", options.pop("factory", lambda: "amd"), **options)
    with pytest.raises(StrategyUnavailableError) as err:
        slot.resolve(requested, platform=FAKE)
    message = str(err.value)
    assert f"test.op: strategy {requested!r} can't run on platform 'fake'" in message
    assert reason in message
    assert err.value.requested == requested and [s.strategy for s in err.value.skipped] == [requested]


# --- registration order is the only ranking -----------------------------------------------------


@pytest.mark.parametrize("order", [("a", "b"), ("b", "a")])
def test_registration_order_is_the_only_ranking(order):
    slot = _slot()
    for name in order:
        slot.register(name, lambda name=name: name)
    assert slot.resolve(platform=FAKE).name == order[0]
    assert slot.names(FAKE) == ("torch", *order)


def test_each_strategy_passed_over_is_recorded_with_why_in_order():
    slot = _slot()
    slot.register("refuses", lambda: "refuses", check=lambda target: f"can't run {target}")
    slot.register("missing", _raises(OSError("libfoo.so: cannot open shared object file")))
    slot.register("wrong", lambda: "wrong", verify=lambda impl: False)
    slot.register("raises_in_verify", lambda: "x", verify=_raises(RuntimeError("device-side assert")))
    resolved = slot.resolve(target="bf16 x 7", platform=FAKE)
    assert resolved.name == "torch"
    assert [(s.strategy, s.kind) for s in resolved.skipped] == [
        ("refuses", SkipKind.CHECK_REFUSED),
        ("missing", SkipKind.UNAVAILABLE),
        ("wrong", SkipKind.VERIFY_FAILED),
        ("raises_in_verify", SkipKind.UNAVAILABLE),
    ]
    assert resolved.skipped[0].reason == "can't run bf16 x 7"
    assert "libfoo.so" in resolved.skipped[1].reason


def test_a_factory_bug_is_not_mistaken_for_an_unavailable_library():
    slot = _slot()
    slot.register("buggy", _raises(ValueError("bad shape")))
    with pytest.raises(ValueError, match="bad shape"):
        slot.resolve(platform=FAKE)


def test_every_resolve_decides_afresh():
    built = []
    slot = _slot()
    slot.register("counted", lambda: built.append(1) or "counted")
    slot.resolve(platform=FAKE)
    slot.resolve(platform=FAKE)
    assert len(built) == 2


# --- platforms and architectures ----------------------------------------------------------------


@pytest.mark.parametrize(("arch", "expected"), [("gfx950", "mi355"), ("gfx942", "torch")])
def test_an_architecture_key_matches_only_that_architecture(monkeypatch, arch, expected):
    _gfx(monkeypatch, arch)
    slot = _slot()
    slot.register("mi355", lambda: "mi355", platforms={"rocm:gfx950"})
    target = SimpleNamespace(device=torch.device("cuda", 0))
    assert slot.resolve(target=target, platform=ROCM).name == expected
    assert ("mi355" in slot.names(ROCM)) is (arch == "gfx950")


def test_a_bare_platform_key_matches_every_architecture(monkeypatch):
    _gfx(monkeypatch, "gfx942")
    slot = _slot()
    slot.register("amd", lambda: "amd", platforms={"rocm"})
    assert slot.resolve(platform=ROCM).name == "amd"
    assert slot.resolve(platform=CudaPlatform()).name == "torch"


def test_the_default_platform_is_the_current_one():
    slot = _slot()
    slot.register("fake-only", lambda: "fake-only", platforms={"fake"})
    with use_platform(FakePlatform()):
        assert slot.resolve().name == "fake-only"
    assert slot.resolve().name == "torch"


# --- registration ------------------------------------------------------------------------------


def test_a_duplicate_name_is_refused():
    slot = _slot()
    with pytest.raises(ValueError, match="already has a strategy named 'torch'"):
        slot.register("torch", lambda: "again")


def test_platforms_must_be_a_collection_not_one_name():
    with pytest.raises(TypeError, match="collection"):
        _slot().register("amd", lambda: "amd", platforms="rocm")
