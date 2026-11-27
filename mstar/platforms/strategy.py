"""Strategy slots: the implementations of one resource family or op, and which one runs here.

A slot lists implementations in the order it prefers them. Each says which platforms it is for,
why it can't run a target (``check``, a reason or None, like ``Impl.check`` in the kernel
registry) and, optionally, how to test a built instance (``verify``). One of them can be the
slot's reference, which runs wherever the others can't::

    PAGE_COPIER = StrategySlot("kv.page_copier", reference="torch")
    PAGE_COPIER.register("libcudart", lambda: CudartCopier(), platforms={"cuda"}, verify=_round_trips)
    PAGE_COPIER.register("torch", lambda: TorchCopier())

    copier = PAGE_COPIER.resolve().impl

``resolve`` decides afresh on each call, so the caller decides when to ask (once, when its
resource is built) and logs what it got. Registration order is the only ranking.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, NamedTuple, TypeVar

from mstar import platforms
from mstar.platforms.interface import Platform

T = TypeVar("T")

# what an implementation that can't load raises while it is built or verified: a missing
# package or library, a missing symbol, or a runtime that refuses
BUILD_ERRORS = (ImportError, OSError, AttributeError, RuntimeError)


class SkipKind(Enum):
    """Why a strategy was passed over."""

    NOT_REGISTERED = "not registered"
    OTHER_PLATFORM = "not for this platform"
    CHECK_REFUSED = "its check refused the target"
    UNAVAILABLE = "building or verifying it raised"
    VERIFY_FAILED = "its verify returned False"


class Skip(NamedTuple):
    strategy: str
    kind: SkipKind
    reason: str


@dataclass(frozen=True)
class Resolved(Generic[T]):
    """The strategy a slot picked, and the ones it passed over on the way, in order."""

    slot: str
    name: str
    impl: T
    skipped: tuple[Skip, ...]


class StrategyUnavailableError(RuntimeError):
    """No strategy of a slot can run here, or the one named can't."""

    def __init__(self, slot: str, platform: str, skipped: tuple[Skip, ...], requested: str | None = None):
        self.slot, self.platform, self.skipped, self.requested = slot, platform, skipped, requested
        what = f"strategy {requested!r}" if requested is not None else "no strategy"
        reasons = "; ".join(f"{s.strategy}: {s.reason}" for s in skipped) or "none is registered for it"
        verb = "can't run" if requested is not None else "runs"
        super().__init__(f"{slot}: {what} {verb} on platform {platform!r} ({reasons})")


@dataclass(frozen=True)
class _Strategy(Generic[T]):
    name: str
    factory: Callable[[], T]
    platforms: frozenset[str] | None
    check: Callable[[Any], str | None] | None
    verify: Callable[[T], bool] | None


class StrategySlot(Generic[T]):
    """The implementations of one resource family or op, and how one is picked."""

    def __init__(self, name: str, *, reference: str | None = None):
        self.name = name
        self.reference = reference
        self._strategies: dict[str, _Strategy[T]] = {}

    def register(
        self,
        name: str,
        factory: Callable[[], T],
        *,
        platforms: Collection[str] | None = None,
        check: Callable[[Any], str | None] | None = None,
        verify: Callable[[T], bool] | None = None,
    ) -> None:
        """Add a strategy after those already registered.

        ``platforms`` names the platforms it is for (``"rocm"``, or ``"rocm:gfx950"`` for one
        architecture); None is every platform. ``check(target)`` says why it can't run the
        target, None when it can. ``factory()`` builds it, and its imports belong inside it,
        so registering one loads nothing. ``verify(impl)`` tests a built instance.
        """
        if name in self._strategies:
            raise ValueError(f"{self.name} already has a strategy named {name!r}")
        if isinstance(platforms, str):
            raise TypeError(f"{self.name}: platforms takes a collection of names, got {platforms!r}")
        self._strategies[name] = _Strategy(
            name, factory, None if platforms is None else frozenset(platforms), check, verify,
        )

    def names(self, platform: Platform | None = None, target: Any = None) -> tuple[str, ...]:
        """The strategies registered for ``platform`` (by default the current one), in order."""
        platform = _platform(platform)
        return tuple(s.name for s in self._strategies.values() if _for(s, platform, target))

    def resolve(self, requested: str | None = None, target: Any = None, *,
                platform: Platform | None = None) -> Resolved[T]:
        """The strategy to run on ``target`` (by default the current platform's device).

        A ``requested`` name runs or raises: it never falls back. Otherwise the first
        strategy for this platform whose check passes, that builds and that verifies, and
        then the reference. Raises :class:`StrategyUnavailableError` with every reason when
        nothing runs. An architecture in ``platforms`` is matched against
        ``target.device`` when the target has one.
        """
        platform = _platform(platform)
        if requested is not None:
            return self._resolve_named(requested, target, platform)
        skipped: list[Skip] = []
        for strategy in self._strategies.values():
            if strategy.name == self.reference or not _for(strategy, platform, target):
                continue
            impl, skip = _try(strategy, target)
            if skip is None:
                return Resolved(self.name, strategy.name, impl, tuple(skipped))
            skipped.append(skip)
        if self.reference is not None:
            reference = self._strategies.get(self.reference)
            if reference is None:
                skipped.append(Skip(self.reference, SkipKind.NOT_REGISTERED, "the slot's reference is not registered"))
            elif not _for(reference, platform, target):
                skipped.append(Skip(reference.name, SkipKind.OTHER_PLATFORM, _not_for(reference, platform)))
            else:
                impl, skip = _try(reference, target)
                if skip is None:
                    return Resolved(self.name, reference.name, impl, tuple(skipped))
                skipped.append(skip)
        raise StrategyUnavailableError(self.name, platform.name, tuple(skipped))

    def _resolve_named(self, name: str, target: Any, platform: Platform) -> Resolved[T]:
        strategy = self._strategies.get(name)
        if strategy is None:
            known = ", ".join(self._strategies) or "none"
            skip = Skip(name, SkipKind.NOT_REGISTERED, f"not registered; {self.name} has {known}")
        elif not _for(strategy, platform, target):
            skip = Skip(name, SkipKind.OTHER_PLATFORM, _not_for(strategy, platform))
        else:
            impl, skip = _try(strategy, target)
            if skip is None:
                return Resolved(self.name, name, impl, ())
        raise StrategyUnavailableError(self.name, platform.name, (skip,), requested=name)

    def __repr__(self) -> str:
        return f"StrategySlot({self.name!r}, {list(self._strategies)})"


def _platform(platform: Platform | None) -> Platform:
    if platform is None or platform is platforms.current_platform:
        return platforms.resolve_platform()
    return platform


def _for(strategy: _Strategy, platform: Platform, target: Any) -> bool:
    if strategy.platforms is None:
        return True
    arch = None
    for key in strategy.platforms:
        name, _, want = key.partition(":")
        if name != platform.name:
            continue
        if not want:
            return True
        if arch is None:
            arch = platform.arch(getattr(target, "device", None))
        if arch == want:
            return True
    return False


def _not_for(strategy: _Strategy, platform: Platform) -> str:
    return f"registered for {', '.join(sorted(strategy.platforms or ()))}, not {platform.name}"


def _try(strategy: _Strategy[T], target: Any) -> tuple[T | None, Skip | None]:
    if strategy.check is not None and (reason := strategy.check(target)) is not None:
        return None, Skip(strategy.name, SkipKind.CHECK_REFUSED, reason)
    try:
        impl = strategy.factory()
        verified = strategy.verify is None or strategy.verify(impl)
    except BUILD_ERRORS as e:
        return None, Skip(strategy.name, SkipKind.UNAVAILABLE, str(e) or type(e).__name__)
    if not verified:
        return None, Skip(strategy.name, SkipKind.VERIFY_FAILED, "failed its verify")
    return impl, None
