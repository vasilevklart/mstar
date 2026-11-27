"""FlashInfer's workspaces: the scratch a plan reserves its split-KV partials in."""

import functools
import logging
import os
import re
from collections.abc import Callable

import torch

logger = logging.getLogger(__name__)

_MIB = 1024 * 1024


@functools.cache
def warn_if_flashinfer_kernels_missing(device: torch.device) -> None:
    """Warn once when no prebuilt FlashInfer kernel (``flashinfer-jit-cache``) matches this GPU and
    the installed ``flashinfer-python``. FlashInfer then compiles each kernel on its first use,
    2 to 4 minutes of a first start on an H200 (docs/installation.rst, "Prebuilt FlashInfer
    kernels"). The sampling module stands in for the rest: every LLM sampler loads it."""
    if device.type != "cuda" or torch.version.hip is not None:
        return
    # FlashInfer's prebuilt kernels start at sm80
    if torch.cuda.get_device_capability(device)[0] < 8:
        return
    try:
        from flashinfer.jit import env

        if env.get_aot_artifacts("sampling"):
            return
        import flashinfer

        version = flashinfer.__version__
    except Exception:  # noqa: BLE001 - a FlashInfer without the lookup (or a stand-in): say nothing
        return
    logger.warning(
        "No prebuilt FlashInfer kernels (flashinfer-jit-cache) match this GPU and flashinfer-python "
        "%s, so FlashInfer compiles each kernel on its first use, which adds minutes to a first "
        "start. Install them with `flashinfer install-jit-cache-wheel --mode minimal` (it needs a "
        "torch built for CUDA 12.9 or newer; see Prebuilt FlashInfer kernels in docs/installation.rst).",
        version,
    )

# How a plan reports a short workspace. The paged wrappers can also be asked ahead of a plan
# (`workspace_size`) but the ragged one can't, so this is the report every wrapper shares. A
# FlashInfer that words it differently fails loudly with its own message (`Workspace.plan`).
_OVERFLOW = re.compile(
    r"Buffer overflow when allocating memory for (\w+) with size (\d+) "
    r"and alignment \d+, but only (\d+) bytes available"
)
# What a plan takes from the workspace we pass. The rest of its allocations are in the int
# workspace FlashInfer allocates itself, far larger than a plan's indices.
_FLOAT_ALLOCATIONS = ("tmp_v", "tmp_s", "partial_o", "partial_lse")
# A plan makes at most two allocations here, a partial output and then its log-sum-exp (tmp_v
# and tmp_s, or partial_o and partial_lse), and each plan that runs short grows the workspace
# by the one that failed, so the third plan has room for both.
_PLANS = 3


def _float_overflow(exc: RuntimeError) -> tuple[str, int, int] | None:
    """(allocation, its bytes, the bytes left) when ``exc`` is a plan short of float workspace."""
    match = _OVERFLOW.search(str(exc))
    if match is None or not match[1].endswith(_FLOAT_ALLOCATIONS):
        return None
    return match[1], int(match[2]), int(match[3])


class Workspace:
    """One FlashInfer float workspace, which wrappers plan against through `plan`.

    What a plan needs depends on its rows, segments and heads and on the GPU's SM count, and
    under a graph it is the same for every plan of a wrapper. So a workspace starts at its
    floor and a plan that overflows it grows it. ``tensor`` keeps its identity across that, so
    every wrapper on it moves too and the key holds one buffer, as large as the largest plan
    made on it.

    A captured graph records the buffer's address, so once `seal`ed it can't grow, and a plan
    that doesn't fit raises what it asked for and which setting would have made room.
    ``floor`` names that setting.
    """

    __slots__ = ("tensor", "name", "sealed", "floor")

    def __init__(
        self, tensor: torch.Tensor, name: str, sealed: bool = False, floor: str = "MSTAR_WORKSPACE_BUFFER_MB",
    ):
        self.tensor = tensor
        self.name = name
        self.sealed = sealed
        self.floor = floor

    def seal(self) -> None:
        self.sealed = True

    def plan(self, plan: Callable, *args, **kwargs):
        """``plan(*args, **kwargs)``. One that runs short of room grows this workspace to the
        bytes it had taken plus the allocation that failed, and runs again."""
        short_of = []
        while True:
            try:
                return plan(*args, **kwargs)
            except RuntimeError as exc:
                overflow = _float_overflow(exc)
                if overflow is None:
                    raise
                allocation, size, left = overflow
                asked = self.tensor.numel() - left + size
                short_of.append(f"{allocation} ({size} bytes)")
                if len(short_of) == _PLANS:
                    why = f"{_PLANS} plans ran short, of {', '.join(short_of)}"
                    raise RuntimeError(self._short(asked, why) + self._floor_hint(asked)) from exc
                self._grow(asked, exc)

    def _grow(self, asked: int, cause: RuntimeError) -> None:
        have = self.tensor.numel()
        if self.sealed:
            why = "it can't grow once a captured graph can hold its address"
            raise RuntimeError(self._short(asked, why) + self._floor_hint(asked)) from cause
        size = -(-asked // _MIB) * _MIB
        device = self.tensor.device
        if device.type == "cuda":
            # any kernel still reading it is done before its memory goes back to the allocator
            torch.cuda.synchronize(device)
        # the old buffer goes first, which leaves room on a GPU that fits one but not both
        self.tensor.data = self.tensor.new_empty(0)
        try:
            self.tensor.data = torch.empty(size, dtype=torch.uint8, device=device)
        except torch.cuda.OutOfMemoryError as oom:
            why = f"the GPU has no room to grow it: {oom}"
            try:
                # back to what it held, which every plan made on it so far fitted
                self.tensor.data = torch.empty(have, dtype=torch.uint8, device=device)
            except torch.cuda.OutOfMemoryError:
                why += f"; nor to take back the {have} bytes it held"
            raise RuntimeError(self._short(asked, why)) from cause
        logger.info(
            "FlashInfer workspace %s: %d -> %d MiB, a plan asked for %d bytes",
            self.name, have // _MIB, size // _MIB, asked,
        )

    def _short(self, asked: int, why: str) -> str:
        return (
            f"FlashInfer workspace {self.name}: a plan asks for {asked} bytes ({-(-asked // _MIB)} MiB) "
            f"and it holds {self.tensor.numel()} bytes; {why}"
        )

    def _floor_hint(self, asked: int) -> str:
        return f". It starts at {self.floor}: set that to at least {-(-asked // _MIB)}."


def workspace_bytes(declared_mb: int | None = None) -> int:
    """The floor of one FlashInfer workspace: what its resource declares, else
    ``MSTAR_WORKSPACE_BUFFER_MB`` (default 512)."""
    if declared_mb is not None:
        return declared_mb * _MIB
    return int(os.environ.get("MSTAR_WORKSPACE_BUFFER_MB", "512")) * _MIB


class WorkspacePool:
    """FlashInfer workspaces, one per (plan label, cg slot).

    Persistent: a wrapper is planned against the buffer it was built with, so these outlive
    any single step. ``owner`` names the resource in what a workspace reports.
    """

    def __init__(self, device: torch.device, size_mb: int | None = None, owner: str = "attention"):
        # the first FlashInfer resource a worker builds, ahead of any kernel it loads
        warn_if_flashinfer_kernels_missing(torch.device(device))
        self._device = device
        self._owner = owner
        self._size = workspace_bytes(size_mb)
        self._floor = "MSTAR_WORKSPACE_BUFFER_MB" if size_mb is None else f"the workspace_mb {owner} declares"
        self._workspaces: dict[str, Workspace] = {}
        self._sealed = False

    def get(self, label: str, cg_slot: int | None = None) -> Workspace:
        key = label if cg_slot is None else f"{label}_{cg_slot}"
        workspace = self._workspaces.get(key)
        if workspace is None:
            # Past the seal a key's first wrapper may still be captured, and a growth for a
            # second one on the key would free what that graph reads: so it starts sealed.
            workspace = self._workspaces[key] = Workspace(
                torch.empty(self._size, dtype=torch.uint8, device=self._device), f"{self._owner}/{key}",
                sealed=self._sealed, floor=self._floor,
            )
        return workspace

    def seal(self) -> None:
        """The workspaces here keep their addresses from now on, for the graphs about to be
        captured over them, and so does any made later."""
        self._sealed = True
        for workspace in self._workspaces.values():
            workspace.seal()
        if self._workspaces:
            logger.info(
                "FlashInfer workspaces of %s: %d, %d MiB in all", self._owner, len(self._workspaces),
                sum(workspace.tensor.numel() for workspace in self._workspaces.values()) // _MIB,
            )
