"""Which ptxas Triton runs.

ptxas before 13.1.115 miscompiles a floor division of a periodic index under a
range select, as in Inductor's fused interleaved mRoPE table. It is ptxas, not
Triton's codegen: the same PTX gives the same wrong SASS under ptxas 12.8, 13.0
and 13.1.80. It shows on sm_103 (B300) and on sm_90 (H200). Triton has two
ptxas variables, ``TRITON_PTXAS_PATH`` for GPUs before sm_100 and
``TRITON_PTXAS_BLACKWELL_PATH`` for sm_100 and up, and Triton 3.6 and 3.7 bundle
an affected ptxas for both: 12.8.93 below sm_100, and 12.9.86 (3.6) or 13.1.80
(3.7) from sm_100. Inductor points Triton at torch's ``torch/bin/ptxas`` through
``TRITON_PTXAS_PATH`` alone, and only the CUDA 13 builds of torch ship one.
"""

from __future__ import annotations

import functools
import importlib.util
import logging
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import torch

logger = logging.getLogger(__name__)

CACHE_TAG_ENV = "TORCH_COMPILE_CACHE_KEY_TAG"
FIXED_PTXAS = (13, 1, 115)
# The oldest Linux driver for CUDA 13.0. A cubin from a 13.x ptxas does not load
# under an older driver, so a 13.x ptxas is only chosen above it.
CUDA13_DRIVER = 580


class TritonPtxas(NamedTuple):
    """One of the two ptxas variables Triton reads."""

    env: str  # the variable naming the ptxas Triton runs
    knob: str  # its name in ``triton.knobs.nvidia``, which holds the bundled default
    tag: str  # what the Inductor cache tag calls it
    gpus: str  # the GPUs Triton runs it for, for the log


PTXAS = TritonPtxas("TRITON_PTXAS_PATH", "ptxas", "ptxas", "GPUs before sm_100")
BLACKWELL_PTXAS = TritonPtxas("TRITON_PTXAS_BLACKWELL_PATH", "ptxas_blackwell", "ptxas-blackwell", "sm_100 and up")
BLACKWELL_PTXAS_ENV = BLACKWELL_PTXAS.env


class NvidiaGpus(NamedTuple):
    """What nvidia-smi reports about the GPUs present."""

    capability_majors: frozenset[int]  # 9 for sm_90, 10 for sm_100 and sm_103
    driver_major: int | None  # 580 for driver 580.95.05; None when it does not say


class PtxasDefault(NamedTuple):
    """The default for one of Triton's ptxas variables."""

    path: str | None  # the ptxas to set the variable to; None keeps the variable as it is
    version: tuple[int, int, int]  # the version Triton then runs, which the cache tag names


@functools.cache
def ptxas_version(path: str) -> tuple[int, int, int] | None:
    """(major, minor, build) from ``ptxas --version``; None if it does not run."""
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"V(\d+)\.(\d+)\.(\d+)", out)
    return tuple(int(part) for part in match.groups()) if match else None


def _nvidia_gpus() -> NvidiaGpus | None:
    # nvidia-smi rather than torch.cuda, which would initialize the CUDA runtime
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    rows = re.findall(r"^\s*(\d+)\.\d+\s*,\s*(\d+)", out, flags=re.MULTILINE)
    if not rows:
        return None
    return NvidiaGpus(frozenset(int(cap) for cap, _ in rows), min(int(driver) for _, driver in rows))


def _triton_bundled_ptxas(tool: TritonPtxas) -> str | None:
    try:
        from triton import knobs

        return knobs.nvidia.knob_descriptors[tool.knob].default_path
    except (ImportError, AttributeError, KeyError):
        return None


def _candidate_ptxas() -> list[str]:
    """The ptxas binaries that may postdate the fix, in the order they are
    preferred: torch's own, the nvidia-cuda-nvcc wheel's, then a CUDA toolkit's."""
    paths = [str(Path(torch.__file__).parent / "bin" / "ptxas")]
    # `pip install "nvidia-cuda-nvcc>=13.1.115"` puts one at nvidia/cu13/bin/ptxas
    if (nvidia := importlib.util.find_spec("nvidia")) and nvidia.submodule_search_locations:
        paths += [os.path.join(root, "cu13", "bin", "ptxas") for root in nvidia.submodule_search_locations]
    paths += [os.path.join(root, "bin", "ptxas") for var in ("CUDA_HOME", "CUDA_PATH") if (root := os.environ.get(var))]
    paths.append("/usr/local/cuda/bin/ptxas")
    if on_path := shutil.which("ptxas"):
        paths.append(on_path)
    return list(dict.fromkeys(paths))


def _driver_loads(version: tuple[int, int, int], driver_major: int | None) -> bool:
    return version[0] < 13 or driver_major is None or driver_major >= CUDA13_DRIVER


def _version_str(version: tuple[int, int, int] | None) -> str:
    return ".".join(map(str, version)) if version else "of unknown version"


def _ptxas_default(tool: TritonPtxas, driver_major: int | None) -> PtxasDefault | None:
    fixed = _version_str(FIXED_PTXAS)
    if path := os.environ.get(tool.env):
        version = ptxas_version(path)
        if version is not None and version < FIXED_PTXAS:
            logger.warning(
                "Triton runs ptxas %s for %s (%s from %s), which predates %s and can miscompile integer "
                "floor division",
                _version_str(version), tool.gpus, path, tool.env, fixed,
            )
        else:
            logger.info("Triton runs ptxas %s for %s (%s from %s)", _version_str(version), tool.gpus, path, tool.env)
        # A cache the bundled ptxas wrote would keep serving its kernels, so the
        # tag names this ptxas too.
        return None if version is None else PtxasDefault(None, version)
    bundled = _triton_bundled_ptxas(tool)
    bundled_version = ptxas_version(bundled) if bundled else None
    if bundled_version is None:
        return None
    if bundled_version >= FIXED_PTXAS:
        logger.info("Triton runs its bundled ptxas %s for %s", _version_str(bundled_version), tool.gpus)
        return None
    candidates = _candidate_ptxas()
    for path in candidates:
        version = ptxas_version(path)
        if version is not None and version >= FIXED_PTXAS and _driver_loads(version, driver_major):
            logger.info(
                "Triton runs ptxas %s for %s (%s): its bundled ptxas %s predates %s, which fixes a miscompile "
                "of integer floor division",
                _version_str(version), tool.gpus, path, _version_str(bundled_version), fixed,
            )
            return PtxasDefault(path, version)
    logger.warning(
        "Triton runs its bundled ptxas %s for %s, which can miscompile integer floor division (Inductor's "
        "interleaved mRoPE tables come out wrong); none of %s is ptxas %s or later that driver %s loads. "
        'Set %s to one, or `pip install "nvidia-cuda-nvcc>=%s"` (a 13.x ptxas needs driver %d or later)',
        _version_str(bundled_version), tool.gpus, ", ".join(candidates), fixed, driver_major or "?", tool.env,
        fixed, CUDA13_DRIVER,
    )
    return None


@functools.cache
def ptxas_env() -> dict[str, str | Callable[[str | None], str]]:
    """Each ptxas variable Triton reads for the GPUs present set to a ptxas at
    13.1.115 or later, torch's or a CUDA toolkit's, when Triton's bundled ptxas
    predates it, and an Inductor cache tag naming the ptxas Triton then runs.
    A variable already set is kept, and only named in the tag. Falling back to
    the bundled ptxas, for want of a fixed one the driver loads, is logged as a
    warning."""
    gpus = _nvidia_gpus()
    if gpus is None:
        return {}
    # Triton 3.6 to 3.8 run the Blackwell ptxas from sm_100 up and the other
    # below (3.9 moves sm_90 to the Blackwell one)
    tools = []
    if min(gpus.capability_majors) < 10:
        tools.append(PTXAS)
    if max(gpus.capability_majors) >= 10:
        tools.append(BLACKWELL_PTXAS)
    env: dict[str, str | Callable[[str | None], str]] = {}
    tags = []
    for tool in tools:
        default = _ptxas_default(tool, gpus.driver_major)
        if default is None:
            continue
        if default.path is not None:
            env[tool.env] = default.path
        tags.append(f"{tool.tag}-{_version_str(default.version)}")
    if tags:
        env[CACHE_TAG_ENV] = _cache_tag("-".join(tags))
    return env


def _cache_tag(suffix: str) -> Callable[[str | None], str]:
    """Inductor's graph cache key ignores the ptxas, so untagged it keeps serving
    kernels another ptxas compiled. torch reads the tag at import, so it takes
    effect in the processes spawned after it is set. A tag the user set keeps
    its own split of the cache, with the ptxas appended."""

    def tag(current: str | None) -> str:
        if not current:
            return suffix
        return current if current.endswith(suffix) else f"{current}-{suffix}"

    return tag
