"""The process env defaults that point Triton at a ptxas at 13.1.115 or later.

ptxas before 13.1.115 miscompiles a floor division of a periodic index under a
range select when a thread holds two elements, on every GPU Triton 3.7.1 runs
it for: its bundled 12.8.93 below sm_100 (H200 fails these CUDA tests) and its
13.1.80 from sm_100 (so does B300). The CUDA tests run such kernels at that
launch config with the defaults applied, so they fail on a box where no fixed
ptxas the driver loads is found, which then serves wrong rotary tables too.
"""

import logging
import os
import stat

import pytest
import torch
import triton
import triton.language as tl

from mstar import platforms
from mstar.platforms import ptxas
from mstar.platforms.interface import CudaPlatform
from mstar.platforms.ptxas import BLACKWELL_PTXAS, PTXAS, NvidiaGpus

BUNDLED = "/site-packages/triton/backends/nvidia/bin/ptxas"
BUNDLED_BLACKWELL = "/site-packages/triton/backends/nvidia/bin/ptxas-blackwell"
TORCH_PTXAS = os.path.join(os.path.dirname(torch.__file__), "bin", "ptxas")
TOOLKIT_PTXAS = "/usr/local/cuda/bin/ptxas"
USER_PTXAS = "/opt/cuda/bin/ptxas"
TAG = "TORCH_COMPILE_CACHE_KEY_TAG"
TRITON_37 = {BUNDLED: (12, 8, 93), BUNDLED_BLACKWELL: (13, 1, 80)}
CU132, CU129 = (13, 2, 78), (12, 9, 86)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (Triton)")


@pytest.fixture(autouse=True)
def _undecided():
    ptxas.ptxas_env.cache_clear()
    yield
    ptxas.ptxas_env.cache_clear()


# (GPU capability majors, driver major, bundled versions, torch's, the CUDA toolkit's,
#  a variable the user set, the defaults wanted, whether a warning says Triton may miscompile)
CASES = {
    "blackwell, torch's is fixed": (
        {10}, 580, TRITON_37, CU132, None, None, {BLACKWELL_PTXAS.env: TORCH_PTXAS, TAG: "ptxas-blackwell-13.2.78"},
        False),
    "hopper, torch's is fixed": (
        {9}, 580, TRITON_37, CU132, None, None, {PTXAS.env: TORCH_PTXAS, TAG: "ptxas-13.2.78"}, False),
    "hopper, CUDA 12 torch, the toolkit's is fixed": (
        {9}, 580, TRITON_37, CU129, (13, 1, 115), None, {PTXAS.env: TOOLKIT_PTXAS, TAG: "ptxas-13.1.115"}, False),
    "hopper, none is fixed": ({9}, 580, TRITON_37, CU129, (13, 1, 80), None, {}, True),
    "hopper, the driver predates CUDA 13": ({9}, 570, TRITON_37, CU129, (13, 1, 115), None, {}, True),
    "hopper and blackwell": (
        {9, 10}, 580, TRITON_37, CU132, None, None,
        {PTXAS.env: TORCH_PTXAS, BLACKWELL_PTXAS.env: TORCH_PTXAS, TAG: "ptxas-13.2.78-ptxas-blackwell-13.2.78"},
        False),
    "triton 3.8 on blackwell": (
        {10}, 580, {BUNDLED: (12, 9, 86), BUNDLED_BLACKWELL: (13, 3, 33)}, CU132, None, None, {}, False),
    "bundled fixed": ({9}, 580, {BUNDLED: (13, 1, 115), BUNDLED_BLACKWELL: (13, 1, 115)}, CU132, None, None, {}, False),
    "bundled unknown": ({9, 10}, 580, {}, CU132, None, None, {}, False),
    "no nvidia gpu": (set(), None, TRITON_37, CU132, None, None, {}, False),
    "already set": (
        {10}, 580, TRITON_37, CU132, None, (BLACKWELL_PTXAS.env, (13, 2, 51)), {TAG: "ptxas-blackwell-13.2.51"},
        False),
    "already set, too old": ({9}, 580, TRITON_37, CU132, None, (PTXAS.env, CU129), {TAG: "ptxas-12.9.86"}, True),
}


@pytest.mark.parametrize(
    ("majors", "driver", "bundled", "torch_version", "toolkit_version", "user", "want", "warns"),
    list(CASES.values()), ids=list(CASES),
)
def test_triton_runs_a_fixed_ptxas_on_every_gpu_whose_bundled_one_predates_it(
    monkeypatch, caplog, majors, driver, bundled, torch_version, toolkit_version, user, want, warns,
):
    versions = {**bundled, TORCH_PTXAS: torch_version, TOOLKIT_PTXAS: toolkit_version}
    gpus = NvidiaGpus(frozenset(majors), driver) if majors else None
    monkeypatch.setattr(ptxas, "_nvidia_gpus", lambda: gpus)
    bundled_paths = {PTXAS: BUNDLED, BLACKWELL_PTXAS: BUNDLED_BLACKWELL}
    monkeypatch.setattr(ptxas, "_triton_bundled_ptxas", bundled_paths.get)
    monkeypatch.setattr(ptxas, "_candidate_ptxas", lambda: [TORCH_PTXAS, TOOLKIT_PTXAS])
    for tool in (PTXAS, BLACKWELL_PTXAS):
        monkeypatch.delenv(tool.env, raising=False)
    if user:
        env, version = user
        monkeypatch.setenv(env, USER_PTXAS)
        versions[USER_PTXAS] = version
    monkeypatch.setattr(ptxas, "ptxas_version", versions.get)

    with caplog.at_level(logging.INFO, logger=ptxas.__name__):
        defaults = {k: v(None) if callable(v) else v for k, v in CudaPlatform().process_env_defaults().items()}
    assert {k: v for k, v in defaults.items() if k in (PTXAS.env, BLACKWELL_PTXAS.env, TAG)} == want
    warned = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert bool(warned) == warns, warned
    assert all("miscompile" in message for message in warned)


def test_nvidia_smi_gives_each_capability_and_the_oldest_driver(monkeypatch, tmp_path):
    fake = tmp_path / "nvidia-smi"
    fake.write_text("#!/bin/sh\nprintf '9.0, 575.57.08\\n10.3, 580.95.05\\n'\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert ptxas._nvidia_gpus() == NvidiaGpus(frozenset({9, 10}), 575)
    fake.write_text("#!/bin/sh\nexit 9\n")  # no driver loaded
    assert ptxas._nvidia_gpus() is None


@pytest.mark.parametrize(
    ("current", "want"),
    [
        (None, "ptxas-blackwell-13.2.78"),
        ("mine", "mine-ptxas-blackwell-13.2.78"),
        ("mine-ptxas-blackwell-13.2.78", "mine-ptxas-blackwell-13.2.78"),  # a worker applies it again
    ],
)
def test_the_ptxas_is_appended_once_to_a_tag_the_user_set(current, want):
    tag = ptxas._cache_tag("ptxas-blackwell-13.2.78")
    assert tag(current) == want


def test_ptxas_version_reads_the_build(tmp_path):
    fake = tmp_path / "ptxas"
    fake.write_text("#!/bin/sh\necho 'Cuda compilation tools, release 13.1, V13.1.80'\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    assert ptxas.ptxas_version(str(fake)) == (13, 1, 80)
    assert ptxas.ptxas_version(str(tmp_path / "missing")) is None


@pytest.fixture
def ptxas_default(monkeypatch, tmp_path):
    """M*'s process env defaults applied as a worker applies them, with a fresh
    Triton cache so no kernel another ptxas compiled is reused."""
    if os.environ.get(PTXAS.env) == TORCH_PTXAS:
        # Inductor's own pick in a process that compiled before, not a user's choice
        monkeypatch.delenv(PTXAS.env)
    for key in (PTXAS.env, BLACKWELL_PTXAS.env, TAG):
        # keep a value already set; otherwise set then delete, so teardown removes it
        monkeypatch.setenv(key, os.environ.get(key, ""))
        if not os.environ[key]:
            monkeypatch.delenv(key)
    platforms.apply_process_env_defaults()
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))


def _ptxas_that_ran() -> str:
    """Which ptxas Triton runs for this GPU, for a failure message."""
    from triton.backends.nvidia.compiler import get_ptxas

    major, minor = torch.cuda.get_device_capability()
    path = get_ptxas(major * 10 + minor).path
    return (f"Triton ran ptxas {ptxas._version_str(ptxas.ptxas_version(path))} ({path}) for sm_{major}{minor}; "
            f"ptxas before 13.1.115 miscompiles this, see the warning from {ptxas.__name__}")


@triton.jit
def _floor_div_under_range_select(out_ptr, n, XBLOCK: tl.constexpr):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    x0 = xindex % 64
    a = x0 - 1
    q = a // 3
    floor = tl.where((a % 3 != 0) & (a < 0), q - 1, q)  # Inductor's div_floor_integer
    tl.store(out_ptr + xindex, tl.where((x0 >= 1) & (x0 < 60), floor, 0), xindex < n)


@needs_cuda
@pytest.mark.parametrize(("xblock", "num_warps"), [(64, 1), (256, 4), (1024, 16)])  # two elements per thread
def test_floor_division_under_a_range_select(ptxas_default, xblock, num_warps):
    n = 64 * 1024
    x0 = torch.arange(n, device="cuda") % 64
    want = torch.where((x0 >= 1) & (x0 < 60), torch.div(x0 - 1, 3, rounding_mode="floor"), 0).int()
    out = torch.empty(n, device="cuda", dtype=torch.int32)
    _floor_div_under_range_select[(triton.cdiv(n, xblock),)](out, n, XBLOCK=xblock, num_warps=num_warps)
    assert torch.equal(out, want), _ptxas_that_ran()


def _elems_per_thread(config) -> float:
    block = 1
    for key in ("XBLOCK", "YBLOCK", "ZBLOCK"):
        block *= config.kwargs.get(key, 1)
    return block / (32 * config.num_warps)


@pytest.fixture
def two_elements_per_thread(monkeypatch, torch_flag):
    """Inductor's pointwise kernels limited to their candidate configs giving
    each thread two elements; returns a list with one entry per limited kernel."""
    from torch._inductor.runtime import triton_heuristics
    from torch._inductor.runtime.hints import HeuristicType

    cached_autotune = triton_heuristics.cached_autotune
    limited = []

    def two_only(size_hints, configs, *args, **kwargs):
        two = [c for c in configs if _elems_per_thread(c) == 2]
        if kwargs.get("heuristic_type") == HeuristicType.POINTWISE and two:
            configs = two
            limited.append(len(two))
        return cached_autotune(size_hints, configs, *args, **kwargs)

    monkeypatch.setattr(triton_heuristics, "cached_autotune", two_only)
    # compile in this process, where the limit applies
    torch_flag(torch._inductor.config, "compile_threads", 1)
    return limited


@needs_cuda
@pytest.mark.parametrize("positions", ["121-frame video", "small table"])
@torch.no_grad()
def test_compiled_interleaved_mrope_matches_eager(ptxas_default, two_elements_per_thread, positions):
    from torch._inductor.utils import fresh_cache

    from mstar.model.cosmos3.components.packing import build_static_inputs
    from mstar.model.cosmos3.components.transformer import Cosmos3RotaryEmbedding
    from mstar.model.cosmos3.config import Cosmos3Config

    cfg = Cosmos3Config(rope_theta=1e8)  # Edge's mRoPE
    rot = Cosmos3RotaryEmbedding(cfg.head_dim, cfg.rope_theta, cfg.rope_axes_dim)
    if positions == "121-frame video":  # 832x480 after a short prompt, as served
        pos = build_static_inputs(list(range(40)), (1, 48, 31, 30, 52), cfg, 4, 24.0, "cuda")["vision_mrope_ids"]
    else:  # 200 x 64 elements, where every pointwise candidate holds two per thread
        n = 200
        pos = torch.stack([torch.arange(n) // 50, (torch.arange(n) // 10) % 5, torch.arange(n) % 10]) + 40
        pos = pos.float().cuda()

    def table(p):
        cos, sin = rot(p.unsqueeze(1), p.device, torch.float32)
        return cos.squeeze(0), sin.squeeze(0)

    cos_e, sin_e = table(pos)
    torch._dynamo.reset()
    with fresh_cache():
        cos_c, sin_c = torch.compile(table, dynamic=False)(pos)
    torch._dynamo.reset()
    assert two_elements_per_thread, "no pointwise kernel ran at two elements per thread"
    # rounding differs by up to ~1e-4 at video positions; the miscompile is off by up to 2
    for compiled, eager in ((cos_c, cos_e), (sin_c, sin_e)):
        torch.testing.assert_close(compiled, eager, rtol=0, atol=1e-3, msg=lambda m: f"{m}\n{_ptxas_that_ran()}")
