"""The chunked GDN prefill reading and writing the pool in place.

On SM90, over an fp32 pool with a sink slot, `GDNPrefillWrapper.run` hands FlashInfer's chunked
kernel the pool and each row's slot (``state_indices``) rather than gathering a packed copy of the
rows' states and scattering the finals back (`GDNManager._prefill_in_place`). What has to hold:
the same outputs, and the same state in every slot but the sink, as the copies give. That covers
a fresh row whose slot holds something, which the gather reads as zeros and which `run` therefore
zeroes before the kernel reads it, and padding rows on the sink, which the copies read as zeros
and the kernel reads as the sink holds it.

The CPU tests run the plumbing against `gdn_cpu`'s stand-in, which takes ``state_indices`` as the
kernel does: it reads whatever a slot holds and refuses a slot outside the pool. The GPU tests
compare the two paths on FlashInfer's kernels, bit for bit, and need SM90.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.linear_attn import gdn_cpu, wrappers
from mstar.engine.resources.linear_attn.config import (
    LinearAttnConfig,
    LinearAttnSpec,
    LinearAttnVariant,
)
from mstar.engine.resources.linear_attn.gdn import (
    GDNManager,
    _flashinfer_takes_sm90_state_indices,
    _prefill_reads_pool,
)
from mstar.engine.resources.linear_attn.wrappers import GDNMixedWrapper, GDNPrefillWrapper
from mstar.engine.resources.recurrent.config import (
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
)
from mstar.engine.resources.recurrent.pool import NO_SLOT, SINK_SLOT
from mstar.engine.resources.spec import CombinedSplit
from mstar.utils.causal_conv1d import PAD_SLOT_ID

CPU = torch.device("cpu")
SLOTS = 12
# (key heads, value heads, head dim): a key head serving two value heads, as Qwen3.5 has; on
# a GPU Qwen3.5-9B's own, at the only head dim the chunked kernel takes
CPU_GEO = (2, 4, 16)
GPU_GEO = (16, 32, 128)

# (span, slot, has_state): a continuing chunk, a fresh prompt, a one-token chunk that resumes,
# and a fresh one-token prompt
PREFILL = [(5, 3, True), (6, 7, False), (1, 9, True), (1, 4, False)]
# a mixed step as the engine orders it: decode rows, then prompt rows
MIXED = [(1, 2, True), (1, 5, True), (4, 7, True), (3, 8, False), (1, 10, False)]
MAX_PROMPTS = 4


def _cols(layout):
    return [s for s, _, _ in layout], [slot for _, slot, _ in layout], [h for _, _, h in layout]


def _slots(layout) -> set[int]:
    return set(_cols(layout)[1])


def _pool(geo, device) -> torch.Tensor:
    # every slot holds something, a fresh row's too: what the copies read as zeros
    _, v_heads, dim = geo
    torch.manual_seed(0)
    return torch.randn(SLOTS, v_heads, dim, dim, device=device) * 0.05


def _inputs(tokens: int, geo, device) -> dict:
    k_heads, v_heads, dim = geo
    torch.manual_seed(1)
    kw = dict(device=device, dtype=torch.bfloat16)
    norm = lambda t: torch.nn.functional.normalize(t.float(), dim=-1).to(t.dtype)  # noqa: E731
    return dict(
        q=norm(torch.randn(tokens, k_heads, dim, **kw)),
        k=norm(torch.randn(tokens, k_heads, dim, **kw)),
        v=torch.randn(tokens, v_heads, dim, **kw),
        a=torch.randn(tokens, v_heads, **kw),
        b=torch.randn(tokens, v_heads, **kw),
        # fp32, as the model keeps them (the decode kernel requires it)
        a_log=torch.randn(v_heads, device=device) * 0.1,
        dt_bias=torch.randn(v_heads, device=device) * 0.1,
    )


def _padded(layout, inputs: dict, tokens: int, rows: int, bucket_tokens: int):
    """``layout`` and ``inputs`` widened to a bucket: padding rows on the sink, a zeroed tail."""
    spans, slots, has = _cols(layout)
    pad = rows - len(spans)
    spans, slots, has = spans + [0] * pad, slots + [SINK_SLOT] * pad, has + [False] * pad
    widened = {
        name: torch.cat((t, t.new_zeros(bucket_tokens - tokens, *t.shape[1:]))) if t.shape[0] == tokens else t
        for name, t in inputs.items()
    }
    return spans, slots, has, widened


def _prefill(in_place: bool, layout, bucket: bool, geo=CPU_GEO, device=CPU, stage_inputs: bool = False,
             snapshots=(), every: int = 0):
    """One chunked step over ``layout``: its output and the pool after it. In a bucket, two
    padding rows on the sink ride along."""
    spans, slots, has = _cols(layout)
    tokens = sum(spans)
    inputs = _inputs(tokens, geo, device)
    if bucket:
        # only a captured step carries padding rows, and the eager wrapper does
        # not size its token mask for their borrowed tokens
        spans, slots, has, inputs = _padded(layout, inputs, tokens, len(layout) + 2, tokens)
    w = GDNPrefillWrapper(
        device=device, pad_slot_id=PAD_SLOT_ID, null_slot_id=SINK_SLOT,
        num_tokens=tokens if bucket else None, bs=len(spans) if bucket else None,
        accelerator_graph=bucket, stage_inputs=stage_inputs, state_in_place=in_place,
    )
    checkpoints = None
    if every:
        _, v_heads, dim = geo
        checkpoints = torch.empty(sum(max(s, 1) // every for s in spans) + 1, v_heads, dim, dim, device=device)
    w.plan(
        spans,
        torch.tensor(slots, dtype=torch.int32, device=device),
        torch.tensor(has, dtype=torch.bool, device=device),
        snapshots, every, checkpoints,
    )
    state = _pool(geo, device)
    out = w.run(state=state, **inputs)[:tokens]
    return out, state


def _mixed(in_place: bool, bucket: bool, geo=CPU_GEO, device=CPU):
    spans, slots, has = _cols(MIXED)
    tokens = sum(spans)
    inputs = _inputs(tokens, geo, device)
    bucket_tokens = bucket_rows = None
    if bucket:
        bucket_tokens, bucket_rows = tokens + 6, len(MIXED) + 2
        spans, slots, has, inputs = _padded(MIXED, inputs, tokens, bucket_rows, bucket_tokens)
    w = GDNMixedWrapper(
        device=device, pad_slot_id=PAD_SLOT_ID, null_slot_id=SINK_SLOT,
        num_tokens=bucket_tokens, bs=bucket_rows, accelerator_graph=bucket,
        max_prefill_rows=MAX_PROMPTS, state_in_place=in_place,
    )
    w.plan(spans, tuple(slots), tuple(has))
    state = _pool(geo, device)
    out = w.run(state=state, **inputs)[:tokens]
    return out, state


def _same(got, ref, written, geo=CPU_GEO):
    """Bit for bit: outputs, and every slot but the sink, which padding rows write as scratch.
    A slot outside ``written`` holds what it held."""
    out, state = got
    ref_out, ref_state = ref
    assert torch.equal(out, ref_out), f"outputs differ by {(out.float() - ref_out.float()).abs().max().item():.3g}"
    kept = [s for s in range(SLOTS) if s != SINK_SLOT]
    assert torch.equal(state[kept], ref_state[kept]), "the pool differs outside the sink"
    untouched = [s for s in kept if s not in written]
    assert torch.equal(state[untouched], _pool(geo, state.device)[untouched])


# --- the plumbing, on CPU --------------------------------------------------


@pytest.mark.parametrize("bucket", [False, True], ids=["eager", "bucket"])
def test_in_place_matches_the_copies(bucket):
    _same(_prefill(True, PREFILL, bucket), _prefill(False, PREFILL, bucket), _slots(PREFILL))


@pytest.mark.parametrize("bucket", [False, True], ids=["eager", "bucket"])
def test_a_mixed_steps_chunked_half_in_place_matches_the_copies(bucket):
    """Under capture the chunked half also runs the decode run as one sequence on the sink."""
    _same(_mixed(True, bucket), _mixed(False, bucket), _slots(MIXED))


def _spy(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    def kernel(**kw):
        calls.append(kw)
        return gdn_cpu.chunk_gated_delta_rule(**kw)

    monkeypatch.setattr(wrappers, "gdn_kernel", lambda name, device: kernel)
    return calls


def _eager(in_place: bool, layout) -> GDNPrefillWrapper:
    spans, slots, has = _cols(layout)
    w = GDNPrefillWrapper(device=CPU, pad_slot_id=PAD_SLOT_ID, null_slot_id=SINK_SLOT, state_in_place=in_place)
    w.plan(spans, torch.tensor(slots, dtype=torch.int32), torch.tensor(has, dtype=torch.bool))
    return w


def test_the_kernel_reads_and_writes_the_pool_at_the_plans_slots(monkeypatch):
    calls = _spy(monkeypatch)
    w = _eager(True, PREFILL)
    state = _pool(CPU_GEO, CPU)
    w.run(state=state, **_inputs(sum(_cols(PREFILL)[0]), CPU_GEO, CPU))
    (kw,) = calls
    assert kw["initial_state"] is state and kw["output_state"] is state
    assert kw["state_indices"] is w._plan_state.slots


def test_a_strided_head_block_keeps_the_copies(monkeypatch):
    """`GDNManager._ulysses` on more than one rank runs the chunked half on a head block of the
    pool, which is strided: there `run` gathers and scatters as before, with the same result."""
    calls = _spy(monkeypatch)
    inputs = _inputs(sum(_cols(PREFILL)[0]), (1, 2, 16), CPU)

    def run(in_place):
        state = _pool(CPU_GEO, CPU)
        block = state[:, 2:]
        assert not block.is_contiguous()
        return _eager(in_place, PREFILL).run(state=block, **inputs), state

    out, state = run(True)
    assert "state_indices" not in calls[-1]
    ref_out, ref_state = run(False)
    assert torch.equal(out, ref_out) and torch.equal(state, ref_state)


def test_in_place_without_a_sink_is_refused():
    """The kernel has no skip for NO_SLOT: a padding row on -1 would read and write slot -1."""
    with pytest.raises(ValueError, match="sink"):
        GDNPrefillWrapper(
            device=CPU, pad_slot_id=PAD_SLOT_ID, has_sink_state=False, null_slot_id=NO_SLOT,
            state_in_place=True,
        )


def test_the_pool_is_read_in_place_only_on_sm90_over_an_fp32_pool_with_a_sink():
    assert _prefill_reads_pool(9, torch.float32, has_sink=True)
    assert not _prefill_reads_pool(9, torch.float32, has_sink=False)
    assert not _prefill_reads_pool(9, torch.bfloat16, has_sink=True)
    for major in (8, 10, 12):
        assert not _prefill_reads_pool(major, torch.float32, has_sink=True)


@pytest.mark.parametrize("version,takes", [("0.6.16", False), ("0.7.1", False), ("0.7.2", True), ("0.8.0", True)])
def test_only_a_flashinfer_whose_sm90_kernel_takes_the_slots_reads_the_pool(monkeypatch, version, takes):
    monkeypatch.setitem(sys.modules, "flashinfer", types.SimpleNamespace(__version__=version))
    _flashinfer_takes_sm90_state_indices.cache_clear()
    try:
        assert _flashinfer_takes_sm90_state_indices() is takes
    finally:
        _flashinfer_takes_sm90_state_indices.cache_clear()


def _manager() -> GDNManager:
    pool = "gdn_state"
    k_heads, v_heads, dim = 2, 4, 128
    spec = RecurrentStateSpec(
        pool, {"llm"},
        RecurrentStateConfig(
            num_layers=1,
            blocks={
                "state": RecurrentBlockConfig(shape=(v_heads, dim, dim), dtype=torch.float32),
                "conv": RecurrentBlockConfig(shape=(2 * k_heads * dim + v_heads * dim, 3), dtype=torch.float32),
            },
            max_slots=SLOTS,
        ),
    )
    return GDNManager.build(
        LinearAttnSpec(
            "linear_attn", {"llm"},
            LinearAttnConfig(recurrent_state=pool, variant=LinearAttnVariant.GDN),
        ),
        EngineResourceInfo(device=CPU, dependencies={pool: spec}),
    )


def test_the_manager_hands_its_choice_to_both_chunked_wrappers():
    manager = _manager()
    # off a GPU the copies stay, as everywhere but SM90
    assert not manager._prefill_in_place
    assert not manager._get_wrapper("main", is_decode=False)._state_in_place
    manager._prefill_in_place = True
    assert manager._get_wrapper("other", is_decode=False)._state_in_place
    split = manager._get_wrapper("other", is_decode=False, split=CombinedSplit("mixed", ("prefill_text",)))
    assert split.halves[1]._state_in_place


# --- FlashInfer's kernels, on SM90 -----------------------------------------
#
# FlashInfer routes a call to its context-parallel (CP) kernels when the sequences times the
# value heads are under half the SMs (66 on an H200), and to the fully fused one otherwise. At
# Qwen3.5-9B's 32 value heads that is CP for one or two sequences, fused for three or more. In
# place, CP is the one with two CTAs on one slot: a sequence's first chunk loads its slot and
# its last chunk stores the final there, in the same launch.


def _on_sm90() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 9
        and _flashinfer_takes_sm90_state_indices()
    )


requires_sm90 = pytest.mark.skipif(
    not _on_sm90(), reason="FlashInfer's chunked kernel takes state_indices on SM90 from 0.7.2"
)


@requires_sm90
@pytest.mark.parametrize("stage_inputs", [False, True], ids=["unstaged", "staged"])
@pytest.mark.parametrize("bucket", [False, True], ids=["eager", "bucket"])
def test_in_place_matches_the_copies_on_sm90(bucket, stage_inputs):
    """Four sequences or more: the fused kernel."""
    device = torch.device("cuda")
    got = _prefill(True, PREFILL, bucket, GPU_GEO, device, stage_inputs)
    ref = _prefill(False, PREFILL, bucket, GPU_GEO, device, stage_inputs)
    _same(got, ref, _slots(PREFILL), GPU_GEO)


@requires_sm90
@pytest.mark.parametrize("bucket", [False, True], ids=["eager", "bucket"])
def test_a_mixed_step_in_place_matches_the_copies_on_sm90(bucket):
    """Three prompt rows eagerly, five sequences under capture: the fused kernel."""
    device = torch.device("cuda")
    got, ref = (_mixed(in_place, bucket, GPU_GEO, device) for in_place in (True, False))
    _same(got, ref, _slots(MIXED), GPU_GEO)


@requires_sm90
@pytest.mark.parametrize("layout", [
    # one long resumed prompt, then it beside a fresh one: CP, several chunks each
    [(3000, 3, True)],
    [(3000, 3, True), (2100, 7, False)],
    # three rows: the fused kernel
    [(3000, 3, True), (2100, 7, False), (700, 9, True)],
], ids=["cp-1", "cp-2", "fused-3"])
def test_long_prompts_in_place_match_the_copies_on_sm90(layout):
    """Repeated, since a race between a sequence's first and last CP chunk on its slot would
    show only now and then."""
    device = torch.device("cuda")
    ref = _prefill(False, layout, False, GPU_GEO, device)
    for _ in range(20):
        _same(_prefill(True, layout, False, GPU_GEO, device), ref, _slots(layout), GPU_GEO)


@requires_sm90
@pytest.mark.parametrize("layout", [
    [(192, 3, True), (130, 7, False)],
    [(192, 3, True), (130, 7, False), (64, 9, True)],
], ids=["cp-2", "fused-3"])
def test_snapshots_in_place_match_the_copies_on_sm90(layout):
    """The kernel's checkpoints are still copied into snapshot slots after it: a resumed and a
    fresh prompt each keep one, at a 64-token boundary."""
    device = torch.device("cuda")
    snapshots = [(0, 128, 10), (1, 64, 11)]
    got, ref = (
        _prefill(in_place, layout, False, GPU_GEO, device, snapshots=snapshots, every=64)
        for in_place in (True, False)
    )
    _same(got, ref, _slots(layout) | {10, 11}, GPU_GEO)


@requires_sm90
def test_a_replay_in_place_matches_the_copies_on_sm90():
    """Captured over one mixed layout and replayed over another: the zeroing and the slots the
    kernel reads are planned, so nothing of the captured step survives."""
    device = torch.device("cuda")
    bucket_tokens, bucket_rows = 32, 10
    other = [(1, 6, True), (5, 9, False), (2, 4, True)]

    def replayed(in_place):
        w = GDNMixedWrapper(
            device=device, pad_slot_id=PAD_SLOT_ID, null_slot_id=SINK_SLOT,
            num_tokens=bucket_tokens, bs=bucket_rows, accelerator_graph=True,
            max_prefill_rows=MAX_PROMPTS, state_in_place=in_place,
        )
        state = _pool(GPU_GEO, device)
        bufs = {
            name: t.new_zeros(bucket_tokens, *t.shape[1:]) if t.dim() > 1 else t.clone()
            for name, t in _inputs(bucket_tokens, GPU_GEO, device).items()
        }

        def stage(layout):
            tokens = sum(s for s, _, _ in layout)
            spans, slots, has, _ = _padded(layout, {}, tokens, bucket_rows, bucket_tokens)
            w.plan(spans, tuple(slots), tuple(has))
            for name, t in _inputs(tokens, GPU_GEO, device).items():
                if t.dim() > 1:
                    bufs[name].zero_()
                    bufs[name][:tokens].copy_(t)
            return tokens

        stage(MIXED)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            w.run(state=state, **bufs)  # warm the kernels outside the graph
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = w.run(state=state, **bufs)
        state.copy_(_pool(GPU_GEO, device))
        tokens = stage(other)
        graph.replay()
        torch.cuda.synchronize()
        return out[:tokens].clone(), state

    _same(replayed(True), replayed(False), _slots(other), GPU_GEO)
