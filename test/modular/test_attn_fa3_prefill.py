"""``prefill_backend: fa3`` runs a captured step's prompt rows on FA3's paged kernel (`FA3RaggedWrapper`): a split
walk's prompt half (`FlashInferMixedWrapper`) and a multi-token bucket's rows. Decode buckets and eager steps stay on
FlashInfer, and so does every capture where FA3 can't run the pool, with a warning naming why.

The selection is checked on the CPU, with FlashInfer's wrapper classes replaced by recorders and FA3's availability
stubbed. The kernel against FlashInfer's on the same inputs, eager and replayed, needs CUDA, ``flashinfer`` and
``flash_attn_3`` (Hopper).
"""
from __future__ import annotations

import logging
import types

import numpy as np
import pytest
import torch

from mstar.engine.resources.attn import fa3_paged as fa3_mod
from mstar.engine.resources.attn import flashinfer as fi_mod
from mstar.engine.resources.attn import wrappers as wrappers_mod
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttentionStep, AttnBackend
from mstar.engine.resources.attn.fa3_paged import FA3RaggedWrapper, fa3_paged_unavailable_reason
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
from mstar.engine.resources.kv.plan import SINK_PAGE, HostLens, SequenceView, build_paged_indptrs
from mstar.engine.resources.spec import CombinedSplit
from mstar.engine.resources.step import BucketKey, SlotLease

CPU = torch.device("cpu")
PAGE, MAX_SEQ = 16, 256


class _Recorder:
    built: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        type(self).built.append(self)


class _Decode(_Recorder):
    built = []


class _Prefill(_Recorder):
    built = []


class _Mixed(_Recorder):
    built = []


@pytest.fixture
def recorders(monkeypatch):
    monkeypatch.setenv("MSTAR_WORKSPACE_BUFFER_MB", "1")
    for cls in (_Decode, _Prefill, _Mixed):
        cls.built = []
    monkeypatch.setattr(fi_mod, "FlashInferDecodeWrapper", _Decode)
    monkeypatch.setattr(fi_mod, "FlashInferPrefillWrapper", _Prefill)
    monkeypatch.setattr(fi_mod, "FlashInferMixedWrapper", _Mixed)


@pytest.fixture
def fa3_here(monkeypatch):
    """FA3 runs the pool here, as on an H200 with the wheel."""
    monkeypatch.setattr(fa3_mod, "fa3_paged_unavailable_reason", lambda *args: None)


def _kv():
    return PagedKVConfig(num_layers=1, num_kv_heads=2, num_qo_heads=4, head_dim=64, max_seq_len=MAX_SEQ,
                         max_num_pages=8, page_size=PAGE)


def _built(**config) -> fi_mod.FlashInferManager:
    """The manager a spec builds, ``config`` its AttentionConfig's fields."""
    spec = AttentionSpec(resource_key="attn", nodes={"n"}, config=AttentionConfig(
        kv_cache="kv", backend=AttnBackend.FLASHINFER, **config,
    ))
    kv = KVSpec(resource_key="kv", nodes={"n"}, config=_kv())
    return AttentionManager.build(spec, EngineResourceInfo(device=CPU, dependencies={"kv": kv}))


def _lease(walk: str, bs: int, tokens: int) -> SlotLease:
    return SlotLease(slot=0, bucket=BucketKey(graph_walk=walk, bs=bs, num_tokens=tokens))


def test_prefill_backend_defaults_to_flashinfer_and_a_deployment_picks_fa3():
    spec = AttentionSpec(resource_key="attn", nodes={"n"}, config=AttentionConfig(kv_cache="kv"))
    assert spec.config.prefill_backend == "flashinfer"
    spec.apply_yaml_overrides(prefill_backend="fa3")
    assert spec.config.prefill_backend == "fa3"
    with pytest.raises(ValueError, match="prefill_backend must be flashinfer or fa3, not 'fa2'"):
        spec.apply_yaml_overrides(prefill_backend="fa2")
    with pytest.raises(ValueError, match="prefill_backend must be flashinfer or fa3"):
        AttentionConfig(kv_cache="kv", prefill_backend="flash_attn_3")


def test_a_captured_prompt_bucket_runs_on_fa3_and_decode_and_eager_steps_stay(recorders, fa3_here):
    manager = _built(prefill_backend="fa3")
    wrapper = manager._cg_wrapper(_lease("prefill", 2, 64), "text", num_rows=2, force_prefill=False)
    assert isinstance(wrapper, FA3RaggedWrapper)
    # the bucket's rows and tokens, and a row of the pool's whole context
    assert (wrapper.batch_size, wrapper._max_q, wrapper.width, wrapper.causal) == (2, 64, MAX_SEQ // PAGE, True)
    assert wrapper.accelerator_graph and _Prefill.built == []
    # a decode bucket, and a packed one of one-token rows, keep FlashInfer's decode and prefill wrappers
    manager._cg_wrapper(_lease("decode", 4, 4), "text", num_rows=4, force_prefill=False)
    assert len(_Decode.built) == 1
    # an eager prompt step stays on FlashInfer
    manager._eager_wrapper("text", is_decode=False, slot=0)
    assert len(_Prefill.built) == 1


def test_a_bucket_wider_than_the_context_gets_a_table_for_its_padding_row(recorders, fa3_here):
    """A padding row reads a sink page per page of its span, which can be the bucket's tokens."""
    manager = _built(prefill_backend="fa3")
    wrapper = manager._cg_wrapper(_lease("prefill", 2, 4 * MAX_SEQ), "text", num_rows=2, force_prefill=False)
    assert wrapper.width == 4 * MAX_SEQ // PAGE


def test_a_split_walks_prompt_half_runs_on_fa3(recorders, fa3_here):
    split = CombinedSplit(combined_walk="mixed", prefill_walks=("prefill",))
    for backend in ("fa3", "flashinfer"):
        manager = fi_mod.FlashInferManager(
            kv_cache="kv", device=CPU, dtype=torch.bfloat16, kv_config=_kv(), splits=(split,),
            max_rows_in_combined={"prefill": 3}, prefill_backend=backend,
        )
        manager._cg_wrapper(_lease("mixed", 8, 128), "text", num_rows=8, force_prefill=False)
    on_fa3, on_flashinfer = (m.kwargs for m in _Mixed.built)
    half = on_fa3["prefill_wrapper"]
    assert isinstance(half, FA3RaggedWrapper)
    # the prompt rows, the decode run's sink row and the padding's
    assert (on_fa3["max_prefill_rows"], half.batch_size, half._max_q) == (3, 5, 128)
    assert on_flashinfer["prefill_wrapper"] is None


def test_the_mixed_wrapper_plans_its_prompt_half_on_the_wrapper_it_is_given(monkeypatch):
    monkeypatch.setattr(wrappers_mod, "FlashInferPrefillWrapper", _Prefill)
    _Prefill.built = []
    kw = dict(
        decode_workspace=None, prefill_workspace=None, num_qo_heads=4, num_kv_heads=2, head_dim=64,
        page_size=PAGE, batch_size=8, max_total_tokens=128, max_num_pages=64, max_prefill_rows=3, device=CPU,
    )
    half = FA3RaggedWrapper(page_size=PAGE, table_width=16, device=CPU, batch_size=5, max_tokens=128,
                            accelerator_graph=True)
    mixed = wrappers_mod.FlashInferMixedWrapper(prefill_wrapper=half, **kw)
    assert mixed._prefill is half
    # the tensor-core decode half alone is FlashInfer's
    assert len(_Prefill.built) == 1 and _Prefill.built[0].kwargs["batch_size"] == 8
    wrong = FA3RaggedWrapper(page_size=PAGE, table_width=16, device=CPU, batch_size=4, max_tokens=128,
                             accelerator_graph=True)
    with pytest.raises(ValueError, match="captured for 4 rows; this split plans 5"):
        wrappers_mod.FlashInferMixedWrapper(prefill_wrapper=wrong, **kw)


def test_the_default_never_asks_about_fa3(recorders, monkeypatch):
    def asked(*args):
        raise AssertionError("prefill_backend flashinfer asked whether FA3 runs")

    monkeypatch.setattr(fa3_mod, "fa3_paged_unavailable_reason", asked)
    wrapper = _built()._cg_wrapper(_lease("prefill", 2, 64), "text", num_rows=2, force_prefill=False)
    assert isinstance(wrapper, _Prefill)


@pytest.mark.parametrize("config,reason", [
    (dict(sm_scale=0.1), "not sm_scale 0.1"),
    # as wide as the context, so the pool keeps every token and needs no window groups
    (dict(sliding_window=MAX_SEQ), "plans no sliding window"),
])
def test_a_pool_fa3_cant_run_keeps_flashinfer_with_a_warning(recorders, fa3_here, caplog, config, reason):
    manager = _built(prefill_backend="fa3", **config)
    with caplog.at_level(logging.WARNING, logger=fi_mod.__name__):
        for slot in (0, 1):
            lease = SlotLease(slot=slot, bucket=BucketKey(graph_walk="prefill", bs=2, num_tokens=64))
            assert isinstance(manager._cg_wrapper(lease, "text", num_rows=2, force_prefill=False), _Prefill)
    warnings = [r.getMessage() for r in caplog.records if "prefill_backend fa3" in r.getMessage()]
    assert len(warnings) == 1 and reason in warnings[0] and "runs on FlashInfer" in warnings[0]


def test_on_the_cpu_fa3_never_runs_and_the_capture_keeps_flashinfer(recorders, caplog):
    """The real check: no wheel here, or a device that isn't an SM90 GPU."""
    manager = _built(prefill_backend="fa3")
    with caplog.at_level(logging.WARNING, logger=fi_mod.__name__):
        wrapper = manager._cg_wrapper(_lease("prefill", 2, 64), "text", num_rows=2, force_prefill=False)
    assert isinstance(wrapper, _Prefill)
    assert any("prefill_backend fa3" in r.getMessage() for r in caplog.records)


def test_a_context_parallel_pool_keeps_flashinfer(fa3_here):
    manager = _built(prefill_backend="fa3")
    manager._placement = object()
    assert manager.fa3_prefill_refusal() == "the pool is context-parallel"
    assert manager._prefill_fa3 is False


def _ctx(lease, views):
    indptrs = build_paged_indptrs(views, page_size=PAGE)
    return types.SimpleNamespace(
        resident=None, slot_lease=lease, is_preplan=False, force_prefill=False, slot=lease.slot,
        plan_phase="res.plan", plan_results={"kv": {"main": types.SimpleNamespace(cpu_indptrs=indptrs)}},
    )


def test_a_planned_capture_stages_its_rows_and_refuses_the_other_mask(recorders, fa3_here):
    """Planned through the manager: the bucket's FA3 buffers hold each row's pages, keys and query offsets, the last
    tokens are read off them, and FA3, which fixes the mask at capture, refuses a non-causal plan of the bucket."""
    manager = _built(prefill_backend="fa3")
    lease = _lease("prefill", 2, 16)
    views = [SequenceView("a", "main", [1], 5, 5), SequenceView("b", "main", [2, 3], 23, 3)]
    manager.plan(AttentionStep(causal=True), _ctx(lease, views))
    wrapper = manager._current_plan_states["main"]
    assert isinstance(wrapper, FA3RaggedWrapper)
    assert wrapper._seqused.tolist() == [5, 23]
    assert wrapper._qo_indptr_buf.tolist() == [0, 5, 8]
    assert wrapper._page_table[0, :1].tolist() == [1] and wrapper._page_table[1, :2].tolist() == [2, 3]
    hidden = torch.arange(16.0).view(16, 1)
    assert manager.select_last_hidden(hidden, "main").view(-1).tolist() == [4.0, 7.0]
    with pytest.raises(RuntimeError, match="FA3 fixes at capture"):
        manager.plan(AttentionStep(causal=False), _ctx(lease, views))


def test_a_flashinfer_capture_takes_either_mask(recorders):
    manager = _built()
    lease = _lease("prefill", 2, 64)
    first = manager._cg_wrapper(lease, "text", num_rows=2, force_prefill=False, causal=True)
    assert manager._cg_wrapper(lease, "text", num_rows=2, force_prefill=False, causal=False) is first


# --- on an SM90 GPU with flashinfer and flash_attn_3 -------------------------------------------------------------

requires_fa3 = pytest.mark.skipif(
    not torch.cuda.is_available() or fa3_paged_unavailable_reason(None, 256, torch.bfloat16) is not None,
    reason="needs an SM90 GPU, flashinfer and flash_attn_3",
)

# Qwen3.5-9B's full-attention geometry
QO_HEADS, KV_HEADS, DIM = 16, 4, 256
BUCKET_BS, BUCKET_TOKENS, MAX_PROMPTS = 8, 512, 3
# (new tokens, cached tokens before this step), decode rows first as the engine orders a mixed step
MIXED_ROWS = [(1, 3000), (1, 77), (1, 1500), (300, 2000), (150, 0), (1, 0)]


def _layout(rows, page):
    """Host index arrays as the KV plan builds them, zero-span padding rows on the sink page."""
    qo, kvp, idx, last, lens = [0], [0], [], [], []
    nxt = SINK_PAGE + 1
    for new, cached in rows:
        n = new + cached
        pages = -(-n // page)
        qo.append(qo[-1] + new)
        idx.extend(range(nxt, nxt + pages))
        nxt += pages
        kvp.append(kvp[-1] + pages)
        last.append((n - 1) % page + 1)
        lens.append(n)
    for _ in range(BUCKET_BS - len(rows)):
        qo.append(qo[-1])
        idx.append(SINK_PAGE)
        kvp.append(kvp[-1] + 1)
        last.append(1)
        lens.append(1)
    i32 = lambda v: torch.tensor(v, dtype=torch.int32)  # noqa: E731
    return dict(
        qo_indptr=i32(qo), paged_kv_indptr=i32(kvp), paged_kv_indices=i32(idx),
        paged_kv_last_page_len=i32(last), kv_lens=i32(lens).as_subclass(HostLens),
    ), nxt


def _mixed(device, page, prefill_wrapper=None):
    from mstar.engine.resources.attn.workspace import Workspace

    ws = lambda: Workspace(torch.empty(128 << 20, dtype=torch.uint8, device=device), "test")  # noqa: E731
    return wrappers_mod.FlashInferMixedWrapper(
        decode_workspace=ws(), prefill_workspace=ws(), num_qo_heads=QO_HEADS, num_kv_heads=KV_HEADS, head_dim=DIM,
        page_size=page, batch_size=BUCKET_BS, max_total_tokens=BUCKET_TOKENS, max_num_pages=4096,
        max_prefill_rows=MAX_PROMPTS, device=device, prefill_wrapper=prefill_wrapper,
    )


@requires_fa3
@pytest.mark.parametrize("page", [16, 128])
def test_the_fa3_prompt_half_attends_as_flashinfers_does_eager_and_replayed(page):
    device = torch.device("cuda")
    plan, num_pages = _layout(MIXED_ROWS, page)
    torch.manual_seed(0)
    cache = torch.randn(num_pages, 2, page, KV_HEADS, DIM, device=device, dtype=torch.bfloat16)
    q = torch.randn(BUCKET_TOKENS, QO_HEADS, DIM, device=device, dtype=torch.bfloat16)
    real = int(plan["qo_indptr"][len(MIXED_ROWS)])

    ref = _mixed(device, page)
    ref.plan(causal=True, dtype=torch.bfloat16, **plan)
    ref_out = ref.run(q, cache)
    width = -(-max(4096, BUCKET_TOKENS) // page)
    half = FA3RaggedWrapper(page_size=page, table_width=width, device=device, batch_size=MAX_PROMPTS + 2,
                            max_tokens=BUCKET_TOKENS, accelerator_graph=True)
    fa3 = _mixed(device, page, half)
    fa3.plan(causal=True, dtype=torch.bfloat16, **plan)
    out = fa3.run(q, cache).clone()

    assert torch.isfinite(out[:real]).all()
    err = (out[:real].float() - ref_out[:real].float()).abs().max().item()
    print(f"page {page}: max err {err:.3g} over {real} tokens, max |ref| {ref_out.float().abs().max().item():.3g}")
    assert err <= 2e-2, f"max err {err}"
    # the decode rows are FlashInfer's in both
    assert torch.equal(out[:3], ref_out[:3])

    # a replay of the captured run is the eager run, bit for bit
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fa3.run(q, cache)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = fa3.run(q, cache)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(replayed[:real], out[:real])


@requires_fa3
def test_a_captured_prompt_bucket_on_fa3_attends_as_flashinfers_prefill_does():
    """A multi-token bucket's rows, two prompts and a padding row on the sink page, against FlashInfer's captured
    prefill wrapper over the same arrays."""
    from mstar.engine.resources.attn.workspace import Workspace

    device, page, bs, tokens = torch.device("cuda"), 128, 3, 1024
    views = [
        SequenceView("a", "main", list(range(1, 17)), 2048, 600),
        SequenceView("b", "main", [17, 18, 19], 300, 300),
        SequenceView("pad", "main", [SINK_PAGE], tokens - 900, tokens - 900),
    ]
    arrays = build_paged_indptrs(views, page_size=page).to_kwargs_dict()
    torch.manual_seed(0)
    cache = torch.randn(20, 2, page, KV_HEADS, DIM, device=device, dtype=torch.bfloat16)
    q = torch.randn(tokens, QO_HEADS, DIM, device=device, dtype=torch.bfloat16)
    ref = wrappers_mod.FlashInferPrefillWrapper(
        workspace_buffer=Workspace(torch.empty(128 << 20, dtype=torch.uint8, device=device), "test"),
        num_qo_heads=QO_HEADS, num_kv_heads=KV_HEADS, head_dim=DIM, page_size=page, batch_size=bs,
        max_total_tokens=tokens, max_num_pages=64, device=device, accelerator_graph=True,
    )
    ref.plan(causal=True, dtype=torch.bfloat16, **arrays)
    fa3 = FA3RaggedWrapper(page_size=page, table_width=16, device=device, batch_size=bs, max_tokens=tokens,
                           accelerator_graph=True)
    fa3.plan(causal=True, dtype=torch.bfloat16, **arrays)
    ref_out, out = ref.run(q, cache), fa3.run(q, cache)
    err = (out[:900].float() - ref_out[:900].float()).abs().max().item()
    print(f"max err {err:.3g} over 900 tokens")
    assert err <= 2e-2, f"max err {err}"
    assert np.isfinite(out.float().cpu().numpy()).all()
