"""GLM-5.2's MoE block (``SharedExpertMoE``): the noaux_tc router, fp8-resident expert
dispatch vs the bf16 reference, kernel resolution, the post-load dtype guards, and the
end-to-end reduced fp8 load (whose checkpoint fabricator the indexer / read-plan / quant
suites reuse).
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _cpu_rmsnorm(x, weight, eps=1e-6):
    x32 = x.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * weight.float()).to(x.dtype)


def _cpu_flashinfer() -> types.ModuleType:
    fi = types.ModuleType("flashinfer")
    fi.norm = types.SimpleNamespace(rmsnorm=_cpu_rmsnorm)
    return fi


@pytest.fixture(autouse=True)
def _cpu_flashinfer_without_the_real_one(monkeypatch):
    # per test: left in sys.modules, the stand-in breaks find_spec("flashinfer")
    # in every file collected after this one
    if importlib.util.find_spec("flashinfer") is None:
        monkeypatch.setitem(sys.modules, "flashinfer", _cpu_flashinfer())


from mstar.model.components.moe import NoAuxTcRouter  # noqa: E402
from mstar.model.components.quantization import (  # noqa: E402
    QUANT_KERNELS,
    Fp8BlockExperts,
    Fp8BlockQuant,
)
from mstar.model.components.shared_expert_moe import SharedExpertMoE  # noqa: E402
from mstar.model.glm52._testing import fake_quantize_fp8_block  # noqa: E402
from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM  # noqa: E402
from mstar.model.glm52.components.indexer import is_full_indexer_layer  # noqa: E402
from mstar.model.glm52.config import Glm52ModelConfig  # noqa: E402
from mstar.model.glm52.weight_loader import (  # noqa: E402
    load_glm52_hf_weights,
    restore_fp32_params,
)
from mstar.utils.fp8_block import dequantize_fp8_block_weight  # noqa: E402

BLOCK = (16, 16)


def _noaux_tc_reference(h, weight, bias, top_k, scale):
    """DeepSeek-V3 noaux_tc with n_group=1, transcribed from the rule:
    sigmoid affinities; the correction bias moves only the SELECTION; the
    combine weights are the raw affinities of the picked experts,
    normalized over the picked set, times the routed scale.
    """
    scores = torch.sigmoid(h.float() @ weight.float().T)
    ids = torch.topk(scores + bias.float(), k=top_k, dim=-1, sorted=False).indices
    weights = torch.gather(scores, 1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True) * scale
    return weights, ids


def test_gate_is_noaux_tc_groupless():
    """GLM's router is the DeepSeek-V3 noaux_tc sigmoid gate with n_group=1 —
    pin it bitwise against an independent transcription of the rule."""
    torch.manual_seed(0)
    glm = NoAuxTcRouter(
        hidden_size=64, n_routed_experts=8, num_experts_per_tok=3,
        routed_scaling_factor=2.5, norm_topk_prob=True,
    )
    glm.weight.data.normal_(0, 0.1)
    glm.e_score_correction_bias.data.normal_(0, 0.5)

    h = torch.randn(17, 64)
    w_ref, ids_ref = _noaux_tc_reference(
        h, glm.weight, glm.e_score_correction_bias, top_k=3, scale=2.5)
    w_glm, ids_glm, state = glm(h)
    assert state is None  # stateless, per the router contract
    assert torch.equal(ids_ref, ids_glm)
    assert torch.equal(w_ref, w_glm)
    # the bias steers selection only: a huge bias on one expert makes it a
    # pick for every token while the combine weights stay raw affinities
    glm.e_score_correction_bias.data[5] = 100.0
    w_glm, ids_glm, _ = glm(h)
    assert (ids_glm == 5).any(dim=-1).all()
    assert (w_glm <= 2.5).all()
    # the fp32 router cache (finalize_weights) is the same values, bitwise
    glm.e_score_correction_bias.data[5] = 0.0
    w_before, ids_before, _ = glm(h)
    glm.finalize_weights()
    w_after, ids_after, _ = glm(h)
    assert torch.equal(w_before, w_after) and torch.equal(ids_before, ids_after)


def _fill_expert_weights(block_fp8, block_bf16, cfg):
    """Give both blocks the same experts: fp8 bytes vs their fp32 dequant."""
    shard = cfg.moe_intermediate_size
    refs = []
    for e in range(cfg.n_routed_experts):
        g = torch.randn(shard, cfg.hidden_size) * 0.1
        u = torch.randn(shard, cfg.hidden_size) * 0.1
        d = torch.randn(cfg.hidden_size, shard) * 0.1
        g8, gs, _ = fake_quantize_fp8_block(g, BLOCK)
        u8, us, _ = fake_quantize_fp8_block(u, BLOCK)
        d8, ds, _ = fake_quantize_fp8_block(d, BLOCK)
        srow = shard // BLOCK[0]

        exp8 = block_fp8.experts
        exp8.gate_up_proj_fp8.data[e, :shard] = g8.view(torch.uint8)
        exp8.gate_up_proj_fp8.data[e, shard:] = u8.view(torch.uint8)
        exp8.gate_up_proj_scale_inv.data[e, :srow] = gs
        exp8.gate_up_proj_scale_inv.data[e, srow:] = us
        exp8.down_proj_fp8.data[e] = d8.view(torch.uint8)
        exp8.down_proj_scale_inv.data[e] = ds

        # fp32 dequant (not bf16) so both paths multiply identical values.
        g32 = dequantize_fp8_block_weight(g8, gs, BLOCK, out_dtype=torch.float32)
        u32 = dequantize_fp8_block_weight(u8, us, BLOCK, out_dtype=torch.float32)
        d32 = dequantize_fp8_block_weight(d8, ds, BLOCK, out_dtype=torch.float32)
        exp16 = block_bf16.experts
        exp16.gate_up_proj.data[e, :shard] = g32
        exp16.gate_up_proj.data[e, shard:] = u32
        exp16.down_proj.data[e] = d32
        refs.append((g8, u8, d8))
    return refs


def test_fp8_resident_dispatch_matches_bf16_dispatch():
    torch.manual_seed(1)
    cfg8 = Glm52ModelConfig.reduced_fp8(block=BLOCK)
    cfg = Glm52ModelConfig.reduced()
    block8 = SharedExpertMoE(cfg8)
    block = SharedExpertMoE(cfg)
    assert isinstance(block8.experts, Fp8BlockExperts)
    assert not isinstance(block.experts, Fp8BlockExperts)

    block8.gate.weight.data.normal_(0, 0.1)
    block8.gate.e_score_correction_bias.data.normal_(0, 0.5)
    block.gate.load_state_dict(block8.gate.state_dict())
    for p, p8 in zip(
        block.shared_expert.parameters(),
        block8.shared_expert.parameters(), strict=True,
    ):
        p8.data.normal_(0, 0.05)
        p.data.copy_(p8.data)

    _fill_expert_weights(block8, block, cfg)

    x = torch.randn(6, cfg.hidden_size) * 0.1
    out8 = block8(x)
    out = block(x)
    # Same loop shape, same fp32 operands -> bitwise equal.
    assert torch.equal(out8, out)


def _hf_routed(flat, topk_weights, topk_ids, expert_weights):
    """HF GlmMoeDsaExperts: the bf16 expert output times the fp32 router
    weight, downcast once into the index_add_."""
    final = torch.zeros_like(flat)
    mask = F.one_hot(topk_ids, num_classes=len(expert_weights)).permute(2, 1, 0)
    for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()[:, 0].tolist():
        top_k_pos, token_idx = torch.where(mask[e])
        gate_up, down = expert_weights[e]
        gate, up = F.linear(flat[token_idx], gate_up).chunk(2, dim=-1)
        out = F.linear(F.silu(gate) * up, down)
        out = out * topk_weights[token_idx, top_k_pos, None]
        final.index_add_(0, token_idx, out.to(final.dtype))
    return final


def _bf16_block(cfg):
    block = SharedExpertMoE(cfg)
    for prm in block.parameters():
        if prm.dtype.is_floating_point:
            prm.data.normal_(0, 0.05)
    block.gate.e_score_correction_bias.data.normal_(0, 0.5)
    block.to(torch.bfloat16)
    restore_fp32_params(block)
    return block


def test_bf16_combine_keeps_fp32_routing_weights():
    """Both reference dispatches match HF's fp32-weight combine bitwise
    with bf16 activations, where a bf16 cast of the weights shows."""
    torch.manual_seed(5)
    cfg = Glm52ModelConfig.reduced()
    cfg8 = Glm52ModelConfig.reduced_fp8(block=BLOCK)
    block, block8 = _bf16_block(cfg), _bf16_block(cfg8)
    _fill_expert_weights(block8, block, cfg)

    x = (torch.randn(32, cfg.hidden_size) * 0.5).bfloat16()
    for blk in (block, block8):
        if isinstance(blk.experts, Fp8BlockExperts):
            deq = [
                tuple(
                    dequantize_fp8_block_weight(
                        getattr(blk.experts, f"{n}_fp8")[e],
                        getattr(blk.experts, f"{n}_scale_inv")[e],
                        block_size=BLOCK, out_dtype=torch.bfloat16,
                    )
                    for n in ("gate_up_proj", "down_proj")
                )
                for e in range(cfg.n_routed_experts)
            ]
        else:
            deq = list(zip(blk.experts.gate_up_proj, blk.experts.down_proj, strict=True))
        topk_weights, topk_ids, _ = blk.gate(x)
        assert topk_weights.dtype == torch.float32
        ref = _hf_routed(x, topk_weights, topk_ids, deq) + blk.shared_expert(x)
        assert torch.equal(blk(x), ref), type(blk.experts).__name__


def test_fused_dispatch_receives_fp32_routing_weights(monkeypatch):
    """The fused kernel loads topk_weights at their own dtype, so the block
    must hand it the fp32 router output."""
    seen = {}

    def fake_fused_experts_fp8(flat, *args, **kwargs):
        seen["dtype"] = args[4].dtype
        return torch.zeros_like(flat)

    fake = types.ModuleType("mstar.utils.fused_moe")
    fake.fused_experts_fp8 = fake_fused_experts_fp8
    monkeypatch.setitem(sys.modules, "mstar.utils.fused_moe", fake)

    torch.manual_seed(6)
    block8 = _bf16_block(Glm52ModelConfig.reduced_fp8(block=BLOCK))
    block8.experts.fused = True
    block8(torch.randn(4, block8.hidden_size).bfloat16())
    assert seen["dtype"] == torch.float32


def test_submodule_refuses_post_load_dtype_cast():
    """The engine manager casts the submodule to bf16 AFTER get_submodule;
    that must not re-narrow the fp32 scales/bias restore_fp32_params set."""
    from mstar.model.glm52.submodules import Glm52LLMSubmodule

    torch.manual_seed(3)
    cfg = Glm52ModelConfig.reduced_fp8(block=BLOCK)
    lm = Glm52ForCausalLM(cfg)
    moe = lm.model.layers[1].mlp
    moe.experts.gate_up_proj_scale_inv.data.normal_(0, 0.1)
    moe.gate.e_score_correction_bias.data.normal_(0, 0.5)
    scales_before = moe.experts.gate_up_proj_scale_inv.data.clone()
    bias_before = moe.gate.e_score_correction_bias.data.clone()

    sub = Glm52LLMSubmodule(language_model=lm, config=cfg)
    sub.to(device="cpu", dtype=torch.bfloat16)  # the engine manager's call

    assert moe.experts.gate_up_proj_scale_inv.dtype == torch.float32
    assert moe.gate.e_score_correction_bias.dtype == torch.float32
    assert torch.equal(moe.experts.gate_up_proj_scale_inv.data, scales_before)
    assert torch.equal(moe.gate.e_score_correction_bias.data, bias_before)
    assert moe.experts.gate_up_proj_fp8.dtype == torch.uint8


def test_quant_kernels_resolution():
    """auto probes the device, an explicit fused ask must not silently downgrade, and
    reference keeps the host loop."""
    cfg = Glm52ModelConfig.reduced_fp8(block=BLOCK)
    assert cfg.quant_kernels == "auto"
    for kernels in ("auto", "reference"):
        cfg.quant_kernels = kernels
        block = SharedExpertMoE(cfg)
        block.process_weights_after_loading("cpu")
        assert block.fused is False  # no CUDA here -> reference
    cfg.quant_kernels = "fused"
    block = SharedExpertMoE(cfg)
    with pytest.raises(RuntimeError, match="quant_kernels='fused'"):
        block.process_weights_after_loading("cpu")


def test_default_dispatch_is_fused_where_the_kernel_runs(monkeypatch):
    """The default resolves to the fused kernel when it can run and to the
    reference loop when it cannot; blocks that never see the load hook stay
    on reference."""
    cfg = Glm52ModelConfig.reduced_fp8(block=BLOCK)
    assert SharedExpertMoE(cfg).fused is False
    for available in (True, False):
        monkeypatch.setattr(Fp8BlockQuant, "fused_available", lambda self, dev, a=available: a)
        block = SharedExpertMoE(cfg)
        block.process_weights_after_loading("cpu")
        assert block.fused is available


def test_fused_kernel_needs_its_scale_block():
    cpu, cuda = torch.device("cpu"), torch.device("cuda")
    assert not Glm52ModelConfig.reduced_fp8(block=(128, 128)).quant.fused_available(cpu)
    assert not Glm52ModelConfig.reduced_fp8(block=BLOCK).quant.fused_available(cuda)
    reference = Glm52ModelConfig.reduced_fp8(block=(128, 128))
    reference.quant_kernels = "reference"
    assert not reference.quant.fused_available(cuda)


def test_unknown_quant_kernels_are_refused():
    """A typo ("Triton", "triton") used to resolve to the reference loop and serve eager
    behind a warning; the method refuses it up front."""
    assert QUANT_KERNELS == ("auto", "fused", "reference")
    for bad in ("Triton", "triton", ""):
        cfg = Glm52ModelConfig.reduced_fp8(block=BLOCK)
        cfg.quant_kernels = bad
        with pytest.raises(ValueError, match="quant_kernels"):
            SharedExpertMoE(cfg)
        with pytest.raises(ValueError, match="quant_kernels"):
            Glm52ModelConfig(quant_kernels=bad)
    for ok in QUANT_KERNELS:
        cfg = Glm52ModelConfig.reduced_fp8(block=BLOCK)
        cfg.quant_kernels = ok
        SharedExpertMoE(cfg)


def test_cuda_graphs_return_with_fused_dispatch():
    """Graph configs are gated on the RESOLVED dispatch: reference -> none
    (uncapturable host loops), fused -> registered."""
    import torch as _torch

    from mstar.model.glm52.submodules import Glm52LLMSubmodule

    cfg = Glm52ModelConfig.reduced_fp8(block=BLOCK)
    sub = object.__new__(Glm52LLMSubmodule)
    sub.config = cfg

    lm = _torch.nn.Module()
    lm.blk = SharedExpertMoE(cfg)
    # object.__new__ skipped nn.Module.__init__, so bypass its __setattr__.
    object.__setattr__(sub, "language_model", lm)

    lm.blk.experts.fused = False
    assert sub.get_accelerator_graph_configs(_torch.device("cpu")) == []
    lm.blk.experts.fused = True
    assert len(sub.get_accelerator_graph_configs(_torch.device("cpu"))) == 2


def test_autocast_then_restore_keeps_fp8_layout():
    cfg8 = Glm52ModelConfig.reduced_fp8(block=BLOCK)
    block8 = SharedExpertMoE(cfg8)
    bytes_before = block8.experts.gate_up_proj_fp8.data.clone()

    block8.to(torch.bfloat16)  # the submodule-load autocast
    assert block8.experts.gate_up_proj_fp8.dtype == torch.uint8  # ints immune
    assert block8.experts.gate_up_proj_scale_inv.dtype == torch.bfloat16
    assert block8.gate.e_score_correction_bias.dtype == torch.bfloat16

    restore_fp32_params(block8)
    assert block8.experts.gate_up_proj_scale_inv.dtype == torch.float32
    assert block8.experts.down_proj_scale_inv.dtype == torch.float32
    assert block8.gate.e_score_correction_bias.dtype == torch.float32
    assert torch.equal(block8.experts.gate_up_proj_fp8.data, bytes_before)


def _fp8_entry(state, refs, base, shape):
    w = torch.randn(*shape) * 0.1
    w8, s, deq = fake_quantize_fp8_block(w, BLOCK)
    state.append((f"{base}.weight", w8))
    state.append((f"{base}.weight_scale_inv", s))
    refs[base] = (w8, s, deq)


def _fabricate_checkpoint(cfg, include_mtp=False):
    """HF-style stream for the reduced config + poison keys that must skip."""
    state: list[tuple[str, torch.Tensor]] = []
    refs: dict[str, tuple] = {}
    hid, q_lora, kv_lora = cfg.hidden_size, cfg.q_lora_rank, cfg.kv_lora_rank
    heads = cfg.num_attention_heads

    state.append(("model.embed_tokens.weight",
                  torch.randn(cfg.vocab_size, hid).bfloat16()))
    state.append(("lm_head.weight", torch.randn(cfg.vocab_size, hid).bfloat16()))
    state.append(("model.norm.weight", torch.randn(hid).bfloat16()))

    for n in range(cfg.num_hidden_layers + (1 if include_mtp else 0)):
        p = f"model.layers.{n}"
        state.append((f"{p}.input_layernorm.weight", torch.randn(hid).bfloat16()))
        state.append((f"{p}.post_attention_layernorm.weight",
                      torch.randn(hid).bfloat16()))
        _fp8_entry(state, refs, f"{p}.self_attn.q_a_proj", (q_lora, hid))
        state.append((f"{p}.self_attn.q_a_layernorm.weight",
                      torch.randn(q_lora).bfloat16()))
        _fp8_entry(state, refs, f"{p}.self_attn.q_b_proj",
                   (heads * cfg.qk_head_dim, q_lora))
        _fp8_entry(state, refs, f"{p}.self_attn.kv_a_proj_with_mqa",
                   (kv_lora + cfg.qk_rope_head_dim, hid))
        state.append((f"{p}.self_attn.kv_a_layernorm.weight",
                      torch.randn(kv_lora).bfloat16()))
        _fp8_entry(state, refs, f"{p}.self_attn.kv_b_proj",
                   (heads * (cfg.qk_nope_head_dim + cfg.v_head_dim), kv_lora))
        _fp8_entry(state, refs, f"{p}.self_attn.o_proj",
                   (hid, heads * cfg.v_head_dim))

        # DSA indexer weights exist only on FULL layers: wq_b/wk are fp8
        # pairs, weights_proj bf16, k_norm a full LayerNorm (weight + bias).
        if is_full_indexer_layer(cfg, n):
            ip = f"{p}.self_attn.indexer"
            _fp8_entry(state, refs, f"{ip}.wq_b",
                       (cfg.index_n_heads * cfg.index_head_dim, q_lora))
            _fp8_entry(state, refs, f"{ip}.wk", (cfg.index_head_dim, hid))
            state.append((f"{ip}.weights_proj.weight",
                          torch.randn(cfg.index_n_heads, hid).bfloat16()))
            state.append((f"{ip}.k_norm.weight",
                          torch.randn(cfg.index_head_dim).bfloat16()))
            state.append((f"{ip}.k_norm.bias",
                          torch.randn(cfg.index_head_dim).bfloat16()))

        if n < cfg.first_k_dense_replace:
            inter = cfg.intermediate_size
            _fp8_entry(state, refs, f"{p}.mlp.gate_proj", (inter, hid))
            _fp8_entry(state, refs, f"{p}.mlp.up_proj", (inter, hid))
            _fp8_entry(state, refs, f"{p}.mlp.down_proj", (hid, inter))
        else:
            moe_inter = cfg.moe_intermediate_size
            state.append((f"{p}.mlp.gate.weight",
                          torch.randn(cfg.n_routed_experts, hid).bfloat16()))
            state.append((f"{p}.mlp.gate.e_score_correction_bias",
                          torch.randn(cfg.n_routed_experts).float()))
            for e in range(cfg.n_routed_experts):
                _fp8_entry(state, refs, f"{p}.mlp.experts.{e}.gate_proj",
                           (moe_inter, hid))
                _fp8_entry(state, refs, f"{p}.mlp.experts.{e}.up_proj",
                           (moe_inter, hid))
                _fp8_entry(state, refs, f"{p}.mlp.experts.{e}.down_proj",
                           (hid, moe_inter))
            _fp8_entry(state, refs, f"{p}.mlp.shared_experts.gate_proj",
                       (moe_inter, hid))
            _fp8_entry(state, refs, f"{p}.mlp.shared_experts.up_proj",
                       (moe_inter, hid))
            _fp8_entry(state, refs, f"{p}.mlp.shared_experts.down_proj",
                       (hid, moe_inter))

    mtp = f"model.layers.{cfg.num_hidden_layers}"
    if include_mtp:
        # The DeepSeek-V3 MTP glue; the decoder inventory for this layer
        # already came out of the loop above.
        state.append((f"{mtp}.enorm.weight", torch.randn(hid).bfloat16()))
        state.append((f"{mtp}.hnorm.weight", torch.randn(hid).bfloat16()))
        _fp8_entry(state, refs, f"{mtp}.eh_proj", (hid, 2 * hid))
        state.append((f"{mtp}.shared_head.norm.weight",
                      torch.randn(hid).bfloat16()))
        return state, refs

    # Poison: MTP layer keys — including the MTP block's own indexer, which
    # must skip by layer index even with load_indexer=True.
    state.append((f"{mtp}.enorm.weight", torch.randn(cfg.hidden_size).bfloat16()))
    state.append((f"{mtp}.mlp.experts.0.gate_proj.weight",
                  torch.randn(cfg.moe_intermediate_size, cfg.hidden_size)
                  .to(torch.float8_e4m3fn)))
    state.append((f"{mtp}.self_attn.indexer.wq_b.weight",
                  torch.randn(cfg.index_n_heads * cfg.index_head_dim,
                              cfg.q_lora_rank).to(torch.float8_e4m3fn)))
    return state, refs


def test_load_end_to_end_reduced_fp8():
    torch.manual_seed(2)
    cfg = Glm52ModelConfig.reduced_fp8(block=BLOCK)
    model = Glm52ForCausalLM(cfg)
    state, refs = _fabricate_checkpoint(cfg)

    loaded = load_glm52_hf_weights(
        model, iter(state), cfg.n_routed_experts, quant=cfg.quant,
        num_hidden_layers=cfg.num_hidden_layers,
    )

    # Completeness both ways: every model param got a tensor, nothing extra.
    assert loaded == set(dict(model.named_parameters()))

    # The quant method keeps the linears fp8: exact bytes.
    q_a = model.model.layers[0].self_attn.q_a_proj.weight
    w8, _, _ = refs["model.layers.0.self_attn.q_a_proj"]
    assert torch.equal(q_a.data, w8.view(torch.uint8))

    # Dense-layer MLP landed in the fused gate_up param.
    mlp0 = model.model.layers[0].mlp
    inter = cfg.intermediate_size
    g8, _, _ = refs["model.layers.0.mlp.gate_proj"]
    u8, _, _ = refs["model.layers.0.mlp.up_proj"]
    assert torch.equal(mlp0.gate_up_proj.weight.data[:inter], g8.view(torch.uint8))
    assert torch.equal(mlp0.gate_up_proj.weight.data[inter:], u8.view(torch.uint8))

    # Routed experts are fp8-resident: exact bytes and fp32 scales.
    moe = model.model.layers[1].mlp
    shard = cfg.moe_intermediate_size
    srow = shard // BLOCK[0]
    for e in range(cfg.n_routed_experts):
        g8, gs, _ = refs[f"model.layers.1.mlp.experts.{e}.gate_proj"]
        u8, us, _ = refs[f"model.layers.1.mlp.experts.{e}.up_proj"]
        d8, ds, _ = refs[f"model.layers.1.mlp.experts.{e}.down_proj"]
        got = moe.experts.gate_up_proj_fp8.data[e]
        assert torch.equal(got[:shard], g8.view(torch.uint8))
        assert torch.equal(got[shard:], u8.view(torch.uint8))
        assert torch.equal(moe.experts.gate_up_proj_scale_inv.data[e, :srow], gs)
        assert torch.equal(moe.experts.gate_up_proj_scale_inv.data[e, srow:], us)
        assert torch.equal(moe.experts.down_proj_fp8.data[e], d8.view(torch.uint8))
        assert torch.equal(moe.experts.down_proj_scale_inv.data[e], ds)
    assert moe.experts.gate_up_proj_scale_inv.dtype == torch.float32

    # Shared expert dequantized into the fused ParallelGatedMLP param.
    _, _, sg_deq = refs["model.layers.1.mlp.shared_experts.gate_proj"]
    shared = moe.shared_expert.gate_up_proj.weight.data
    assert torch.equal(shared[:shard], sg_deq.float())

    # Router selection bias stays fp32.
    assert moe.gate.e_score_correction_bias.dtype == torch.float32


def test_per_token_group_quant_is_compiler_disabled():
    """The quant wrapper must stay outside torch.compile: Inductor's recompile
    of its Triton kernel crashes (PassManager::run failed, in-process and
    subprocess alike) and takes every graph capture down with it. The graph
    break keeps compile and graphs coexisting; this pins the wrap so a
    refactor cannot silently drop it.
    """
    from mstar.utils.quant_fp8 import per_token_group_quant_fp8

    assert hasattr(per_token_group_quant_fp8, "_torchdynamo_disable") or hasattr(
        per_token_group_quant_fp8, "_torchdynamo_orig_callable"
    ), "per_token_group_quant_fp8 lost its torch.compiler.disable wrap"

