"""The quantization method (components/quantization.py) as any DeepSeek-V3-shaped model meets
it, here with DeepSeek-V3's own config names and no GLM code: the checkpoint's config picks the
method, the MoE block and the MLA take their layouts from it, and on host tensors the fp8 block
computes what a bf16 block given the dequantized weights computes. Also the kernel tables, keyed
by arch and shape."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.model.components.distributed import ColumnParallelLinear  # noqa: E402
from mstar.model.components.fp8_linear import Fp8Linear, Fp8ParallelGatedMLP  # noqa: E402
from mstar.model.components.mla import MLAAttention  # noqa: E402
from mstar.model.components.quantization import (  # noqa: E402
    Fp8BlockExperts,
    Fp8BlockQuant,
    quant_method_for,
)
from mstar.model.components.shared_expert_moe import Bf16Experts, SharedExpertMoE  # noqa: E402
from mstar.utils.fp8_block import Fp8BlockQuantConfig, dequantize_fp8_block_weight  # noqa: E402

BLOCK = (16, 16)
FP8_CONFIG = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": list(BLOCK),
              "activation_scheme": "dynamic"}


def _deepseek_v3(quant=None, **dims):
    """DeepSeek-V3's config fields at reduced sizes."""
    fields = dict(
        hidden_size=64, n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
        n_shared_experts=1, routed_scaling_factor=2.5, norm_topk_prob=True, hidden_act="silu",
        num_attention_heads=4, q_lora_rank=48, kv_lora_rank=32, qk_nope_head_dim=16,
        qk_rope_head_dim=8, v_head_dim=16, rms_norm_eps=1e-6, mla_absorb=True, quant=quant)
    fields.update(dims)
    cfg = SimpleNamespace(**fields)
    cfg.qk_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    cfg.padded_head_dim = 64
    cfg.softmax_scale = cfg.qk_head_dim ** -0.5
    return cfg


def test_the_checkpoint_config_picks_the_method():
    assert quant_method_for(None) is None
    method = quant_method_for(FP8_CONFIG)
    assert isinstance(method, Fp8BlockQuant) and method.block_size == BLOCK
    assert method.kernels == "auto" and not method.reference
    assert quant_method_for(Fp8BlockQuantConfig(weight_block_size=BLOCK), "reference").reference
    with pytest.raises(ValueError, match="no quantization method"):
        quant_method_for({"quant_method": "awq", "bits": 4})
    with pytest.raises(ValueError, match="quant_kernels"):
        quant_method_for(FP8_CONFIG, kernels="triton")


def test_the_method_decides_what_stays_quantized():
    method = quant_method_for(FP8_CONFIG)
    assert method.keeps_quantized("linear") and method.keeps_quantized("routed_experts")
    # the decode kernels fuse a bf16 shared expert into the routed experts' launch
    assert not method.keeps_quantized("shared_expert")


def test_a_deepseek_v3_block_takes_its_layout_from_the_method():
    quantized = SharedExpertMoE(_deepseek_v3(quant_method_for(FP8_CONFIG)))
    assert isinstance(quantized.experts, Fp8BlockExperts)
    assert quantized.experts.gate_up_proj_fp8.dtype == torch.uint8
    assert tuple(quantized.experts.gate_up_proj_scale_inv.shape) == (8, 4, 4)
    plain = SharedExpertMoE(_deepseek_v3())
    assert isinstance(plain.experts, Bf16Experts)


def test_a_deepseek_v3_mla_takes_its_projections_from_the_method():
    kw = dict(kv_key="kv", attn_key="attn", pos_key="pos")
    attn = MLAAttention(_deepseek_v3(quant_method_for(FP8_CONFIG)), **kw)
    for name in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"):
        assert isinstance(getattr(attn, name), Fp8Linear), name
    assert isinstance(attn.kv_b_proj, ColumnParallelLinear)  # absorbed into bf16 w_kc / w_vc
    plain = MLAAttention(_deepseek_v3(), **kw)
    assert isinstance(plain.q_a_proj, torch.nn.Linear)
    assert not any(isinstance(m, Fp8Linear) for m in plain.modules())


def test_gated_mlp_and_linear_reference_mode():
    method = quant_method_for(FP8_CONFIG, kernels="reference")
    mlp = method.gated_mlp(64, 32)
    assert isinstance(mlp, Fp8ParallelGatedMLP)
    assert mlp.gate_up_proj.reference and mlp.down_proj.reference
    assert method.linear(64, 32).reference


def _quantize(w):
    """(..., N, K) fp32 -> e4m3 bytes and fp32 scales per BLOCK."""
    *lead, n, k = w.shape
    b = w.reshape(-1, n // BLOCK[0], BLOCK[0], k // BLOCK[1], BLOCK[1])
    s = b.abs().amax(dim=(2, 4)) / 448.0
    s = torch.where(s == 0, torch.ones_like(s), s)
    q = (b / s[:, :, None, :, None]).to(torch.float8_e4m3fn)
    return q.reshape(*lead, n, k).view(torch.uint8), s.reshape(*lead, n // BLOCK[0], k // BLOCK[1])


def test_a_deepseek_v3_fp8_block_on_the_host_is_the_bf16_block():
    """No GLM code on the path: the method's reference loop dequantizes exactly what a bf16
    checkpoint would hold, so the two blocks agree bit for bit."""
    torch.manual_seed(0)
    fp8 = SharedExpertMoE(_deepseek_v3(quant_method_for(FP8_CONFIG)))
    bf16 = SharedExpertMoE(_deepseek_v3())
    fp8.process_weights_after_loading("cpu")
    bf16.process_weights_after_loading("cpu")
    assert not fp8.fused  # host tensors: the reference loop
    fp8.gate.weight.data.normal_(0, 0.1)
    fp8.gate.e_score_correction_bias.data.normal_(0, 0.5)
    bf16.gate.load_state_dict(fp8.gate.state_dict())
    fp8.gate.finalize_weights()
    bf16.gate.finalize_weights()
    for p, p8 in zip(bf16.shared_expert.parameters(), fp8.shared_expert.parameters(), strict=True):
        p8.data.normal_(0, 0.05)
        p.data.copy_(p8.data)
    exp8, exp = fp8.experts, bf16.experts
    for name in ("gate_up_proj", "down_proj"):
        q, s = _quantize(torch.randn(getattr(exp, name).shape) * 0.1)
        getattr(exp8, f"{name}_fp8").data.copy_(q)
        getattr(exp8, f"{name}_scale_inv").data.copy_(s)
        getattr(exp, name).data.copy_(torch.stack([
            dequantize_fp8_block_weight(q[e], s[e], BLOCK, out_dtype=torch.float32)
            for e in range(q.shape[0])]))
    x = torch.randn(6, 64) * 0.1
    assert torch.equal(fp8(x), bf16(x))


def test_dense_tiles_are_keyed_by_arch_and_shape():
    pytest.importorskip("triton")
    from mstar.utils import dense_fp8

    swept = dense_fp8._w8a16_tiles(1, 6144, 2048, False, (128, 128), "sm_90")  # GLM-5.2 o_proj
    heuristic = dense_fp8._w8a16_tiles(1, 6144, 2048, False, (128, 128), "sm_100")
    assert swept != heuristic
    assert swept == dict(DOT=False, BN=8, BK=512, SPLIT=1, STAGES=3, num_warps=2)
    # DeepSeek-V3's o_proj at TP8 (16 heads x 128): no sweep yet, the heuristic serves it
    assert dense_fp8._w8a16_tiles(1, 7168, 2048, False, (128, 128), "sm_90") == \
        dense_fp8._w8a16_tiles(1, 7168, 2048, False, (128, 128), None)
