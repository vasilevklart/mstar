"""GLM-5.2 on the fp8 block-scale quant method, on CPU: the checkpoint picks the method, the
module layout, TP shards along whole scale blocks, the load, and a reduced model whose forward
is bitwise a bf16 model's given the checkpoint dequantized the same way."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


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


from mstar.distributed.communication import CommGroup  # noqa: E402
from mstar.engine.resources import (  # noqa: E402
    AttentionStep,
    KVStep,
    Segment,
    StepContext,
    SubmoduleStep,
)
from mstar.model.components.distributed import ColumnParallelLinear, ParallelGatedMLP  # noqa: E402
from mstar.model.components.fp8_linear import COLUMN, ROW, Fp8Linear, Fp8ParallelGatedMLP  # noqa: E402
from mstar.model.components.quantization import Fp8BlockQuant  # noqa: E402
from mstar.model.glm52._testing import build_cpu_resources, install_cpu_rope  # noqa: E402
from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM  # noqa: E402
from mstar.model.glm52.config import ATTN_RESOURCE, KV_RESOURCE, Glm52ModelConfig  # noqa: E402
from mstar.model.utils import process_weights_after_loading  # noqa: E402


@pytest.fixture
def cpu_flashinfer(monkeypatch):
    """The CPU forward's RMSNorm; on a box with real flashinfer an earlier import would
    route host tensors into GPU kernels."""
    monkeypatch.setitem(sys.modules, "flashinfer", _cpu_flashinfer())
    install_cpu_rope(monkeypatch)


def _cfg(fp8=True, block=(16, 16)):
    cfg = Glm52ModelConfig.reduced_fp8(block=block) if fp8 else Glm52ModelConfig.reduced()
    cfg.mla_absorb = True
    return cfg


def test_the_checkpoint_picks_the_method():
    from mstar.model.glm52.glm52_model import Glm52Model

    assert Glm52ModelConfig().quant is None  # until the checkpoint's config.json says fp8
    quant = Glm52ModelConfig.reduced_fp8(block=(16, 16)).quant
    assert isinstance(quant, Fp8BlockQuant) and quant.block_size == (16, 16)
    assert quant.kernels == "auto"
    model = Glm52Model("unused", config_variant="reduced_fp8", quant_kernels="reference")
    assert model.config.quant.kernels == "reference"
    with pytest.raises(ValueError, match="quant_kernels"):
        Glm52Model("unused", config_variant="reduced_fp8", quant_kernels="triton")


def test_fp8_linears_except_kv_b_and_the_shared_expert():
    model = Glm52ForCausalLM(_cfg())
    attn = model.model.layers[0].self_attn
    for name in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"):
        lin = getattr(attn, name)
        assert isinstance(lin, Fp8Linear), name
        assert lin.weight.dtype == torch.uint8 and lin.weight_scale_inv.dtype == torch.float32
    assert isinstance(attn.kv_b_proj, ColumnParallelLinear)  # folded into bf16 w_kc / w_vc
    assert tuple(attn.kv_a_proj_with_mqa.weight_scale_inv.shape) == (3, 8)  # 40 rows: ragged
    assert isinstance(model.model.layers[0].mlp, Fp8ParallelGatedMLP)  # dense layer
    # bf16: the decode kernels fuse it into the routed experts' launch
    assert type(model.model.layers[1].mlp.shared_expert) is ParallelGatedMLP

    off = Glm52ForCausalLM(_cfg(fp8=False))
    assert isinstance(off.model.layers[0].self_attn.q_a_proj, torch.nn.Linear)
    assert not any(isinstance(m, Fp8Linear) for m in off.modules())


def test_bf16_autocast_leaves_the_fp8_layout_to_restore():
    from mstar.model.glm52.weight_loader import restore_fp32_params

    lin = Fp8Linear(64, 32, (16, 16))
    lin.to(torch.bfloat16)  # the submodule-load autocast
    assert lin.weight.dtype == torch.uint8
    restore_fp32_params(lin)
    assert lin.weight_scale_inv.dtype == torch.float32


def _full(rows, cols, block):
    """A full checkpoint tensor pair: e4m3 weight and distinct fp32 scales per block."""
    w = (torch.randn(rows, cols) * 0.1).to(torch.float8_e4m3fn)
    s = torch.arange(1, 1 + (-(-rows // block[0])) * (-(-cols // block[1])),
                     dtype=torch.float32).view(-(-rows // block[0]), -1)
    return w, s


def test_tp_shards_take_whole_scale_blocks():
    block = (16, 16)
    group = CommGroup(my_global_rank=1, my_group_rank=1, group_members=[0, 1])  # rank 1 of 2

    col = Fp8Linear(64, 96, block, group, shard=COLUMN)
    w, s = _full(96, 64, block)
    col.weight.weight_loader(col.weight, w)
    col.weight_scale_inv.weight_loader(col.weight_scale_inv, s)
    assert torch.equal(col.weight.data, w[48:].view(torch.uint8))
    assert torch.equal(col.weight_scale_inv.data, s[3:])

    row = Fp8Linear(64, 32, block, group, shard=ROW, reduce_results=True)
    w, s = _full(32, 64, block)
    row.weight.weight_loader(row.weight, w)
    row.weight_scale_inv.weight_loader(row.weight_scale_inv, s)
    assert torch.equal(row.weight.data, w[:, 32:].view(torch.uint8))
    assert torch.equal(row.weight_scale_inv.data, s[:, 2:])

    mlp = Fp8ParallelGatedMLP(64, 64, block, group)
    gu = mlp.gate_up_proj
    (gw, gs), (uw, us) = _full(64, 64, block), _full(64, 64, block)
    for sid, (w, s) in enumerate(((gw, gs), (uw, us))):
        gu.weight.weight_loader(gu.weight, w, sid)
        gu.weight_scale_inv.weight_loader(gu.weight_scale_inv, s, sid)
    assert torch.equal(gu.weight.data, torch.cat([gw[32:], uw[32:]]).view(torch.uint8))
    assert torch.equal(gu.weight_scale_inv.data, torch.cat([gs[2:], us[2:]]))

    with pytest.raises(AssertionError, match="scale block"):
        Fp8Linear(64, 40, block, group, shard=COLUMN)  # 20 rows per rank
    with pytest.raises(TypeError, match="fp8"):
        col.weight.weight_loader(col.weight, torch.zeros(96, 64, dtype=torch.bfloat16))


def _load(cfg, state):
    model = Glm52ForCausalLM(cfg)
    loaded = model.load_weights(iter(state))
    return model, loaded


def test_load_keeps_fp8_bytes_and_dequantizes_kv_b():
    from test_glm52_moe import BLOCK, _fabricate_checkpoint

    torch.manual_seed(2)
    cfg = _cfg(block=BLOCK)
    state, refs = _fabricate_checkpoint(cfg)
    model, loaded = _load(cfg, state)
    assert loaded == set(dict(model.named_parameters()))

    attn = model.model.layers[1].self_attn
    for name in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"):
        w8, s, _ = refs[f"model.layers.1.self_attn.{name}"]
        assert torch.equal(getattr(attn, name).weight.data, w8.view(torch.uint8)), name
        assert torch.equal(getattr(attn, name).weight_scale_inv.data, s), name
    _, _, kv_b = refs["model.layers.1.self_attn.kv_b_proj"]
    assert torch.equal(attn.kv_b_proj.weight.data, kv_b.float())

    shared = model.model.layers[1].mlp.shared_expert.gate_up_proj
    _, _, g = refs["model.layers.1.mlp.shared_experts.gate_proj"]
    assert torch.equal(shared.weight.data[:cfg.moe_intermediate_size], g.float())
    d8, ds, _ = refs["model.layers.0.mlp.down_proj"]
    assert torch.equal(model.model.layers[0].mlp.down_proj.weight.data, d8.view(torch.uint8))
    assert torch.equal(model.model.layers[0].mlp.down_proj.weight_scale_inv.data, ds)

    process_weights_after_loading(model, torch.device("cpu"))
    a8, as_, _ = refs["model.layers.1.self_attn.q_a_proj"]
    k8, ks, _ = refs["model.layers.1.self_attn.kv_a_proj_with_mqa"]
    assert torch.equal(attn.fused_qkv_a_proj_weight, torch.cat([a8, k8]).view(torch.uint8))
    assert torch.equal(attn.fused_qkv_a_proj_scale_inv, torch.cat([as_, ks]))
    assert attn.q_a_proj.weight.numel() == 0 and attn.q_a_proj.weight_scale_inv.numel() == 0


def test_load_refuses_a_linear_without_its_scales():
    from test_glm52_moe import BLOCK, _fabricate_checkpoint

    torch.manual_seed(3)
    cfg = _cfg(block=BLOCK)
    state, _ = _fabricate_checkpoint(cfg)
    state = [(k, v) for k, v in state if k != "model.layers.0.self_attn.o_proj.weight_scale_inv"]
    with pytest.raises(RuntimeError, match="received no checkpoint weights"):
        _load(cfg, state)


def _forward_cpu(model, cfg, ids):
    resources, runner = build_cpu_resources(cfg, ["r0"], page_size=8)
    for layer in model.model.layers:
        layer.self_attn.bind_resources(resources)
    n = ids.shape[0]
    step = SubmoduleStep(segments=[Segment("r0", "main", n)],
                         steps={KV_RESOURCE: KVStep(), ATTN_RESOURCE: AttentionStep(causal=True)})
    step.set_ctx(StepContext(request_ids=("r0",), graph_walk="w", slot=0, capture=False))
    assert runner.admit(step).ok
    runner.plan(step)
    with torch.no_grad():
        logits = model(ids, torch.arange(n))
    runner.commit(step)
    return logits


def test_cpu_forward_is_bitwise_the_bf16_model(cpu_flashinfer):
    """A bf16 model given the checkpoint dequantized as the fp8 model meets each weight: the
    loader's bf16 for what the method dequantizes (kv_b, the shared expert), the reference
    paths' activation dtype (fp32 here) for what it keeps fp8. Every shard, the fused q_a/kv_a
    scales and the SwiGLU halves line up only if the logits agree bit for bit."""
    from test_glm52_moe import BLOCK, _fabricate_checkpoint

    from mstar.model.glm52.weight_loader import quantized_role, skip_phase_b_keys
    from mstar.utils.fp8_block import dequant_fp8_block_stream

    torch.manual_seed(4)
    fp8_cfg = _cfg(block=BLOCK)
    quant = fp8_cfg.quant
    state, _ = _fabricate_checkpoint(fp8_cfg)

    def kept(base):
        role = quantized_role(base)
        return role is not None and quant.keeps_quantized(role)

    # the loader drops the MTP layer's poison keys before it dequantizes; so does this copy
    stream = skip_phase_b_keys(iter(state), fp8_cfg.num_hidden_layers, load_indexer=False)
    stream = dequant_fp8_block_stream(stream, quant.config, keep_fp8=kept)
    dequantized = list(dequant_fp8_block_stream(stream, quant.config, out_dtype=torch.float32))
    logits = []
    for cfg, weights in ((_cfg(fp8=False), dequantized), (fp8_cfg, state)):
        model, _ = _load(cfg, weights)
        process_weights_after_loading(model, torch.device("cpu"))
        ids = torch.randint(0, cfg.vocab_size, (9,), generator=torch.Generator().manual_seed(5))
        logits.append(_forward_cpu(model.eval(), cfg, ids))
    assert torch.isfinite(logits[0]).all()
    assert torch.equal(logits[0], logits[1])
