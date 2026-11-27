"""The per-layer plan from a node's spec, its model's declaration and its checkpoint, and the specs it refuses
at boot. One rank's root is built on the meta device at the deployment's TP size, as the boot check does."""

import pytest
import torch
from quant_helpers import HEADS, HIDDEN, INTER, KV_HEADS, LAYERS, ROUTED, VOCAB, TinyRoot, declaration, group

from mstar.model.components.quant_linear import QuantizableLinears
from mstar.quantization.checkpoint import CheckpointQuant, TensorInfo
from mstar.quantization.resolve import NO_CUDA_CAPABILITY, QUANT_METHODS, Action, resolve_plan
from mstar.quantization.spec import parse_quantization

PER_LAYER = ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj")


def meta_root(tp=1):
    with torch.device("meta"):
        return TinyRoot(group(0, tp))


def plan_for(quantization, ckpt=None, tp=1, capability=None, linears=None):
    decl = declaration(None) if linears is None else declaration(None, linears)
    return resolve_plan(meta_root(tp), decl, parse_quantization(quantization).for_node("LLM"), ckpt, capability)


def fp8_checkpoint(fp8=lambda base: True) -> CheckpointQuant:
    """Headers of the tiny model's checkpoint with the bases ``fp8`` picks in block fp8."""
    shapes = {"lm_head": (VOCAB, HIDDEN)}
    for i in range(LAYERS):
        p = f"layers.{i}"
        shapes |= {f"{p}.self_attn.q_proj": (HEADS * 128, HIDDEN), f"{p}.self_attn.k_proj": (KV_HEADS * 128, HIDDEN),
                   f"{p}.self_attn.v_proj": (KV_HEADS * 128, HIDDEN), f"{p}.self_attn.o_proj": (HIDDEN, HEADS * 128),
                   f"{p}.mlp.gate_proj": (INTER, HIDDEN), f"{p}.mlp.up_proj": (INTER, HIDDEN),
                   f"{p}.mlp.down_proj": (HIDDEN, INTER), f"{p}.gate": (ROUTED, HIDDEN)}
    tensors = {}
    for base, (n, k) in shapes.items():
        if fp8(base):
            tensors[f"{base}.weight"] = TensorInfo("F8_E4M3", (n, k))
            tensors[f"{base}.weight_scale_inv"] = TensorInfo("F32", (-(-n // 128), -(-k // 128)))
        else:
            tensors[f"{base}.weight"] = TensorInfo("BF16", (n, k))
    return CheckpointQuant.from_tensors(tensors, {"quant_method": "fp8", "weight_block_size": [128, 128]})


def actions(plan):
    return {path: (layer.runtime, layer.action) for path, layer in plan.layers.items()}


def test_the_plan_loads_quant_method_fp8():
    # what the declared roots' loaders let through #1357's checkpoint guard
    assert QUANT_METHODS == ("fp8",)


def test_a_bf16_checkpoint_quantizes_every_candidate_once_loaded():
    plan = plan_for("fp8_block")
    assert not plan.problems and not plan.notes
    for i in range(LAYERS):
        for leaf in PER_LAYER:
            assert actions(plan)[f"layers.{i}.{leaf}"] == ("fp8_block", Action.QUANTIZE)
    assert plan.layers["lm_head"].action is Action.KEEP and plan.layers["lm_head"].origin == "model"


def test_no_block_and_a_bf16_checkpoint_change_nothing():
    plan = plan_for(None)
    assert not plan.active and not plan.problems


def test_rules_go_first_and_say_where_they_came_from():
    plan = plan_for({"format": "fp8_block", "layers": [{"match": "layers.0.**", "format": "bf16"},
                                                       {"match": "layers.*.self_attn.*", "format": "fp8_channel"}]})
    assert actions(plan)["layers.0.self_attn.qkv_proj"] == ("bf16", Action.KEEP)
    assert actions(plan)["layers.1.self_attn.qkv_proj"] == ("fp8_channel", Action.QUANTIZE)
    assert plan.layers["layers.1.self_attn.qkv_proj"].origin == "quantization.layers[1]"
    assert actions(plan)["layers.1.mlp.down_proj"] == ("fp8_block", Action.QUANTIZE)


def test_a_pre_quantized_checkpoint_serves_as_stored_by_default():
    plan = plan_for(None, fp8_checkpoint(lambda base: base != "lm_head"))
    assert not plan.problems
    assert actions(plan)["layers.1.self_attn.qkv_proj"] == ("fp8_block", Action.DIRECT)
    assert plan.layers["layers.1.self_attn.qkv_proj"].block == (128, 128)
    assert actions(plan)["layers.1.mlp.down_proj"] == ("fp8_block", Action.DIRECT)
    # the router is no declared layer: loaded dequantized, as bf16
    assert plan.layers["layers.1.gate"].kind == "other"
    assert actions(plan)["layers.1.gate"] == ("bf16", Action.DEQUANT)


def test_keeping_a_stored_layer_bf16_dequantizes_it_as_it_loads():
    plan = plan_for({"keep_bf16": ["layers.0.**"]}, fp8_checkpoint(lambda base: base != "lm_head"))
    assert actions(plan)["layers.0.mlp.gate_up_proj"] == ("bf16", Action.DEQUANT)
    assert actions(plan)["layers.1.mlp.gate_up_proj"] == ("fp8_block", Action.DIRECT)


def test_a_stored_linear_the_model_keeps_bf16_is_dequantized():
    plan = plan_for(None, fp8_checkpoint())
    assert actions(plan)["lm_head"] == ("bf16", Action.DEQUANT)


def test_a_fused_layer_stored_half_quantized_is_dequantized_or_requantized():
    # Qwen3.5-27B-FP8 keeps in_proj_a/b bf16 beside fp8 in_proj_qkv/z, all one fused GEMM here
    mixed = fp8_checkpoint(lambda base: base.endswith("q_proj"))
    plan = plan_for(None, mixed)
    assert actions(plan)["layers.0.self_attn.qkv_proj"] == ("bf16", Action.DEQUANT)
    assert any("different formats" in note for note in plan.notes)
    plan = plan_for("fp8_block", mixed)
    assert actions(plan)["layers.0.self_attn.qkv_proj"] == ("fp8_block", Action.REQUANT)


def test_at_tp4_the_checkpoint_blocks_straddle_and_the_shard_is_requantized():
    plan = plan_for(None, fp8_checkpoint(lambda base: base != "lm_head"), tp=4)
    assert not plan.problems
    # gate and up are 64 rows a rank: a 128-row scale block covers both, so the loaded bf16 is re-quantized
    assert actions(plan)["layers.0.mlp.gate_up_proj"] == ("fp8_block", Action.REQUANT)
    # down's K is 64 a rank: no 128 group fits, it serves bf16
    assert actions(plan)["layers.0.mlp.down_proj"] == ("bf16", Action.DEQUANT)
    assert actions(plan)["layers.0.self_attn.o_proj"] == ("fp8_block", Action.DIRECT)


def test_a_default_that_doesnt_fit_a_layer_is_a_note_a_rule_is_a_problem():
    plan = plan_for("fp8_block", tp=4)
    assert actions(plan)["layers.0.mlp.down_proj"] == ("bf16", Action.KEEP)
    assert any("layers.0.mlp.down_proj: stays bf16" in note for note in plan.notes)
    plan = plan_for({"layers": [{"match": "layers.0.mlp.down_proj", "format": "fp8_block"}]}, tp=4)
    assert len(plan.problems) == 1 and "whole 128x128 blocks" in plan.problems[0]


@pytest.mark.parametrize("quantization, ckpt, capability, message", [
    ({"layers": [{"match": "lm_head", "format": "fp8_block"}]}, None, None, "which the model keeps bf16"),
    ({"layers": [{"match": "layers.*.attn.*", "format": "bf16"}]}, None, None, "matches none of its"),
    ("fp8_channel", "fp8", None, "compounds two roundings"),
    ("fp8_block", None, (8, 0), "needs compute capability 8.9"),
    # a "cuda" device with no CUDA compute capability (ROCm): refused, not taken as runnable
    ("fp8_block", None, NO_CUDA_CAPABILITY, "this GPU has no CUDA compute capability"),
    # catalogued, so named, but not served here yet
    ("nvfp4", None, (10, 0), "this build serves nvfp4 on no linear layer yet"),
    ("w4a16", None, (9, 0), "this build serves w4a16 on no linear layer yet"),
    ("mxfp8", None, None, "this build serves mxfp8 on no linear layer yet"),
])
def test_specs_the_build_cant_serve_are_refused(quantization, ckpt, capability, message):
    plan = plan_for(quantization, fp8_checkpoint() if ckpt else None, capability=capability)
    assert any(message in problem for problem in plan.problems), plan.problems


def test_a_stored_fp8_checkpoint_on_a_gpu_without_fp8_serves_dequantized():
    # `auto` keeps a stored layer the GPU can't run as bf16 (exact up to the fp8 weights), noted per layer
    plan = plan_for(None, fp8_checkpoint(lambda base: base != "lm_head"), capability=(8, 0))
    assert not plan.problems
    assert actions(plan)["layers.0.mlp.gate_up_proj"] == ("bf16", Action.DEQUANT)
    assert any("needs compute capability 8.9" in note for note in plan.notes)


@pytest.mark.parametrize("tensors, method, refused", [
    # compressed-tensors' NVFP4 and int4, AWQ: no loader yet
    ({"layers.0.mlp.gate_proj.weight_packed": TensorInfo("U8", (INTER, HIDDEN // 2)),
      "layers.0.mlp.gate_proj.weight_scale": TensorInfo("F8_E4M3", (INTER, HIDDEN // 16)),
      "layers.0.mlp.gate_proj.weight_global_scale": TensorInfo("F32", (1,))}, "compressed-tensors",
     "nvfp4 (compressed-tensors)"),
    ({"layers.0.mlp.gate_proj.qweight": TensorInfo("I32", (HIDDEN, INTER // 8)),
      "layers.0.mlp.gate_proj.qzeros": TensorInfo("I32", (HIDDEN // 128, INTER // 8)),
      "layers.0.mlp.gate_proj.scales": TensorInfo("F16", (HIDDEN // 128, INTER))}, "awq", "w4a16 (awq)"),
    # fp8 in another checkpoint layout (compressed-tensors' float-quantized, per channel): not loaded yet either
    ({"layers.0.mlp.gate_proj.weight": TensorInfo("F8_E4M3", (INTER, HIDDEN)),
      "layers.0.mlp.gate_proj.weight_scale": TensorInfo("F32", (INTER, 1))}, "compressed-tensors",
     "fp8_channel (compressed-tensors)"),
])
def test_a_checkpoint_format_without_a_loader_is_refused_by_name(tensors, method, refused):
    plan = plan_for(None, CheckpointQuant.from_tensors(tensors, {"quant_method": method}))
    assert any(f"as {refused}" in p and f"no loader for {refused}" in p for p in plan.problems), plan.problems


def test_a_linear_the_model_didnt_declare_is_a_problem_not_a_silent_skip():
    plan = plan_for("fp8_block", linears=QuantizableLinears(candidates=("layers.*.mlp.*",), keep_bf16=("lm_head",)))
    assert any("neither a low-precision candidate nor kept bf16" in p for p in plan.problems)


def test_a_rule_refused_on_many_layers_is_one_problem():
    plan = plan_for({"layers": [{"match": "layers.*.mlp.down_proj", "format": "fp8_block"}]}, tp=4)
    assert len(plan.problems) == 1
    layers = "['layers.0.mlp.down_proj', 'layers.1.mlp.down_proj']"
    assert plan.problems[0].startswith(f"node 'LLM': 2 layers (e.g. {layers}) can't run fp8_block")


def test_a_rule_whose_reason_reads_like_a_fatal_one_is_still_a_note_under_the_default(monkeypatch):
    # what makes a problem fatal is its kind, not its words: a layer's own reason starting "this build" stays a note
    from mstar.model.components.distributed import RowParallelLinear
    from mstar.model.components.quant_linear import QuantCapability

    monkeypatch.setattr(RowParallelLinear, "quant_capability",
                        lambda self: QuantCapability(False, "this build's row-parallel linears can't"))
    plan = plan_for("fp8_block")
    assert not plan.problems
    assert actions(plan)["layers.0.self_attn.o_proj"] == ("bf16", Action.KEEP)
    assert any("this build's row-parallel linears can't" in note for note in plan.notes)


def test_a_stored_partial_last_row_block_serves_as_stored_where_no_shard_cuts_it():
    # GLM's kv_a_proj_with_mqa is 576 rows: four whole 128-row blocks and a partial one
    from torch import nn

    from mstar.model.components.distributed import ColumnParallelLinear
    from mstar.model.loader.base import CheckpointRules
    from mstar.quantization.resolve import QuantizationRoot

    class Root(nn.Module):
        def __init__(self, cg):
            super().__init__()
            self.proj = ColumnParallelLinear(cg, 256, 576, bias=False)

    decl = QuantizationRoot("LLM", Root, QuantizableLinears(("proj",), ()), CheckpointRules(), lambda: None, Root)
    ckpt = CheckpointQuant.from_tensors({"proj.weight": TensorInfo("F8_E4M3", (576, 256)),
                                         "proj.weight_scale_inv": TensorInfo("F32", (5, 2))},
                                        {"quant_method": "fp8"})
    spec = parse_quantization(None).for_node("LLM")
    for tp, action in ((1, Action.DIRECT), (2, Action.DEQUANT)):
        with torch.device("meta"):
            plan = resolve_plan(Root(group(0, tp)), decl, spec, ckpt)
        assert not plan.problems and plan.layers["proj"].action is action, (tp, plan.layers["proj"])
    # 288 rows a rank cut a block, and no whole block remains to quantize: bf16, with the reason logged
    assert "straddle" in plan.notes[0]
    # quantizing at load still takes whole row blocks
    with torch.device("meta"):
        plan = resolve_plan(Root(group(0, 1)), decl, parse_quantization("fp8_block").for_node("LLM"), None)
    assert plan.layers["proj"].action is Action.KEEP and "stays bf16" in plan.notes[0]
