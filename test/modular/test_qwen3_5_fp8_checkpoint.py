"""Qwen3.5 dense from an FP8 checkpoint, on CPU, through the real loader: a tiny Qwen3.5 written by transformers,
its decoder linears stored as Qwen's ``*-FP8`` stores them (``quant_method: fp8``, e4m3 weights with a bf16
``weight_scale_inv`` per 128 x 128 block, the delta net's ``in_proj_a`` / ``in_proj_b`` kept bf16, the vision
tower and the head bf16), loaded as the engine manager loads a node (``load_submodule``). Main refuses the
checkpoint at #1357's guard; here each layer holds the checkpoint's own bytes and scales."""

import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).parent))
from test_qwen3_5_video import write_synthetic_checkpoint  # noqa: E402  (sibling helper)

from mstar.distributed.communication import CommGroup  # noqa: E402
from mstar.model.components.distributed import (  # noqa: E402
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from mstar.model.components.distributed.linear_attn import _FusedBlockColumnParallelLinear  # noqa: E402
from mstar.model.loader.quantization import UnservedQuantizationError  # noqa: E402
from mstar.quantization.resolve import Action  # noqa: E402
from mstar.quantization.scope import load_submodule, node_scope  # noqa: E402
from mstar.quantization.validate import quantization_problems  # noqa: E402
from mstar.utils.fp8_block import dequantize_fp8_block_weight, quantize_fp8_block_weight  # noqa: E402

# whole 128 blocks for every projection; the delta net's a / b gate rows are a head each (2 rows)
TEXT = dict(hidden_size=256, intermediate_size=256, head_dim=128, num_attention_heads=2, num_key_value_heads=1,
            linear_key_head_dim=128, linear_value_head_dim=128, tie_word_embeddings=False)
PLACED = {"node_groups": [{"node_names": ["LLM", "vision_encoder"], "ranks": [0]}]}
TEXT_PREFIX = "model.language_model."


def _to_fp8(src: Path, dst: Path) -> None:
    """``src`` as Qwen's FP8 checkpoints store it: every 2-D decoder linear but ``in_proj_a`` / ``in_proj_b`` as
    e4m3 plus a bf16 ``weight_scale_inv`` per 128 x 128 block; the rest (norms, conv, head, embedding, vision) as
    it is, listed under ``modules_to_not_convert``."""
    dst.mkdir()
    out, kept = {}, []
    for shard in sorted(src.glob("*.safetensors")):
        for name, tensor in load_file(str(shard)).items():
            base = name.removesuffix(".weight")
            if (name.startswith(TEXT_PREFIX + "layers.") and name.endswith(".weight") and tensor.dim() == 2
                    and not base.endswith(("in_proj_a", "in_proj_b"))):
                q, scale_inv = quantize_fp8_block_weight(tensor)
                out[name], out[base + ".weight_scale_inv"] = q, scale_inv.to(torch.bfloat16)
            else:
                out[name] = tensor
                if name.endswith(".weight") and tensor.dim() >= 2:
                    kept.append(base)
    save_file(out, str(dst / "model.safetensors"), metadata={"format": "pt"})
    for path in src.iterdir():
        if path.suffix != ".safetensors" and path.name != "model.safetensors.index.json" and path.is_file():
            (dst / path.name).write_bytes(path.read_bytes())
    config = json.loads((src / "config.json").read_text())
    config["quantization_config"] = {
        "quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic", "weight_block_size": [128, 128],
        "modules_to_not_convert": sorted(kept),
    }
    (dst / "config.json").write_text(json.dumps(config))


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    root = write_synthetic_checkpoint(tmp_path_factory.mktemp("qwen3_5_bf16"))
    config = json.loads((root / "config.json").read_text())
    config["text_config"].update(TEXT)
    # past the tokenizer's ids to a multiple of 128, so the vocab shards at TP 2
    config["text_config"]["vocab_size"] = 384
    config["vision_config"]["out_hidden_size"] = TEXT["hidden_size"]
    config["tie_word_embeddings"] = False
    torch.manual_seed(0)
    hf = Qwen3_5ForConditionalGeneration(Qwen3_5Config(**config)).to(torch.bfloat16)
    hf.save_pretrained(root)
    fp8 = tmp_path_factory.mktemp("qwen3_5_fp8") / "ckpt"
    _to_fp8(root, fp8)
    return {"bf16": root, "fp8": fp8, "tensors": load_file(str(fp8 / "model.safetensors"))}


def _model(path):
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    return Qwen3_5DenseModel(str(path))


def _llm(path, quantization=None):
    """The LLM node loaded as ``load_submodule`` (the engine manager's call) loads it, and its plan (None where it
    loaded with none)."""
    model = _model(path)
    config = {} if quantization is None else {"quantization": quantization}
    with node_scope(model, "LLM", config, "cpu") as scope:
        submodule = model.get_submodule("LLM", "cpu")
        scope.finish(submodule)
    plans = scope.plans
    return submodule.model, (next(iter(plans.values()))[1] if plans else None)


def _stored(tensors, base):
    """A checkpoint linear's e4m3 weight and its fp32 block scales."""
    return tensors[f"{TEXT_PREFIX}{base}.weight"], tensors[f"{TEXT_PREFIX}{base}.weight_scale_inv"].float()


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_main_refuses_the_checkpoint_outside_the_engines_scope(checkpoints):
    # a bare get_submodule, as before this plan: #1357's guard, at the iterator
    with pytest.raises(UnservedQuantizationError, match="quant_method fp8"):
        _model(checkpoints["fp8"]).get_submodule("LLM", "cpu")


def test_the_fp8_checkpoint_loads_as_stored_into_the_tensor_parallel_linears(checkpoints):
    lm, plan = _llm(checkpoints["fp8"])
    tensors = checkpoints["tensors"]
    full, linear = lm.model.layers[1], lm.model.layers[0]  # layer_types: [linear_attention, full_attention]

    expected = {
        "model.layers.1.self_attn.qkv_proj": (QKVParallelLinear, Action.DIRECT),
        "model.layers.1.self_attn.o_proj": (RowParallelLinear, Action.DIRECT),
        "model.layers.0.self_attn.out_proj": (RowParallelLinear, Action.DIRECT),
        "model.layers.0.mlp.gate_up_proj": (MergedColumnParallelLinear, Action.DIRECT),
        "model.layers.1.mlp.down_proj": (RowParallelLinear, Action.DIRECT),
        # [q|k|v|z] fp8 beside bf16 a / b: dequantized and quantized again at load, the gate rows kept bf16
        "model.layers.0.self_attn.in_proj_fused": (_FusedBlockColumnParallelLinear, Action.REQUANT),
        "lm_head": (ColumnParallelLinear, Action.KEEP),
    }
    for path, (kind, action) in expected.items():
        assert type(lm.get_submodule(path)) is kind, path
        assert plan.layers[path].action is action, (path, plan.layers[path])
        assert plan.layers[path].runtime == ("bf16" if action is Action.KEEP else "fp8_block"), path
    assert not plan.problems

    # a DIRECT layer holds the checkpoint's bytes and scales, in integer containers, as fp8_block
    qkv = full.self_attn.qkv_proj
    assert qkv.quant_format.name == "fp8_block" and qkv.weight.dtype == torch.uint8
    assert qkv.weight_scale_inv.dtype == torch.int32
    parts = [_stored(tensors, f"layers.1.self_attn.{p}_proj") for p in "qkv"]
    assert torch.equal(qkv.weight, torch.cat([w for w, _ in parts]).view(torch.uint8))
    assert torch.equal(qkv.weight_scale_inv.view(torch.float32), torch.cat([s for _, s in parts]))
    down_w, down_s = _stored(tensors, "layers.1.mlp.down_proj")
    assert torch.equal(full.mlp.down_proj.weight, down_w.view(torch.uint8))
    assert torch.equal(full.mlp.down_proj.weight_scale_inv.view(torch.float32), down_s)
    assert full.mlp.down_proj.weight_scale_inv.shape == (TEXT["hidden_size"] // 128, TEXT["intermediate_size"] // 128)

    # the fused delta-net input: q|k|v|z in fp8_block, a / pad / b as the checkpoint's bf16
    fused = linear.self_attn.in_proj_fused
    head = sum(fused.output_sizes[:4])
    assert fused.quant_format.name == "fp8_block" and fused.weight.shape == (head, TEXT["hidden_size"])
    a, b = (tensors[f"{TEXT_PREFIX}layers.0.linear_attn.in_proj_{g}.weight"] for g in "ab")
    pad = fused.output_sizes[5]
    assert torch.equal(fused.weight_tail, torch.cat([a, torch.zeros(pad, a.shape[1], dtype=a.dtype), b]))

    # bf16 where the checkpoint is: the head, the embedding, the norms and the conv
    assert lm.lm_head.weight.dtype == torch.bfloat16 and lm.lm_head.quant_format is None
    assert lm.model.embed_tokens.weight.dtype == torch.bfloat16
    assert linear.self_attn.conv1d.weight.dtype == torch.bfloat16

    # the engine manager's own call loads the same tensors
    engine = load_submodule(_model(checkpoints["fp8"]), "LLM", {}, "cpu").model
    assert engine.state_dict().keys() == lm.state_dict().keys()
    assert all(torch.equal(t, lm.state_dict()[name]) for name, t in engine.state_dict().items())


def test_a_stored_layer_computes_its_dequantized_checkpoint_weight(checkpoints):
    # on CPU the format runs its reference: x times the checkpoint's e4m3 weight times its scales, in bf16
    lm, _ = _llm(checkpoints["fp8"])
    tensors = checkpoints["tensors"]
    x = torch.randn(5, TEXT["hidden_size"], dtype=torch.bfloat16)
    gate_up = torch.cat([dequantize_fp8_block_weight(*_stored(tensors, f"layers.1.mlp.{p}_proj"))
                         for p in ("gate", "up")]).to(torch.bfloat16)
    assert torch.equal(lm.model.layers[1].mlp.gate_up_proj(x), F.linear(x, gate_up))


def test_the_fp8_layers_track_the_bf16_checkpoint(checkpoints):
    ref, ref_plan = _llm(checkpoints["bf16"])
    lm, _ = _llm(checkpoints["fp8"])
    assert ref_plan is None
    torch.manual_seed(0)
    x = torch.randn(7, TEXT["hidden_size"], dtype=torch.bfloat16)
    for path in ("model.layers.1.self_attn.qkv_proj", "model.layers.0.self_attn.in_proj_fused",
                 "model.layers.0.mlp", "model.layers.1.mlp"):
        out, want = lm.get_submodule(path)(x), ref.get_submodule(path)(x)
        # e4m3 keeps 3 mantissa bits: about 2-3% of a GEMM over random rows; a wrong scale or row is order 1
        assert rel(out, want) < 0.05, path
    o = torch.randn(7, lm.model.layers[1].self_attn.o_proj.input_size_per_partition, dtype=torch.bfloat16)
    assert rel(lm.model.layers[1].self_attn.o_proj(o), ref.model.layers[1].self_attn.o_proj(o)) < 0.05


def test_a_bf16_checkpoint_takes_mains_path(checkpoints):
    # no `quantization:` block and a checkpoint that declares none: no plan, the same tensors as a bare load
    lm, plan = _llm(checkpoints["bf16"])
    bare = _model(checkpoints["bf16"]).get_submodule("LLM", "cpu").model
    assert plan is None
    assert lm.state_dict().keys() == bare.state_dict().keys()
    assert all(torch.equal(t, bare.state_dict()[name]) for name, t in lm.state_dict().items())
    assert not any(t.dtype in (torch.uint8, torch.int32) for t in lm.state_dict().values())


def test_a_bf16_checkpoint_quantizes_at_load_when_asked(checkpoints):
    lm, plan = _llm(checkpoints["bf16"], "fp8_block")
    assert plan.layers["model.layers.1.mlp.gate_up_proj"].action is Action.QUANTIZE
    assert lm.model.layers[1].mlp.gate_up_proj.quant_format.name == "fp8_block"
    assert plan.layers["lm_head"].action is Action.KEEP


@pytest.mark.parametrize("tp", [2])
def test_each_rank_loads_its_rows_and_scale_rows(checkpoints, tp):
    tensors = checkpoints["tensors"]
    for rank in range(tp):
        model = _model(checkpoints["fp8"])
        group = CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=list(range(tp)))
        # `_create_submodule` (no TP all-reduce fusion to set up on CPU), under the node's scope
        with node_scope(model, "LLM", {}) as scope:
            lm = model._create_submodule("LLM", "cpu", torch.bfloat16, group).model
            scope.finish(lm)
        down = lm.model.layers[1].mlp.down_proj
        w, s = _stored(tensors, "layers.1.mlp.down_proj")
        assert torch.equal(down.weight, w.view(torch.uint8).chunk(tp, 1)[rank])
        assert torch.equal(down.weight_scale_inv.view(torch.float32), s.chunk(tp, 1)[rank])
        gate_up = lm.model.layers[1].mlp.gate_up_proj
        scales = [_stored(tensors, f"layers.1.mlp.{p}_proj")[1].chunk(tp)[rank] for p in ("gate", "up")]
        assert torch.equal(gate_up.weight_scale_inv.view(torch.float32), torch.cat(scales))


def test_the_vision_tower_of_an_fp8_checkpoint_loads_bf16(checkpoints):
    # Qwen's FP8 checkpoints keep the tower bf16; it loads past the guard, and with the stored values
    model = _model(checkpoints["fp8"])
    tower = load_submodule(model, "vision_encoder", {}, "cpu").model
    name, tensor = next((n, t) for n, t in checkpoints["tensors"].items() if n.startswith("model.visual."))
    assert torch.equal(tower.state_dict()[name.removeprefix("model.visual.")], tensor)


def test_the_boot_check_passes_the_checkpoint_and_refuses_a_second_rounding(checkpoints):
    fp8 = _model(checkpoints["fp8"])
    assert quantization_problems(fp8, PLACED) == []
    problems = quantization_problems(fp8, {**PLACED, "quantization": "fp8_channel"})
    assert any("compounds two roundings" in p for p in problems), problems
    assert quantization_problems(_model(checkpoints["bf16"]), {**PLACED, "quantization": "fp8_block"}) == []
    assert quantization_problems(_model(checkpoints["bf16"]), {
        **PLACED, "quantization": {"nodes": {"vision_encoder": "fp8_block"}}}) == [
        "quantization.nodes.vision_encoder: Qwen3_5DenseModel declares no quantizable layers there",
    ]


def test_the_boot_check_refuses_an_unserved_method_from_the_declaration_alone(checkpoints, tmp_path):
    # the conductor holds only a repo's metadata: config.json declares AWQ and no shard is local yet
    meta = tmp_path / "meta"
    meta.mkdir()
    for path in checkpoints["bf16"].iterdir():
        if path.suffix != ".safetensors" and path.name != "model.safetensors.index.json" and path.is_file():
            (meta / path.name).write_bytes(path.read_bytes())
    config = json.loads((meta / "config.json").read_text())
    config["quantization_config"] = {"quant_method": "awq", "bits": 4, "group_size": 128}
    (meta / "config.json").write_text(json.dumps(config))
    (problem,) = quantization_problems(_model(meta), PLACED)
    assert "declares quant_method awq" in problem and "quant_method fp8 only" in problem
