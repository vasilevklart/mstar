"""A tiny weights root with every linear kind ``mstar.quantization`` handles here (QKV, merged gate/up,
row-parallel, a plain ``nn.Linear`` head and router), its checkpoint in bf16 or block fp8, and a load under a
quantization scope the way the engine runs one. Sizes keep every per-rank sub-block a whole 128 at TP 2."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed import QKVParallelLinear, RowParallelLinear
from mstar.model.components.distributed.mlp import ParallelGatedMLP
from mstar.model.components.quant_linear import QuantizableLinears
from mstar.model.loader import iter_safetensors_shards, load_weights_into
from mstar.model.loader.base import LLAMA_STACKED_PARAMS, CheckpointRules
from mstar.quantization import scope as quant_scope
from mstar.quantization.resolve import QuantizationRoot
from mstar.quantization.scope import node_scope
from mstar.utils.fp8_block import quantize_fp8_block_weight

HIDDEN, HEADS, KV_HEADS, HEAD_DIM, INTER, ROUTED, VOCAB, LAYERS = 256, 4, 2, 128, 256, 4, 384, 2

STACKED = list(LLAMA_STACKED_PARAMS)

LINEARS = QuantizableLinears(
    candidates=("layers.*.self_attn.*", "layers.*.mlp.*"), keep_bf16=("lm_head",),
)


class TinyAttention(nn.Module):
    def __init__(self, cg: CommGroup):
        super().__init__()
        self.qkv_proj = QKVParallelLinear(cg, HIDDEN, HEAD_DIM, HEADS, KV_HEADS, bias=False)
        self.o_proj = RowParallelLinear(cg, HEADS * HEAD_DIM, HIDDEN, bias=False, reduce_results=False)


class TinyRouter(nn.Module):
    """A router-shaped weight outside every declared layer (no linear kind: the plan doesn't walk it)."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(ROUTED, HIDDEN))


class TinyLayer(nn.Module):
    def __init__(self, cg: CommGroup):
        super().__init__()
        self.self_attn = TinyAttention(cg)
        self.mlp = ParallelGatedMLP(HIDDEN, INTER, cg, reduce_results=False)
        self.gate = TinyRouter()


class TinyRoot(nn.Module):
    def __init__(self, cg: CommGroup | None = None):
        super().__init__()
        cg = cg or CommGroup.trivial()
        self.layers = nn.ModuleList(TinyLayer(cg) for _ in range(LAYERS))
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)


def group(rank: int = 0, tp: int = 1) -> CommGroup:
    return CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=list(range(tp)))


def build(rank: int = 0, tp: int = 1, device: str = "cpu") -> TinyRoot:
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            return TinyRoot(group(rank, tp))
    finally:
        torch.set_default_dtype(torch.float32)


def full_weights(seed: int = 0) -> dict[str, torch.Tensor]:
    """A bf16 checkpoint's tensors, by checkpoint name."""
    g = torch.Generator().manual_seed(seed)

    def w(*shape):
        return (torch.randn(*shape, generator=g) * 0.05).to(torch.bfloat16)

    out = {"lm_head.weight": w(VOCAB, HIDDEN)}
    for i in range(LAYERS):
        p = f"layers.{i}"
        out[f"{p}.self_attn.q_proj.weight"] = w(HEADS * HEAD_DIM, HIDDEN)
        out[f"{p}.self_attn.k_proj.weight"] = w(KV_HEADS * HEAD_DIM, HIDDEN)
        out[f"{p}.self_attn.v_proj.weight"] = w(KV_HEADS * HEAD_DIM, HIDDEN)
        out[f"{p}.self_attn.o_proj.weight"] = w(HIDDEN, HEADS * HEAD_DIM)
        out[f"{p}.mlp.gate_proj.weight"] = w(INTER, HIDDEN)
        out[f"{p}.mlp.up_proj.weight"] = w(INTER, HIDDEN)
        out[f"{p}.mlp.down_proj.weight"] = w(HIDDEN, INTER)
        out[f"{p}.gate.weight"] = w(ROUTED, HIDDEN)
    return out


def write_checkpoint(root: Path, weights: dict[str, torch.Tensor], fp8=lambda base: False) -> Path:
    """``weights`` as a checkpoint, the bases ``fp8`` picks stored as DeepSeek-style block fp8."""
    root.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for name, tensor in weights.items():
        base = name.removesuffix(".weight")
        if tensor.dim() == 2 and fp8(base):
            q, scale_inv = quantize_fp8_block_weight(tensor)
            tensors[name], tensors[f"{base}.weight_scale_inv"] = q, scale_inv
        else:
            tensors[name] = tensor
    save_file(tensors, str(root / "model.safetensors"))
    config = {}
    if any(t.dtype == torch.float8_e4m3fn for t in tensors.values()):
        config["quantization_config"] = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    (root / "config.json").write_text(json.dumps(config))
    return root


def declaration(checkpoint: Path | None, linears: QuantizableLinears = LINEARS) -> QuantizationRoot:
    return QuantizationRoot(
        node="LLM", root_type=TinyRoot, linears=linears,
        rules=CheckpointRules(stacked_params=tuple(STACKED)),
        checkpoint_dir=lambda: None if checkpoint is None else str(checkpoint),
        build=TinyRoot,
    )


def load(checkpoint: Path, quantization=None, rank: int = 0, tp: int = 1, linears: QuantizableLinears = LINEARS):
    """Build rank ``rank`` of ``tp`` and load ``checkpoint`` into it under the serving block ``quantization``, as
    the engine manager does around ``get_submodule``. Returns ``(root, plan)``; the plan is None where the root
    loaded as it would with no scope."""
    model = SimpleNamespace(get_quantization_roots=lambda: [declaration(checkpoint, linears)])
    config = {} if quantization is None else {"quantization": quantization}
    with node_scope(model, "LLM", config) as scope:
        root = build(rank, tp)
        stream = iter_safetensors_shards(checkpoint, quant_methods=quant_scope.quant_methods())
        load_weights_into(root, stream, stacked_params=STACKED)
        scope.finish(root)
    plan = next(iter(scope.plans.values()))[1] if scope.plans else None
    return root, plan


@contextlib.contextmanager
def default_dtype(dtype):
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)
