"""GLM-5.2's layers carry (partial, residual) and reduce each all-reduce inside the next
residual add + RMSNorm (``RMSNorm.forward_residual``), as Qwen3.5's do: the row-parallel
outputs stay partial, the MoE block sums its two halves before one reduction, and every norm
reduces through the group under TP."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_glm52_engine_cycle import (  # noqa: E402
    _cfg,
    _drive,
    _Driver,
    _force_cpu_flashinfer,  # noqa: F401 — autouse here too
    _fwd_info,
)

from mstar.model.components.shared_expert_moe import SharedExpertMoE  # noqa: E402
from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM  # noqa: E402
from mstar.model.glm52.components.language_model import build_dense_mlp  # noqa: E402
from mstar.model.glm52.config import Glm52ModelConfig  # noqa: E402
from mstar.model.glm52.submodules import Glm52LLMSubmodule  # noqa: E402
from mstar.model.utils import process_weights_after_loading  # noqa: E402


class _RecordingGroup:
    """A 2-rank CommGroup stand-in for one process: collectives are identity ops that record
    what was reduced. Shard sizes follow world_size."""

    def __init__(self, rank=0, world_size=2):
        self.rank = rank
        self.global_rank = rank
        self.world_size = world_size
        self.group_members = list(range(world_size))
        self.device_group = None
        self.reduced: list[torch.Tensor] = []

    def all_reduce(self, t):
        self.reduced.append(t.clone())
        return t

    def all_gather(self, t, dim=-1):
        return torch.cat([t] * self.world_size, dim=dim)

    def reduce_scatter(self, t, dim=-1):
        return t


class _FusingGroup(_RecordingGroup):
    """Records each fused all-reduce + add + RMSNorm; one process, so the reduce is identity."""

    def __init__(self):
        super().__init__()
        self.fused = 0

    def allreduce_add_rmsnorm(self, x, residual, weight, eps, weight_bias=0.0):
        self.fused += 1
        r = x + residual
        v = r.float()
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
        return (v * (weight.float() + weight_bias)).to(r.dtype), r


def _model(cfg, comm_group=None, seed=0) -> Glm52ForCausalLM:
    """test_glm52_engine_cycle's model, on a given group: same draws, same weights."""
    torch.manual_seed(seed)
    model = Glm52ForCausalLM(cfg, comm_group=comm_group)
    for name, p in model.named_parameters():
        if "norm" in name:
            p.data.normal_(1.0, 0.02)
        elif name.endswith("gate.weight") or "e_score_correction_bias" in name:
            p.data.normal_(0, 1.0)
        else:
            p.data.normal_(0, 0.05)
    process_weights_after_loading(model, torch.device("cpu"))
    return model.eval()


def test_the_row_parallel_outputs_stay_partial():
    cfg = Glm52ModelConfig.reduced()
    cfg.num_hidden_layers = cfg.first_k_dense_replace + 1
    model = Glm52ForCausalLM(cfg)
    kinds = set()
    for layer in model.model.layers:
        assert not layer.self_attn.o_proj.reduce_results
        mlp = layer.mlp
        kinds.add(type(mlp))
        if isinstance(mlp, SharedExpertMoE):
            assert not mlp.reduce_results and not mlp.shared_expert.down_proj.reduce_results
        else:
            assert not mlp.down_proj.reduce_results
    assert len(kinds) == 2  # dense and MoE layers


def test_the_moe_block_reduces_its_two_halves_once():
    """A reducing block issues one all-reduce, of routed + shared, and a partial block none,
    returning that same sum."""
    torch.manual_seed(3)
    cfg = Glm52ModelConfig.reduced()
    g_reduce, g_partial = _RecordingGroup(), _RecordingGroup()
    reducing = SharedExpertMoE(cfg, comm_group=g_reduce)
    partial = SharedExpertMoE(cfg, comm_group=g_partial, reduce_results=False)
    for p in reducing.parameters():
        p.data.normal_(0, 0.05)
    partial.load_state_dict(reducing.state_dict())
    x = torch.randn(5, cfg.hidden_size) * 0.1
    reducing(x)
    out = partial(x)
    assert g_partial.reduced == [] and len(g_reduce.reduced) == 1
    assert torch.equal(out, g_reduce.reduced[0].view_as(out))


def test_dense_mlp_partial_skips_the_reduce():
    cfg = Glm52ModelConfig.reduced()
    group = _RecordingGroup()
    mlp = build_dense_mlp(cfg, comm_group=group, reduce_results=False)
    for p in mlp.parameters():
        p.data.normal_(0, 0.05)
    mlp(torch.randn(3, cfg.hidden_size))
    assert group.reduced == []


def test_norms_reduce_through_the_group_under_tp():
    cfg = _cfg(True)
    group = _FusingGroup()
    model = _model(cfg, group)
    seen = []
    model.lm_head.register_forward_hook(lambda m, args, out: seen.append(args[0].clone()))
    sub = Glm52LLMSubmodule(model, cfg)
    tokens = _drive(_Driver(sub, cfg, ["a"]), torch.arange(5, dtype=torch.long) + 3,
                    _fwd_info("a", 10, True))
    assert tokens.numel() > 0
    # per forward: each layer's second norm, each later layer's first, and the final one
    assert group.fused == 2 * cfg.num_hidden_layers * len(seen) > 0
    # the vocab-parallel embedding is the one reduction outside a norm
    assert len(group.reduced) == len(seen)
