"""DeepSeek-V3's MoE block: a noaux_tc router, routed experts and an ungated shared expert.

GLM-5.x, Kimi K2 and DeepSeek V3.x share this block shape. ``config.quant`` (a
``QuantMethod``, or None for a bf16 checkpoint) holds the routed experts and picks their
kernels. A fused fp8 block runs batches of up to ``DECODE_MAX_TOKENS`` tokens as one launch
chain: router, routed experts and shared expert.

Under TP both halves return this rank's partial, and one all-reduce sums them. With
``reduce_results=False`` the block skips it, and the caller folds the reduction into its next
residual add + RMSNorm.
"""
from __future__ import annotations

import logging
from functools import partial

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide
from mstar.model.components.distributed import ParallelGatedMLP
from mstar.model.components.moe import (
    NoAuxTcRouter,
    _dispatch,
    _down_proj_weight_loader,
    _gate_up_weight_loader,
    dispatch_experts_fused,
)
from mstar.model.components.quantization import DECODE_MAX_TOKENS, Fp8BlockExperts

logger = logging.getLogger(__name__)

_BACKEND_LOGGED = False


class Bf16Experts(nn.Module):
    """Routed experts of an unquantized checkpoint, sharded on the intermediate dim."""

    def __init__(self, num_experts: int, intermediate_size: int, hidden_size: int,
                 comm_group: CommGroup | None = None) -> None:
        super().__init__()
        comm_group = comm_group or CommGroup.trivial()
        self.tp_rank, self.tp_size = comm_group.rank, comm_group.world_size
        self.num_experts, self.intermediate_size = num_experts, intermediate_size
        shard_inter = divide(intermediate_size, self.tp_size)
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * shard_inter, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, shard_inter))
        self._attach_weight_loaders()

    def _attach_weight_loaders(self) -> None:
        rank, size, inter = self.tp_rank, self.tp_size, self.intermediate_size
        self.gate_up_proj.weight_loader = partial(_gate_up_weight_loader, rank, size, inter)
        self.down_proj.weight_loader = partial(_down_proj_weight_loader, rank, size, inter)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._attach_weight_loaders()
        return result

    def forward(self, x: torch.Tensor, topk_weights: torch.Tensor,
                topk_ids: torch.Tensor) -> torch.Tensor:
        dispatch = _dispatch if self.tp_size == 1 else dispatch_experts_fused
        return dispatch(x, self.gate_up_proj, self.down_proj, self.num_experts, topk_ids,
                        topk_weights)


class SharedExpertMoE(nn.Module):
    """Routed experts + an ungated shared expert behind a noaux_tc router."""

    def __init__(self, config, comm_group: CommGroup | None = None,
                 reduce_results: bool = True) -> None:
        super().__init__()
        comm_group = comm_group or CommGroup.trivial()
        self.comm_group = comm_group
        self.tp_size = comm_group.world_size
        self.reduce_results = reduce_results
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts
        quant = getattr(config, "quant", None)
        self.quant = quant

        self.gate = NoAuxTcRouter(
            hidden_size=config.hidden_size,
            n_routed_experts=config.n_routed_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            routed_scaling_factor=config.routed_scaling_factor,
            norm_topk_prob=config.norm_topk_prob,
        )
        if quant is not None and quant.keeps_quantized("routed_experts"):
            self.experts = quant.routed_experts(
                config.n_routed_experts, config.moe_intermediate_size, config.hidden_size,
                comm_group)
        else:
            self.experts = Bf16Experts(
                config.n_routed_experts, config.moe_intermediate_size, config.hidden_size,
                comm_group)
        shared_inter = config.moe_intermediate_size * config.n_shared_experts
        if quant is not None and quant.keeps_quantized("shared_expert"):
            self.shared_expert = quant.gated_mlp(
                config.hidden_size, shared_inter, comm_group, reduce_results=False)
        else:
            self.shared_expert = ParallelGatedMLP(
                hidden_size=config.hidden_size, intermediate_size=shared_inter,
                comm_group=comm_group, activation=config.hidden_act, bias=False,
                reduce_results=False)

    @property
    def fused(self) -> bool:
        """The routed experts resolved to the capturable fused kernels."""
        return bool(getattr(self.experts, "fused", False))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape
        flat = hidden_states.view(-1, self.hidden_size).contiguous()
        if self._takes_decode_chain(flat):
            out = self._forward_decode(flat)
        else:
            # topk_weights stay fp32 into the combine, as in HF's experts.
            topk_weights, topk_ids, _ = self.gate(flat)
            out = self.experts(flat, topk_weights, topk_ids) + self.shared_expert(flat)
        if self.reduce_results and self.tp_size > 1:
            out = self.comm_group.all_reduce(out)
        return out.view(input_shape)

    def _takes_decode_chain(self, flat: torch.Tensor) -> bool:
        """A small fused fp8 batch in bf16, with the shared expert in the routed experts'
        dtype and shard, takes the decode kernels' one launch chain."""
        if not (isinstance(self.experts, Fp8BlockExperts) and self.experts.fused
                and flat.shape[0] <= DECODE_MAX_TOKENS and flat.is_cuda):
            return False
        shared = self.shared_expert.down_proj.weight
        return (shared.dtype == flat.dtype == torch.bfloat16
                and shared.shape[1] == self.experts.down_proj_fp8.shape[2])

    # Triton launches with PDL: dynamo stays out, the kernels run inside the captured graph.
    @torch.compiler.disable
    def _forward_decode(self, flat: torch.Tensor) -> torch.Tensor:
        """Router, routed and shared experts as one decode launch chain; this rank's partial."""
        from mstar.utils.fused_moe import decode as moe_decode

        gate, exp, shared = self.gate, self.experts, self.shared_expert
        return moe_decode.forward(
            flat, gate.weight, gate.e_score_correction_bias,
            exp.gate_up_proj_fp8, exp.gate_up_proj_scale_inv,
            exp.down_proj_fp8, exp.down_proj_scale_inv,
            shared.gate_up_proj.weight, shared.down_proj.weight,
            top_k=gate.top_k, scale=gate.routed_scaling_factor, normalize=gate.norm_topk_prob,
            block_size=exp.block_size)

    def process_weights_after_loading(self, device) -> None:
        """Resolve the experts' kernels on the real device, then the router's to match."""
        self.gate.finalize_weights()
        if isinstance(self.experts, Fp8BlockExperts):
            self.experts.resolve_kernels(device)
            if self.experts.fused:
                from mstar.utils.fused_moe import decode as moe_decode  # registers the op

                # sigmoid, bias, top-k and normalize in one launch; ids can differ at exact ties
                self.gate._topk_op = moe_decode.router_topk
        global _BACKEND_LOGGED
        if not _BACKEND_LOGGED and self.quant is not None:
            logger.info(
                "MoE routed experts: %s (quant_kernels=%s, tp_size=%d)",
                "fused fp8 kernels" if self.fused else "fp8-resident reference dispatch",
                self.quant.kernels, self.tp_size)
            _BACKEND_LOGGED = True
