"""How a quantized checkpoint's weights are held and multiplied: one method per format.

The checkpoint's ``quantization_config`` picks the method (`quant_method_for`). Layers ask it
for their linears and routed experts instead of reading per-kernel flags, so a model built on
it serves every format a method exists for, with each method choosing its own kernels.

``kernels`` is the one switch, the same for every model:

* ``auto``: the fastest path this device runs, falling back (loudly) to ``reference``.
* ``fused``: the fused kernels or a load-time error, for a deployment that must capture.
* ``reference``: each weight dequantized per call and multiplied in the activation dtype, the
  numerics of a bf16 copy. Eager only; the anchor parity tests compare against.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.fp8_linear import (
    COLUMN,
    REPLICATED,
    ROW,
    Fp8Linear,
    Fp8ParallelGatedMLP,
)
from mstar.utils.fp8_block import (
    Fp8BlockQuantConfig,
    dequant_fp8_block_stream,
    dequantize_fp8_block_weight,
)

logger = logging.getLogger(__name__)

QUANT_KERNELS = ("auto", "fused", "reference")


class QuantMethod:
    """A quantized checkpoint's weights in a model. Subclass per format (fp8 block-scale here;
    INT4 or NVFP4 would be siblings), each with its own layouts and kernels."""

    kernels: str = "auto"

    # Which of a checkpoint's weights stay quantized: "linear" (attention projections and dense
    # MLPs), "routed_experts", "shared_expert". The rest load dequantized to the model's dtype.
    def keeps_quantized(self, role: str) -> bool:
        raise NotImplementedError

    def checkpoint_stream(
        self, weights: Iterable[tuple[str, torch.Tensor]], keep: Callable[[str], bool],
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """The checkpoint's tensors, each weight whose name (without ``.weight``) ``keep``
        rejects dequantized to bf16 and the rest passed through quantized."""
        raise NotImplementedError

    def linear(
        self, in_features: int, out_features: int, comm_group: CommGroup | None = None,
        shard: int | None = REPLICATED, merged: int = 1, reduce_results: bool = False,
    ) -> nn.Module:
        """A linear over this rank's shard of a quantized weight (``shard``: REPLICATED,
        COLUMN or ROW; ``merged`` stacks that many column shards, as gate and up)."""
        raise NotImplementedError

    def gated_mlp(
        self, hidden_size: int, intermediate_size: int, comm_group: CommGroup | None = None,
        reduce_results: bool = True,
    ) -> nn.Module:
        """A SwiGLU MLP over quantized gate/up and down weights."""
        raise NotImplementedError

    def routed_experts(
        self, num_experts: int, intermediate_size: int, hidden_size: int,
        comm_group: CommGroup | None = None,
    ) -> nn.Module:
        """The routed experts' weights, this rank's intermediate shard; the module's
        ``forward(x, topk_weights, topk_ids)`` returns this rank's unreduced partial."""
        raise NotImplementedError


@dataclass(frozen=True)
class Fp8BlockQuant(QuantMethod):
    """e4m3 weights with one fp32 scale per block (DeepSeek-V3's format: GLM-5.x, Kimi K2,
    DeepSeek V3.x). Linears and routed experts stay fp8; the shared expert loads as bf16,
    because the small-batch MoE kernels fuse a bf16 shared expert into the routed launch."""

    config: Fp8BlockQuantConfig
    kernels: str = "auto"

    def __post_init__(self) -> None:
        if self.kernels not in QUANT_KERNELS:
            raise ValueError(f"quant_kernels={self.kernels!r} is not one of {QUANT_KERNELS}")

    @property
    def block_size(self) -> tuple[int, int]:
        return tuple(self.config.weight_block_size)

    @property
    def reference(self) -> bool:
        return self.kernels == "reference"

    def keeps_quantized(self, role: str) -> bool:
        return role in ("linear", "routed_experts")

    def checkpoint_stream(self, weights, keep):
        return dequant_fp8_block_stream(weights, self.config, keep_fp8=keep)

    def linear(self, in_features, out_features, comm_group=None, shard=REPLICATED, merged=1,
               reduce_results=False):
        return Fp8Linear(in_features, out_features, self.block_size, comm_group, shard=shard,
                         merged=merged, reduce_results=reduce_results, reference=self.reference)

    def gated_mlp(self, hidden_size, intermediate_size, comm_group=None, reduce_results=True):
        return Fp8ParallelGatedMLP(hidden_size, intermediate_size, self.block_size, comm_group,
                                   reduce_results=reduce_results, reference=self.reference)

    def routed_experts(self, num_experts, intermediate_size, hidden_size, comm_group=None):
        return Fp8BlockExperts(num_experts, intermediate_size, hidden_size, self, comm_group)

    def fused_available(self, device: torch.device) -> bool:
        """Whether the fused MoE kernels serve here: CUDA, an importable kernel, and the
        128 x 128 scale block they compile for."""
        if self.reference or device.type != "cuda" or self.block_size != (128, 128):
            return False
        try:
            from mstar.utils.fused_moe import fused_experts_fp8  # noqa: F401
        except Exception as exc:  # noqa: BLE001 - 'auto' serves the reference loop, loudly
            logger.warning("fused fp8 MoE kernel unavailable (%r): the reference dispatch "
                           "serves eager, without CUDA graphs", exc)
            return False
        return True


def quant_method_for(
    quantization_config: Fp8BlockQuantConfig | dict | None, kernels: str = "auto",
) -> QuantMethod | None:
    """The method for a checkpoint's ``quantization_config`` (parsed, or HF's dict); None for
    an unquantized checkpoint. A format no method serves is refused rather than misread."""
    if quantization_config is None:
        return None
    if isinstance(quantization_config, dict):
        parsed = Fp8BlockQuantConfig.from_hf_config_dict(quantization_config)
        if parsed is None:
            raise ValueError(
                f"quantization_config {quantization_config.get('quant_method')!r} has no "
                "quantization method; fp8 block-scale is served")
        quantization_config = parsed
    return Fp8BlockQuant(quantization_config, kernels)


# Up to this many tokens a fused MoE block takes the decode kernels' one launch chain.
DECODE_MAX_TOKENS = 64


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


class Fp8BlockExperts(nn.Module):
    """Routed experts kept fp8 (uint8 containers, so ``module.to(bf16)`` leaves them alone)
    with fp32 block scales, sharded on the intermediate dim.

    The owning block resolves the kernels on the real device (``resolve_kernels``): the grouped
    runner up to ``DECODE_MAX_TOKENS`` tokens (where the block usually takes the decode chain
    instead), the wide-tile prefill kernels above (the runner's bits), and the per-expert
    reference loop where neither runs.
    """

    def __init__(
        self, num_experts: int, intermediate_size: int, hidden_size: int,
        quant: Fp8BlockQuant, comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        comm_group = comm_group or CommGroup.trivial()
        self.tp_rank, self.tp_size = comm_group.rank, comm_group.world_size
        self.quant = quant
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.block_size = bo, bi = quant.block_size
        shard_inter = intermediate_size // self.tp_size
        assert shard_inter * self.tp_size == intermediate_size
        assert shard_inter % bo == 0 and shard_inter % bi == 0, (
            f"per-rank intermediate {shard_inter} must be a whole number of the fp8 scale "
            f"block {self.block_size} for clean TP slicing")
        E = num_experts
        self.gate_up_proj_fp8 = nn.Parameter(
            torch.empty(E, 2 * shard_inter, hidden_size, dtype=torch.uint8), requires_grad=False)
        self.gate_up_proj_scale_inv = nn.Parameter(
            torch.empty(E, 2 * (shard_inter // bo), _ceil_div(hidden_size, bi), dtype=torch.float32),
            requires_grad=False)
        self.down_proj_fp8 = nn.Parameter(
            torch.empty(E, hidden_size, shard_inter, dtype=torch.uint8), requires_grad=False)
        self.down_proj_scale_inv = nn.Parameter(
            torch.empty(E, _ceil_div(hidden_size, bo), shard_inter // bi, dtype=torch.float32),
            requires_grad=False)
        # resolved by resolve_kernels; host tests stay on the reference loop
        self.fused = False
        self._attach_weight_loaders()

    def _attach_weight_loaders(self) -> None:
        from mstar.model.components.moe import (
            _down_proj_fp8_weight_loader,
            _gate_up_fp8_weight_loader,
        )

        bo, bi = self.block_size
        rank, size, inter = self.tp_rank, self.tp_size, self.intermediate_size
        self.gate_up_proj_fp8.weight_loader = partial(_gate_up_fp8_weight_loader, rank, size, inter, 1)
        self.gate_up_proj_scale_inv.weight_loader = partial(
            _gate_up_fp8_weight_loader, rank, size, inter, bo)
        self.down_proj_fp8.weight_loader = partial(_down_proj_fp8_weight_loader, rank, size, inter, 1)
        self.down_proj_scale_inv.weight_loader = partial(
            _down_proj_fp8_weight_loader, rank, size, inter, bi)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._attach_weight_loaders()
        return result

    def resolve_kernels(self, device) -> None:
        """Pick the kernels on the real device: ``fused`` must not silently downgrade."""
        dev = torch.device(device) if device is not None else torch.device("cpu")
        ok = self.quant.fused_available(dev)
        if self.quant.kernels == "fused" and not ok:
            raise RuntimeError(
                "quant_kernels='fused' but the fused fp8 MoE kernels can't serve here (they need "
                f"CUDA, triton and 128 x 128 scale blocks; these are {self.block_size}). Use "
                "'auto' to fall back to the reference dispatch.")
        self.fused = ok

    def forward(self, x: torch.Tensor, topk_weights: torch.Tensor,
                topk_ids: torch.Tensor) -> torch.Tensor:
        """This rank's routed partial, unreduced; routing weights stay fp32 into the combine."""
        if not self.fused:
            return self.reference(x, topk_weights, topk_ids)
        if x.shape[0] > DECODE_MAX_TOKENS:
            return self._prefill(x, topk_weights, topk_ids)
        from mstar.utils.fused_moe import fused_experts_fp8

        return fused_experts_fp8(
            x, self.gate_up_proj_fp8, self.down_proj_fp8, self.gate_up_proj_scale_inv,
            self.down_proj_scale_inv, topk_weights, topk_ids, block_size=self.block_size)

    # Triton launches stay outside dynamo, as the runner's fp8 quant does.
    @torch.compiler.disable
    def _prefill(self, x, topk_weights, topk_ids):
        """The grouped runner's output from the wide-tile prefill kernels (same bits)."""
        from mstar.utils.fused_moe import prefill as moe_prefill

        return moe_prefill.experts(
            x, self.gate_up_proj_fp8, self.down_proj_fp8, self.gate_up_proj_scale_inv,
            self.down_proj_scale_inv, topk_weights, topk_ids, block_size=self.block_size,
            swiglu_limit=None)

    def reference(self, x: torch.Tensor, topk_weights: torch.Tensor,
                  topk_ids: torch.Tensor) -> torch.Tensor:
        """Per-expert loop that dequantizes only the experts this batch hit (host syncs:
        eager only)."""
        final = torch.zeros_like(x)
        with torch.no_grad():
            expert_mask = F.one_hot(topk_ids, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx_t in expert_hit:
            e = expert_idx_t[0]
            top_k_pos, token_idx = torch.where(expert_mask[e])
            tokens = x[token_idx]
            gate_up_w = dequantize_fp8_block_weight(
                self.gate_up_proj_fp8[e], self.gate_up_proj_scale_inv[e],
                block_size=self.block_size, out_dtype=x.dtype)
            down_w = dequantize_fp8_block_weight(
                self.down_proj_fp8[e], self.down_proj_scale_inv[e],
                block_size=self.block_size, out_dtype=x.dtype)
            gate, up = torch.mm(tokens, gate_up_w.T).chunk(2, dim=-1)
            out = torch.mm(F.silu(gate) * up, down_w.T)
            out = out * topk_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, out.to(final.dtype))
        return final


__all__ = [
    "COLUMN",
    "DECODE_MAX_TOKENS",
    "QUANT_KERNELS",
    "REPLICATED",
    "ROW",
    "Fp8BlockExperts",
    "Fp8BlockQuant",
    "QuantMethod",
    "quant_method_for",
]
