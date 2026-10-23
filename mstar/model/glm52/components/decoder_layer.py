"""GLM-5.2 decoder layer with MLA position ids threaded through attention."""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.glm52.components.attention import Glm52MLAAttention
from mstar.model.glm52.components.language_model import (
    build_mlp_for_layer,
    build_rmsnorm,
)
from mstar.model.glm52.config import Glm52ModelConfig


class Glm52DecoderLayer(nn.Module):
    """Attention and FFN return this rank's partials, and each reduction runs fused into the
    next residual add + RMSNorm (``RMSNorm.forward_residual``, as Qwen3.5's layers do)."""

    def __init__(
        self,
        config: Glm52ModelConfig,
        layer_idx: int,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.comm_group = comm_group or CommGroup.trivial()
        self.self_attn = Glm52MLAAttention(
            config, comm_group=comm_group, layer_idx=layer_idx, reduce_results=False)
        self.mlp = build_mlp_for_layer(config, layer_idx, comm_group=comm_group, reduce_results=False)
        self.input_layernorm = build_rmsnorm(config)
        self.post_attention_layernorm = build_rmsnorm(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Takes the previous layer's ``(FFN partial, residual)`` (``residual`` None:
        ``hidden_states`` is the embedding) and returns this layer's."""
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm.forward_residual(
                hidden_states, residual, self.comm_group)
        hidden_states = self.self_attn(hidden_states, position_ids)
        hidden_states, residual = self.post_attention_layernorm.forward_residual(
            hidden_states, residual, self.comm_group)
        return self.mlp(hidden_states), residual
