"""Assembled GLM-5.2 text backbone."""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.glm52.components.decoder_layer import Glm52DecoderLayer
from mstar.model.glm52.components.language_model import (
    build_embedding,
    build_lm_head,
    build_rmsnorm,
)
from mstar.model.glm52.config import Glm52ModelConfig


class Glm52LanguageModel(nn.Module):
    def __init__(
        self, config: Glm52ModelConfig, comm_group: CommGroup | None = None
    ) -> None:
        super().__init__()
        self.comm_group = comm_group or CommGroup.trivial()
        self.embed_tokens = build_embedding(config, comm_group=comm_group)
        self.layers = nn.ModuleList(
            [
                Glm52DecoderLayer(config, layer_idx, comm_group=comm_group)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = build_rmsnorm(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Layer order IS the IndexShare order under dsa_long_context: each FULL
        layer's pick is the one the SHARED layers after it reuse, until the next
        FULL layer (the DSA resource keeps it for the forward). Each layer hands the next its
        (FFN partial, residual); the final norm reduces the last one."""
        hidden_states, residual = self.embed_tokens(input_ids), None
        for decoder_layer in self.layers:
            hidden_states, residual = decoder_layer(hidden_states, position_ids, residual)
        normed, _ = self.norm.forward_residual(hidden_states, residual, self.comm_group)
        return normed


class Glm52ForCausalLM(nn.Module):
    def __init__(
        self, config: Glm52ModelConfig, comm_group: CommGroup | None = None
    ) -> None:
        super().__init__()
        self.config = config
        self.model = Glm52LanguageModel(config, comm_group=comm_group)
        self.lm_head = build_lm_head(config, comm_group=comm_group)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, position_ids)
        return self.lm_head(hidden_states)

    def load_weights(self, weights, **kwargs) -> set[str]:
        from mstar.model.glm52.weight_loader import load_glm52_hf_weights

        # The indexer exists only on the DSA engine path (Glm52MLAAttention
        # builds none flag-off), so its keys are skipped before the dequant
        # stream rather than dequantized and dropped as unmatched.
        load_indexer = self.config.dsa_long_context
        loaded = load_glm52_hf_weights(
            self, weights, self.config.n_routed_experts,
            quant=self.config.quant,
            num_hidden_layers=self.config.num_hidden_layers,
            load_indexer=load_indexer,
        )
        if load_indexer:
            missing = {
                name for name, _ in self.named_parameters() if ".self_attn.indexer." in name
            } - loaded
            if missing:
                raise RuntimeError(
                    f"dsa_long_context is on but {len(missing)} indexer "
                    f"parameters received no checkpoint weights (e.g. "
                    f"{sorted(missing)[:3]}) — the weight stream is missing "
                    "the FULL-layer indexer keys. Selecting from uninitialized "
                    "memory is silent garbage; refuse to serve."
                )
        missing = sorted(set(dict(self.named_parameters())) - loaded)
        if missing:
            raise RuntimeError(
                f"{len(missing)} parameters received no checkpoint weights "
                f"and would serve uninitialized memory: {missing}"
            )
        return loaded
