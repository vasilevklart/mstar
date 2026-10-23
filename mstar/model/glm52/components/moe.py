"""GLM-5.2's MoE block is DeepSeek-V3's: ``components/shared_expert_moe.py``."""
from mstar.model.components.shared_expert_moe import SharedExpertMoE as Glm52SparseMoeBlock

__all__ = ["Glm52SparseMoeBlock"]
