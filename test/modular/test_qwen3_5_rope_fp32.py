"""Qwen3.5's rope frequencies stay fp32 through the engine's cast to the model's dtype.

In bf16 a frequency is off by up to 0.3 %, and the angle error grows with the distance
between two tokens (or patches), so long-range attention reads the wrong keys. Upstream
keeps both buffers in fp32, on every platform.
"""

import torch

from mstar.model.components.mrope import compute_inv_freq
from mstar.model.qwen3_5.components.language_model import Qwen3_5LanguageModel
from mstar.model.qwen3_5.components.vision import Qwen3_5VisionModel
from mstar.model.qwen3_5.config import FULL_ATTENTION, Qwen3_5Config, Qwen3_5VisionConfig


def _tiny_language_model() -> Qwen3_5LanguageModel:
    config = Qwen3_5Config(
        num_hidden_layers=1, hidden_size=64, intermediate_size=128, layer_types=[FULL_ATTENTION],
        num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, vocab_size=16, rms_norm_eps=1e-6, max_position_embeddings=64,
        rope_theta=10_000_000.0, mrope_section=[2, 1, 1], mrope_interleaved=True, partial_rotary_factor=0.25,
    )
    return Qwen3_5LanguageModel(config)


def _tiny_vision_model() -> Qwen3_5VisionModel:
    config = Qwen3_5VisionConfig(
        depth=1, hidden_size=64, intermediate_size=128, num_heads=2, in_channels=3, patch_size=2,
        temporal_patch_size=1, spatial_merge_size=1, num_position_embeddings=4, out_hidden_size=64,
    )
    return Qwen3_5VisionModel(config)


def test_rope_frequencies_survive_the_dtype_cast():
    lm = _tiny_language_model().to(dtype=torch.bfloat16)
    want = compute_inv_freq(lm.config.rotary_dim, lm.config.rope_theta)
    assert lm.inv_freq.dtype is torch.float32
    assert torch.equal(lm.inv_freq, want)
    # far along a long prompt a rounded frequency lands on another angle
    position = 20_000.0
    cos, _ = lm.build_cos_sin(torch.full((3, 1), position), torch.float32)
    assert torch.equal(cos[0, 1], (want[1] * position).cos())


def test_vision_rope_frequencies_survive_the_dtype_cast():
    rope = _tiny_vision_model().to(dtype=torch.bfloat16).rotary_pos_emb
    dim = 16  # half the head
    want = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    assert rope.inv_freq.dtype is torch.float32
    assert torch.equal(rope.inv_freq, want)
    cos, _ = rope(torch.tensor([[0, 255]]))
    assert torch.equal(cos[0, dim // 2:dim], (255.0 * want).cos())
