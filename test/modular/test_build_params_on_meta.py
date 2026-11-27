"""``MultimodalDecoderModel._build(params_on_meta=True)``: parameters made on meta (no random
init) and given uninitialized storage on the device, for a model whose loader fills every one.

A Qwen3.5 stack built that way and loaded from a checkpoint holds the same bytes as one built
the default way, ties included, and its buffers are the CPU-built ones."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import nn

pytest.importorskip("safetensors")
import lora_module_helpers as H  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from mstar.model.multimodal_decoder import MultimodalDecoderModel  # noqa: E402

build = MultimodalDecoderModel._build


class _Toy(nn.Module):
    """A tied head, a parameter tagged before and one tagged after it is registered, a
    CPU-computed buffer, and an init that records the device it ran on."""

    inits: list[str] = []

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.head = nn.Linear(4, 8, bias=False)
        self.head.weight = self.embed.weight
        scale = nn.Parameter(torch.empty(4))
        scale.weight_loader = "before"
        self.register_parameter("scale", scale)
        self.scale.output_dim = 0
        self.register_buffer("table", torch.arange(4.0).cos(), persistent=False)
        nn.init.constant_(self.scale, 3.0)
        type(self).inits.append(self.scale.device.type)


@pytest.fixture(autouse=True)
def _clear_inits():
    _Toy.inits = []


def test_parameters_skip_their_init_and_land_on_the_device_in_the_build_dtype():
    model = build(_Toy, torch.bfloat16, "cpu", params_on_meta=True)
    assert _Toy.inits == ["meta"]
    for name, p in model.named_parameters():
        assert p.device.type == "cpu" and p.dtype == torch.bfloat16, name
        assert isinstance(p, nn.Parameter), name
    assert torch.get_default_dtype() == torch.float32


def test_ties_attributes_and_buffers_survive():
    model = build(_Toy, torch.bfloat16, "cpu", params_on_meta=True)
    assert model.head.weight is model.embed.weight
    assert model.scale.weight_loader == "before" and model.scale.output_dim == 0
    plain = build(_Toy, torch.bfloat16, "cpu")
    assert torch.equal(model.table, plain.table) and model.table.dtype == plain.table.dtype
    assert [n for n, _ in model.named_parameters()] == [n for n, _ in plain.named_parameters()]


def test_the_default_build_still_initializes_and_leaves_register_parameter_alone():
    register = nn.Module.register_parameter
    plain = build(_Toy, torch.bfloat16, "cpu")
    assert _Toy.inits == ["cpu"] and torch.equal(plain.scale, torch.full((4,), 3.0, dtype=torch.bfloat16))
    assert nn.Module.register_parameter is register

    def fails():
        _Toy()
        raise RuntimeError("make failed")

    with pytest.raises(RuntimeError, match="make failed"):
        build(fails, torch.bfloat16, "cpu", params_on_meta=True)
    assert nn.Module.register_parameter is register
    assert torch.get_default_dtype() == torch.float32
    # a module built afterwards gets ordinary parameters again
    assert nn.Linear(2, 2).weight.device.type == "cpu"


@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
def test_a_qwen3_5_stack_built_on_meta_loads_the_same_bytes(tmp_path, tied):
    from mstar.model.qwen3_5.components.language_model import Qwen3_5ForCausalLM
    from mstar.model.qwen3_5.weight_loader import load_qwen3_5_weights

    config = dataclasses.replace(H.tiny_config(), tie_word_embeddings=tied)
    weights = H.hf_text_weights()
    if tied:  # a tied checkpoint carries no head
        weights.pop("lm_head.weight")
    save_file({k: v.contiguous() for k, v in weights.items()}, str(tmp_path / "model.safetensors"))
    built = {}
    for on_meta in (False, True):
        model = build(lambda: Qwen3_5ForCausalLM(config, None), H.DTYPE, "cpu", params_on_meta=on_meta)
        load_qwen3_5_weights(model, tmp_path, device="cpu")
        built[on_meta] = model
    plain, meta = built[False], built[True]
    assert meta.lm_head.weight is meta.model.embed_tokens.weight if tied else True
    plain_state, meta_state = plain.state_dict(keep_vars=True), meta.state_dict(keep_vars=True)
    assert plain_state.keys() == meta_state.keys()
    for name, value in plain_state.items():
        assert value.dtype == meta_state[name].dtype, name
        assert torch.equal(value, meta_state[name]), name
        assert type(value) is type(meta_state[name]), name
    plain_attrs = {n: sorted(vars(p)) for n, p in plain.named_parameters()}
    assert plain_attrs == {n: sorted(vars(p)) for n, p in meta.named_parameters()}
    plain_buffers = dict(plain.named_buffers())
    for name, buf in meta.named_buffers():
        assert torch.equal(buf, plain_buffers[name]), name


def test_a_qwen3_5_vision_tower_built_on_meta_keeps_its_buffers_and_attributes():
    from mstar.model.qwen3_5.components.vision import Qwen3_5VisionModel
    from mstar.model.qwen3_5.config import Qwen3_5VisionConfig

    config = Qwen3_5VisionConfig(
        depth=2,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=16,
        out_hidden_size=H.HIDDEN,
    )
    plain = build(lambda: Qwen3_5VisionModel(config), torch.bfloat16, "cpu")
    meta = build(lambda: Qwen3_5VisionModel(config), torch.bfloat16, "cpu", params_on_meta=True)
    shapes = {n: (p.shape, p.dtype, sorted(vars(p))) for n, p in plain.named_parameters()}
    assert shapes == {n: (p.shape, p.dtype, sorted(vars(p))) for n, p in meta.named_parameters()}
    assert all(p.device.type == "cpu" for p in meta.parameters())
    plain_buffers = dict(plain.named_buffers())
    assert plain_buffers.keys() == dict(meta.named_buffers()).keys()
    for name, buf in meta.named_buffers():
        assert torch.equal(buf, plain_buffers[name]), name
