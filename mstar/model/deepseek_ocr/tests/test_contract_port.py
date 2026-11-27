"""DeepSeek-OCR on the state contract builds the specs, steps and layouts it wrote by hand, field for field, and its
forward makes the same calls on them.

The engine sees the same objects whether a model declares its state or writes its specs, so the port changes nothing
it runs. The hand-written specs, steps and calls below are the reference, copied from the model as it stood before the
port (``get_node_resources``, ``layer_steps`` and the decoder's forward at b1266bfe9), so the adapter can't pass by
agreeing with itself.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

# the resources import Triton kernels (a torch dependency on Linux)
pytest.importorskip("triton")

from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttentionStep  # noqa: E402
from mstar.engine.resources.kv.config import KVSpec, KVStep, PagedKVConfig  # noqa: E402
from mstar.engine.resources.position.config import PositionConfig, PositionSpec, PositionStep  # noqa: E402
from mstar.engine.resources.sampler.config import SamplerSpec, SamplerStep  # noqa: E402
from mstar.engine.resources.spec import apply_yaml_overrides  # noqa: E402
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep  # noqa: E402
from mstar.engine.state import SampleSettings  # noqa: E402
from mstar.engine.state.adapters import (  # noqa: E402
    StateSetup,
    kv_layouts,
    node_handles,
    node_resources,
    sampling_configs,
)
from mstar.engine.state.analysis import state_declaration_problems  # noqa: E402
from mstar.model.base import enabled_resources, serving_resources  # noqa: E402
from mstar.model.components.sampled_token_decoder import MEDIA_EMBEDS, MEDIA_MASK  # noqa: E402
from mstar.model.deepseek_ocr.config import ATTN  # noqa: E402
from mstar.model.keys import KV_CACHE, MIXED, PREFILL_TEXT, ROPE, SAMPLER  # noqa: E402
from mstar.model.multimodal_decoder import MIXED_SPLIT, MultimodalDecoderModel  # noqa: E402
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine, engine_step  # noqa: E402

# DeepSeek-OCR's config.json, as the package's other tests read it
FIXTURE = Path(__file__).parent / "fixtures" / "config.json"
PUBLISHED = Path(__file__).parents[4] / "configs" / "deepseek_ocr.yaml"
PREFILL_VISION = "prefill_vision"
# A decoder a CPU can run: the checkpoint's 12 layers at a small width, each one dense, since the MoE's kernels need a
# GPU.
SMALL = {
    "hidden_size": 16, "intermediate_size": 24, "num_attention_heads": 2, "num_key_value_heads": 2,
    "first_k_dense_replace": 12,
}
# What a deployment with no `resources:` block of its own could set.
OVERRIDES = {"resources": {
    KV_CACHE: {"max_num_pages": 512, "cpu_offload_pages": 64}, ATTN: {"flashinfer_backend": "fa2"},
    SAMPLER: {"enable_logprobs": True},
}}


class _CharTokenizer:
    """One id per character, enough for the prompt's encode; ``<image>`` is 7."""

    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        ids = []
        for i, piece in enumerate(text.split("<image>")):
            ids += [7] * (i > 0) + [ord(c) % 1000 + 10 for c in piece]
        return ids

    def convert_tokens_to_ids(self, token):
        return 7


@pytest.fixture(autouse=True)
def _no_tokenizer_files(monkeypatch):
    import transformers

    monkeypatch.setattr(transformers.PreTrainedTokenizerFast, "from_pretrained",
                        classmethod(lambda cls, path: _CharTokenizer()))


def _model(tmp_path, generation: dict | None = None, language: dict | None = None, **kwargs):
    from mstar.model.deepseek_ocr.deepseek_ocr_model import DeepseekOCRModel

    config = json.loads(FIXTURE.read_text())
    config["language_config"].update(language or {})
    (tmp_path / "config.json").write_text(json.dumps(config))
    if generation is not None:
        (tmp_path / "generation_config.json").write_text(json.dumps(generation))
    return DeepseekOCRModel(str(tmp_path), **kwargs)


def _reference(model) -> list:
    """The specs DeepSeek-OCR's hand-written ``get_node_resources`` returned."""
    config = model.config
    return [
        KVSpec(resource_key=KV_CACHE, nodes={"LLM"}, config=PagedKVConfig(
            num_layers=config.num_hidden_layers, num_qo_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads, head_dim=config.head_dim,
            max_seq_len=model.max_seq_len,
        )),
        AttentionSpec(resource_key=ATTN, nodes={"LLM"}, config=AttentionConfig(
            kv_cache=KV_CACHE, splits=(MIXED_SPLIT,),
        )),
        PositionSpec(resource_key=ROPE, nodes={"LLM"}, config=PositionConfig(
            kv_cache=KV_CACHE, rotary_dim=config.head_dim, interleave=False,
            rope_theta=config.rope_theta, rope_table_dtype=torch.bfloat16,
        )),
        model._sampler_spec(config.vocab_size),
    ]


def _assert_equal_per_key(specs, reference) -> None:
    by_key = {spec.resource_key: spec for spec in specs}
    assert len(by_key) == len(specs), "two specs under one key"
    expected = {spec.resource_key: spec for spec in reference}
    assert by_key.keys() == expected.keys()
    for key, spec in expected.items():
        assert by_key[key] == spec, f"{key} differs from the hand-written spec"


@pytest.mark.parametrize("generation, kwargs", [
    (None, {}),
    # a deployment's shorter context sizes the pool, as the model's max_seq_len did
    (None, {"max_seq_len": 4096}),
    # min-p on where a fine-tune's generation_config defaults it
    ({"min_p": 0.05}, {}),
    (None, {"image_mode": "base"}),
], ids=["default", "max_seq_len", "min_p", "one_view_mode"])
def test_specs_equal_the_hand_written_ones(tmp_path, generation, kwargs):
    model = _model(tmp_path, generation, **kwargs)
    _assert_equal_per_key(enabled_resources(model, {}, StateSetup()), _reference(model))
    assert state_declaration_problems(model) == []


def test_specs_equal_the_hand_written_ones_after_a_deployments_overrides(tmp_path):
    model = _model(tmp_path)
    expected = _reference(model)
    apply_yaml_overrides(expected, OVERRIDES)
    _assert_equal_per_key(serving_resources(model, OVERRIDES), expected)


def test_specs_equal_the_hand_written_ones_on_the_published_config(tmp_path):
    deployment = yaml.safe_load(PUBLISHED.read_text())
    model = _model(tmp_path, **deployment.get("model_kwargs", {}))
    expected = _reference(model)
    apply_yaml_overrides(expected, deployment)
    _assert_equal_per_key(serving_resources(model, deployment), expected)


def test_the_sampler_is_the_scaffolds(tmp_path):
    """The declared sampler builds the scaffold's `_sampler_spec`, min-p on where the checkpoint defaults it."""
    model = _model(tmp_path, {"min_p": 0.05})
    (spec,) = [s for s in node_resources(model) if isinstance(s, SamplerSpec)]
    assert spec == model._sampler_spec(model.config.vocab_size) and spec.enable_min_p


def test_a_request_samples_through_the_contracts_settings(tmp_path):
    """The request's settings convert to the `SamplingReqConfig` the hand-written path returned."""
    model = _model(tmp_path)
    kwargs = {"temperature": 0.5, "top_p": 0.9, "repetition_penalty": 1.1}
    settings = model.get_request_resource_configs(["image", "text"], ["text"], kwargs)
    assert isinstance(settings[SAMPLER], SampleSettings), "the request would skip the sampler's name and draws checks"
    scaffold = MultimodalDecoderModel.get_request_resource_configs(model, ["image", "text"], ["text"], kwargs)
    assert sampling_configs(settings, model.declare_state()) == scaffold


class _StubResource:
    """Stands in for the built resource under ``key``, and logs each call a forward makes on it."""

    # as the paged attention's kernels, which read the pages an add wrote
    requires_kv_write = True

    def __init__(self, key: str, calls: list[tuple]):
        self.key, self.calls = key, calls

    def layer_view(self, i):
        return torch.empty(0)

    def opens_label(self, rid, label):
        return True

    def apply_qk(self, q, k, label=None, rope_dtype=None):
        self.calls.append((self.key, "apply_qk", label))
        return q, k

    def write_kv(self, k, v, layer_idx=None, label=None):
        self.calls.append((self.key, "write_kv", layer_idx, label))

    def run(self, q, label=None, kv_cache_layer=None, k=None, v=None, layer_idx=None):
        self.calls.append((self.key, "run", layer_idx, label))
        return q

    def select_last_hidden(self, x, label=None):
        self.calls.append((self.key, "select_last_hidden", label))
        return x[-1:]

    def loopback_tokens(self, rids):
        self.calls.append((self.key, "loopback_tokens", tuple(rids)))
        return torch.zeros(len(rids), dtype=torch.long)

    def sample(self, rids, logits=None):
        self.calls.append((self.key, "sample", tuple(rids)))
        return logits.argmax(-1)


def _node(model, language_model=None, calls: list[tuple] | None = None):
    """The LLM node bound as the engine binds it, over stand-ins that log to ``calls``."""
    from mstar.model.deepseek_ocr.submodules import DeepseekOCRLLMSubmodule

    node = DeepseekOCRLLMSubmodule(language_model, model.config, model.encoders,
                                   max_prompt_tokens=model.max_prompt_tokens())
    calls = [] if calls is None else calls
    resources = {spec.resource_key: _StubResource(spec.resource_key, calls) for spec in node_resources(model)}
    node.bind_node_resources(resources)
    node.bind_node_state(node_handles(model, "LLM", resources))
    return node


def _reference_step(rids, inputs, prefill_tokens, advance=None) -> SubmoduleStep:
    """The step the resource-path scaffold declared for DeepSeek-OCR: its ``layer_steps`` and ``node_step``."""
    return SubmoduleStep(
        segments=[Segment(rid, "main", inp.input_seq_len) for rid, inp in zip(rids, inputs, strict=True)],
        steps={
            KV_CACHE: KVStep(),
            ATTN: AttentionStep(causal=True),
            SAMPLER: SamplerStep(apply_penalty=True, prefill_tracked_tokens=prefill_tokens),
            ROPE: PositionStep(advance=advance),
        },
    )


@pytest.mark.parametrize("walk", [PREFILL_TEXT, PREFILL_VISION, "decode", MIXED])
def test_steps_equal_the_hand_written_ones(tmp_path, walk):
    rows = {
        PREFILL_TEXT: [(PREFILL_TEXT, 5), (PREFILL_TEXT, 3)],
        # a page's whole prompt as one row each, its pads included
        PREFILL_VISION: [(PREFILL_VISION, 913), (PREFILL_VISION, 276)],
        "decode": [("decode", 1), ("decode", 1)],
        MIXED: [("decode", 1), (PREFILL_TEXT, 5)],
    }[walk]
    rids = [1, 2]
    inputs = [
        ARNodeInputs(input_ids=torch.arange(n), input_seq_len=n, graph_walk=None if walk != MIXED else own)
        for own, n in rows
    ]
    tokens = {rid: inp.input_ids for rid, inp, (own, _) in zip(rids, inputs, rows, strict=True) if own != "decode"}
    step = engine_step(_node(_model(tmp_path)), graph_walk=walk, request_ids=rids, inputs=inputs)
    reference = _reference_step(rids, inputs, tokens)
    assert step.state, "a ported walk declares its state"
    assert step.steps.keys() == reference.steps.keys()
    assert step.segments == reference.segments and step.cg_key_info == reference.cg_key_info
    for key, expected in reference.steps.items():
        actual = step.steps[key]
        if isinstance(expected, SamplerStep):
            # dataclass equality can't compare a dict of tensors
            got, want = actual.prefill_tracked_tokens, expected.prefill_tracked_tokens
            assert got.keys() == want.keys() and all(torch.equal(got[rid], want[rid]) for rid in want)
            actual, expected = replace(actual, prefill_tracked_tokens={}), replace(expected, prefill_tracked_tokens={})
        assert actual == expected, f"{key} differs from the hand-written step"


def _reference_calls(walk: str, layers: int) -> list[tuple]:
    """The calls the hand-written node made on its resources in a step of one row: decode's loop-back read; per layer,
    `DeepseekOCRAttention`'s rotation, then the KV write and the attention under the layer cursor; a prefill's
    last-token select; the draw."""
    calls = [(SAMPLER, "loopback_tokens", (1,))] if walk == "decode" else []
    for layer in range(layers):
        calls += [(ROPE, "apply_qk", "main"), (KV_CACHE, "write_kv", layer, "main"), (ATTN, "run", layer, "main")]
    if walk != "decode":
        calls.append((ATTN, "select_last_hidden", "main"))
    return [*calls, (SAMPLER, "sample", (1,))]


@pytest.mark.parametrize("walk", [PREFILL_TEXT, PREFILL_VISION, "decode"])
def test_the_forward_makes_the_hand_written_ones_calls(tmp_path, walk):
    """The node's step as the engine runs it, through the handles of the model's own declaration: every layer rotates,
    adds and reads at its own layer of the array, in the order the hand-written forward did under its cursors. The
    layers pass no rotation parameter, where `DeepseekOCRAttention` passed them by hand: they are the spec's, which
    the tests above compare."""
    from mstar.model.deepseek_ocr.components.language_model import DeepseekOCRLanguageModel

    model = _model(tmp_path, language=SMALL)
    calls: list[tuple] = []
    node = _node(model, DeepseekOCRLanguageModel(model.config), calls)
    # an image prompt: the BOS, three pads that its encoder rows replace, and two text tokens
    ids = torch.tensor([0, *[model.image_token_id] * 3, 40, 41])
    media = {MEDIA_EMBEDS: torch.zeros(3, model.config.hidden_size), MEDIA_MASK: ids == model.image_token_id}
    row = {
        PREFILL_TEXT: ARNodeInputs(input_ids=torch.arange(6), input_seq_len=6),
        PREFILL_VISION: ARNodeInputs(input_ids=ids, input_seq_len=6, tensor_inputs=media),
        # its token loops back off the sampler
        "decode": ARNodeInputs(input_seq_len=1),
    }[walk]
    step = engine_step(node, graph_walk=walk, request_ids=[1], inputs=[row])
    step.set_ctx(StepContext(request_ids=(1,), graph_walk=walk, slot=0, capture=False))
    step.begin_calls()
    try:
        engine_inputs = ModelInputsFromEngine(request_ids=[1], per_request_info={}, step=step)
        node.forward(walk, engine_inputs, **node.preprocess(walk, engine_inputs, [row]))
        # the handles' own check: every declared layer added, and the sampler drew once
        step.check_calls()
    finally:
        step.end_calls()
    assert calls == _reference_calls(walk, model.config.num_hidden_layers)


@pytest.mark.parametrize("modalities, images", [
    (["text"], []),
    # a page that tiles 2x3 and a small image (its global view only)
    (["image", "image", "text"], [(2240, 1596), (300, 400)]),
], ids=["text", "images"])
def test_the_prompt_layout_converts_to_the_hand_written_one(tmp_path, modalities, images):
    """`process_prompt` hands on the scaffold's layout as a ``state_layout``, which the data worker turns back into the
    ``kv_layout`` the hand-written path returned."""
    model = _model(tmp_path, max_crops=6)
    tensors = {"image_inputs": [torch.full((3, h, w), 255, dtype=torch.uint8) for h, w in images]}
    out = model.process_prompt("Free OCR.", modalities, ["text"], tensors=tensors)
    scaffold = MultimodalDecoderModel.process_prompt(model, "Free OCR.", modalities, ["text"], tensors=tensors)
    assert "kv_layout" not in out.metadata, "a ported model returns its layout under state_layout"
    # as the data worker reads them
    converted = kv_layouts(model.declare_state(), out.metadata["state_layout"], out.metadata.get("keyed_spans", {}))
    assert converted == scaffold.metadata["kv_layout"]
    assert {key: value for key, value in out.metadata.items() if key not in ("state_layout", "keyed_spans")} == {
        key: value for key, value in scaffold.metadata.items() if key != "kv_layout"
    }
    assert all(torch.equal(a, b) for name in out.new_input_tensors
               for a, b in zip(out.new_input_tensors[name], scaffold.new_input_tensors[name], strict=True))
