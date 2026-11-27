"""MiniMax-H3-Image on the state contract builds the specs and steps it wrote by hand, field for field.

The DiT and the token refiner declare MiniMax-H3's state (`dit_state`), and the decoder's sequence shard over its tiles
is MiniMax-H3's, written by hand (`vae_shard_spec`), since the decoder has no arrays. The engine sees the same objects
whichever way a spec arrives, so the port changes nothing it runs: the same resource names, kernels and capture sizes.
The hand-written specs and steps below are the reference, copied from the model as it stood before it took H3's
declarations (its own ``get_node_resources`` and the decoder's shard), with the fields main's contract renamed, so the
adapter can't pass by agreeing with itself.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

# the resources import Triton kernels (a torch dependency on Linux)
pytest.importorskip("triton")

from mstar.distributed.communication import CommGroup, JointGroups  # noqa: E402
from mstar.engine.resources.attn.config import AttentionStep  # noqa: E402
from mstar.engine.resources.attn.ragged.config import RaggedAttentionConfig, RaggedAttentionSpec  # noqa: E402
from mstar.engine.resources.base import EngineResourceInfo  # noqa: E402
from mstar.engine.resources.sequence_shard import SeqShardSpec, SequenceShard  # noqa: E402
from mstar.engine.resources.spec import apply_yaml_overrides  # noqa: E402
from mstar.engine.resources.step import ResourceStep, Segment, StepContext, SubmoduleStep  # noqa: E402
from mstar.engine.state import SeqShard  # noqa: E402
from mstar.engine.state.adapters import StateSetup, node_handles, node_resources, state_setup, state_steps  # noqa: E402
from mstar.engine.state.analysis import state_declaration_problems  # noqa: E402
from mstar.model.base import enabled_resources, serving_resources  # noqa: E402
from mstar.model.minimax_h3.config import DENOISE_LOOP  # noqa: E402
from mstar.model.minimax_h3.submodules import H3TextEncoderSubmodule  # noqa: E402
from mstar.model.minimax_h3_image.config import H3ImageConfig  # noqa: E402
from mstar.model.minimax_h3_image.layout import H3ImageLayout  # noqa: E402
from mstar.model.minimax_h3_image.minimax_h3_image_model import (  # noqa: E402
    ENCODE_REFERENCES_WALK,
    ENCODE_TEXT_WALK,
    GENERATE_EDIT_WALK,
    GENERATE_WALK,
    H3ImageModel,
)
from mstar.model.minimax_h3_image.submodules import H3ImageDecoderSubmodule, H3ImageDenoiseSubmodule  # noqa: E402
from mstar.model.submodule_base import NodeInputs, engine_step, with_segment_steps  # noqa: E402

CONFIGS = sorted(str(path) for path in (Path(__file__).parents[4] / "configs").glob("minimax_h3_image*.yaml"))


def _model() -> H3ImageModel:
    model = H3ImageModel(model_path_hf="test/minimax_h3_image", skip_weight_loading=True)
    model.set_config(H3ImageConfig())
    return model


def _reference() -> list:
    """The specs H3-Image wrote by hand: the DiT's attention over each request's packed rows and their padding (two
    segments), the refiner's over the prompt, both bf16 in 128 MB workspaces, the DiT's sequence shard over its rows,
    and the decoder's over its tiles, which nothing exchanges, one count per request."""
    t = H3ImageConfig().transformer
    return [
        RaggedAttentionSpec(
            resource_key=key, nodes={node}, labels={label},
            config=RaggedAttentionConfig(
                num_qo_heads=t.num_attention_heads, num_kv_heads=t.num_attention_heads, head_dim=t.attention_head_dim,
                max_segments_per_request=segments, dtype=torch.bfloat16, workspace_mb=128,
            ),
        )
        for key, node, label, segments in (("dit_attn", "dit", "main", 2), ("refiner_attn", "text_encoder", "text", 1))
    ] + [
        SeqShardSpec(resource_key="dit_shard", nodes={"dit"}, labels=("main",), axis=0),
        SeqShardSpec(resource_key="vae_shard", nodes={"vae_decoder"}, labels=("tiles",), axis=1, exchanges=False),
    ]


def _assert_equal_per_key(specs, reference) -> None:
    keys = [spec.resource_key for spec in specs]
    assert len(set(keys)) == len(keys), "two specs under one key"
    assert keys == [spec.resource_key for spec in reference]
    for spec, expected in zip(specs, reference, strict=True):
        assert spec == expected, f"{expected.resource_key} differs from the hand-written spec"


def test_the_geometry_is_the_checkpoints():
    """The reference reads the config, so pin the numbers it reads: 56 heads of 128, 50 DiT and 2 refiner layers."""
    t = H3ImageConfig().transformer
    assert (t.num_attention_heads, t.attention_head_dim, t.num_layers, t.token_refiner_num_layers) == (56, 128, 50, 2)


def test_specs_equal_the_hand_written_ones():
    model = _model()
    _assert_equal_per_key(node_resources(model), _reference())
    _assert_equal_per_key(enabled_resources(model, {}, StateSetup()), _reference())


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: Path(path).name)
def test_specs_equal_the_hand_written_ones_on_each_published_config(path):
    deployment = yaml.safe_load(Path(path).read_text())
    expected = _reference()
    # derives each spec's fields from the others, as the deployment's own do: the DiT's attention exchanges its
    # shard's rows by Ulysses, the refiner's nothing
    apply_yaml_overrides(expected, deployment)
    specs = serving_resources(_model(), deployment)
    _assert_equal_per_key(specs, expected)
    by_key = {spec.resource_key: spec for spec in specs}
    assert (by_key["dit_attn"].sp_exchange, by_key["refiner_attn"].sp_exchange) == ("ulysses", "none")


def test_specs_equal_the_hand_written_ones_after_yaml_overrides():
    overrides = {"resources": {
        **{key: {"max_segments_per_request": 7, "flashinfer_backend": "fa2"} for key in ("dit_attn", "refiner_attn")},
        "vae_shard": {"min_tokens": 3},
    }}
    specs, expected = node_resources(_model()), _reference()
    apply_yaml_overrides(specs, overrides)
    apply_yaml_overrides(expected, overrides)
    _assert_equal_per_key(specs, expected)
    assert next(spec for spec in specs if spec.resource_key == "vae_shard").min_tokens == 3, "the YAML should reach it"


def test_sdpa_builds_only_the_shards_and_the_declarations_have_no_problems():
    """The docs' ``backend: sdpa`` on both attentions builds no attention spec, as the hand-written model's did."""
    model = _model()
    sdpa = {"resources": {"dit_attn": {"backend": "sdpa"}, "refiner_attn": {"backend": "sdpa"}}}
    setup, problems = state_setup(sdpa, model.declare_state())
    assert not problems, problems
    _assert_equal_per_key(node_resources(model, setup=setup), _reference()[2:])
    assert state_declaration_problems(model) == []


class _StubRagged:
    """Stands in for the built resource a handle binds to."""

    def run(self, q, k=None, v=None, label=None):
        return q


def _inputs(key) -> NodeInputs:
    return NodeInputs(tensor_inputs={}, resource_step_info=key)


def _one_rank_shards(model) -> dict:
    """``model``'s declared sequence shards as one rank holds them, which split nothing."""
    return {
        declared.name: SequenceShard(tuple(declared.labels), 0, None, 0, SimpleNamespace(world_size=1))
        for declared in model.declare_state() if isinstance(declared, SeqShard)
    }


def _by_label(step) -> dict[str, tuple[Segment, ...]]:
    labels: dict[str, list[Segment]] = {}
    for segment in step.segments:
        labels.setdefault(segment.label, []).append(segment)
    return {label: tuple(segments) for label, segments in labels.items()}


def _assert_plans_equal(step: SubmoduleStep, reference: SubmoduleStep, key: str) -> None:
    assert step.steps.keys() == reference.steps.keys() and step.cg_key_info == reference.cg_key_info
    assert step.segments == reference.segments
    assert isinstance(step.steps[key], AttentionStep) and step.steps[key].causal is False
    assert step.steps[key].fixed_layout == reference.steps[key].fixed_layout, "a captured bucket replays its layout"
    assert _by_label(step.steps[key]) == _by_label(reference.steps[key])


def _layout() -> H3ImageLayout:
    """A one-reference edit of 13 rows (6 prompt, 1 reference, 6 generated), which no SP size past 1 divides."""
    return H3ImageLayout(text_len=6, latent_height=4, latent_width=6, references=((2, 2),))


@pytest.mark.parametrize("row_bucket", [1, 256], ids=["eager", "compiled"])
@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("walk", [GENERATE_WALK, GENERATE_EDIT_WALK])
def test_dit_steps_equal_the_hand_written_one_on_every_rank(walk, world, row_bucket):
    """Every rank plans the whole row, its padding a segment of its own, since the exchange gives each rank all of it.
    The rows pad to a multiple of the row bucket (one compiled graph per 256 rows) and of the shard's SP ranks."""
    model, layout = _model(), _layout()
    sub = H3ImageDenoiseSubmodule(SimpleNamespace(), model.config, loop_name=DENOISE_LOOP, row_bucket=row_bucket)
    sub.bind_node_state({"dit_shard": SimpleNamespace(ranks=world)})
    rows = layout.total_rows
    assert rows == 13
    padded = {1: {1: 13, 2: 14, 4: 16, 8: 16}, 256: dict.fromkeys((1, 2, 4, 8), 256)}[row_bucket][world]
    segments = [Segment(request_id=0, label="main", span=span) for span in (rows, padded - rows) if span]
    reference = SubmoduleStep(
        segments=segments, steps={"dit_attn": AttentionStep(causal=False, fixed_layout=True)}, cg_key_info=layout,
    )
    handles = node_handles(model, "dit", {**_one_rank_shards(model), "dit_attn": _StubRagged()})
    _assert_plans_equal(state_steps(sub.declare_step(walk, [0], [_inputs(layout)]), handles), reference, "dit_attn")


@pytest.mark.parametrize("walk", [ENCODE_TEXT_WALK, ENCODE_REFERENCES_WALK])
def test_refiner_step_equals_the_hand_written_one(walk):
    handles = node_handles(_model(), "text_encoder", {"refiner_attn": _StubRagged()})
    sub = H3TextEncoderSubmodule(nn.Identity(), nn.Identity())
    inputs = [NodeInputs(tensor_inputs={}, input_seq_len=n) for n in (7, 5)]
    reference = SubmoduleStep(segments=[Segment(request_id=0, label="text", span=7),
                                        Segment(request_id=1, label="text", span=5)],
                              steps={"refiner_attn": AttentionStep(causal=False)})
    _assert_plans_equal(state_steps(sub.declare_step(walk, [0, 1], inputs), handles), reference, "refiner_attn")


def _decoder_shard(world: int) -> SequenceShard:
    """The decoder's shard, built from the model's own spec as the engine builds it, rank 0 of ``world`` SP ranks."""
    (spec,) = [spec for spec in node_resources(_model()) if spec.resource_key == "vae_shard"]
    group = SimpleNamespace(world_size=world, rank=0)
    joint = JointGroups(tp_group=CommGroup.trivial(), sp_group=group, instance=group)
    return SequenceShard.build(spec, EngineResourceInfo(device=torch.device("cpu"), joint_comm_group=joint))


@pytest.mark.parametrize("world, counts", [(1, None), (2, (3, 3)), (4, (2, 2, 1, 1))])
def test_decoder_step_equals_the_hand_written_one_and_splits_the_tiles(world, counts):
    """Each request's six tiles (a 10 x 6 latent grid in 64-px tiles) under the shard's label, as the decoder declared
    them by hand, which the shard deals out contiguously, a rank getting none where tiles run short."""
    from mstar.model.minimax_h3_image.tests.test_model import _tiny_decode_case

    vae, config, layout, _ = _tiny_decode_case()
    node = H3ImageDecoderSubmodule(vae, config)
    shard = _decoder_shard(world)
    node.bind_node_resources({"vae_shard": shard})
    rids = ["r0", "r1"]
    step = engine_step(node, GENERATE_WALK, rids, [_inputs(layout)] * 2)
    reference = with_segment_steps(
        SubmoduleStep(steps={}, segments=[Segment(rid, "tiles", 6) for rid in rids]), ("vae_shard",),
    )
    assert step.segments == reference.segments and step.steps.keys() == reference.steps.keys() == {"vae_shard"}
    assert step.steps["vae_shard"].segments == reference.steps["vae_shard"].segments
    ctx = StepContext(request_ids=rids, graph_walk=GENERATE_WALK, slot=0, capture=False)
    plan = shard.plan(ResourceStep(segments=tuple(step.segments)), ctx)
    assert (plan.ranges["tiles"].counts if counts else plan.ranges) == (counts or {})


def test_a_decoder_without_a_shard_declares_no_step():
    from mstar.model.minimax_h3_image.tests.test_model import _tiny_decode_case

    vae, config, layout, _ = _tiny_decode_case()
    assert engine_step(H3ImageDecoderSubmodule(vae, config), GENERATE_WALK, ["r0"], [_inputs(layout)]) is None
