"""``mstar serve`` with a name it has no config for suggests the key the user meant."""

import pytest

from mstar.cli import main as cli


def _refusal(model: str) -> str:
    with pytest.raises(SystemExit) as exited:
        cli._resolve_config(model, None)
    return str(exited.value)


@pytest.mark.parametrize(("given", "meant"), [
    ("Qwen/Qwen3.5-9B", "qwen3_5_9b"),
    ("https://huggingface.co/Qwen/Qwen3.5-0.8B", "qwen3_5_0.8b"),
    ("ByteDance-Seed/BAGEL-7B-MoT", "bagel"),
    ("black-forest-labs/FLUX.2-klein-4B", "flux2_klein"),
    ("openai/whisper-large-v3-turbo", "whisper_large_v3_turbo"),
])
def test_a_hugging_face_repo_id_gets_its_model_key(given, meant):
    message = _refusal(given)
    assert f"error: unknown model {given!r}. mstar serve takes a model key, not a Hugging Face repo id. " \
           f"Did you mean {meant}?" in message


@pytest.mark.parametrize(("given", "meant"), [("kokro", "kokoro"), ("bagle", "bagel"), ("qwen3.5-9b", "qwen3_5_9b")])
def test_a_misspelled_key_gets_the_closest_one(given, meant):
    message = _refusal(given)
    assert f"Did you mean {meant}" in message
    assert "Hugging Face" not in message


def test_a_repo_with_no_config_gets_no_guess_but_still_the_list():
    message = _refusal("Qwen/Qwen3-8B")
    assert "Did you mean" not in message
    assert "Known models: bagel" in message and "--config <path.yaml>" in message


def test_every_suggestion_is_a_servable_key():
    for given in ("Qwen/Qwen3.5-27B", "nvidia/parakeet-tdt-0.6b-v3", "qwen3_5_9", "zai-org/GLM-5.2"):
        likely = cli._likely_models(given)
        assert likely and set(likely) <= set(cli.DEFAULT_CONFIGS), given
