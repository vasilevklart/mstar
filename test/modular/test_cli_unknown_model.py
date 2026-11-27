"""``mstar serve`` with a name it has no config for suggests the key the user meant."""

import subprocess
import sys

import pytest

from mstar.cli import main as cli
from mstar.model.registry import HF_MODELS


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
    # a longer key of the same family serves these, not the shortest one they start with
    ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "qwen3_tts_base"),
    ("Qwen/Qwen3-ASR-0.6B", "qwen3_asr_realtime"),
    ("nvidia/Cosmos3-Nano-Policy-DROID", "cosmos3_droid"),
    # nothing in the key spells these
    ("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", "cosyvoice3"),
    ("microsoft/Phi-4-multimodal-instruct", "phi4mm"),
    # one repo, two keys
    ("ResembleAI/chatterbox", "chatterbox or chatterbox_multilingual"),
])
def test_a_hugging_face_repo_id_gets_its_model_key(given, meant):
    message = _refusal(given)
    assert f"error: unknown model {given!r}. mstar serve takes a model key, not a Hugging Face repo id. " \
           f"Did you mean {meant}?" in message


def test_every_checkpoint_the_registry_serves_gets_the_keys_that_serve_it():
    def checkpoint(key):
        return HF_MODELS.get(key, {}).get("model_path_hf")

    repos = {checkpoint(key) for key in cli.DEFAULT_CONFIGS if "/" in (checkpoint(key) or "")}
    assert len(repos) > 40
    for repo in repos:
        keys = [key for key in cli.DEFAULT_CONFIGS if checkpoint(key) == repo]
        assert f"Did you mean {' or '.join(keys)}?" in _refusal(repo), repo


@pytest.mark.parametrize(("given", "meant"), [
    ("zai-org/GLM-5.2", "glm52 (serves zai-org/GLM-5.2-FP8)"),
    ("Qwen/Qwen3.5-9B-Base", "qwen3_5_9b (serves Qwen/Qwen3.5-9B)"),
    ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "wan22 (serves Wan-AI/Wan2.2-TI2V-5B-Diffusers)"),
])
def test_a_guess_for_another_checkpoint_names_the_one_it_serves(given, meant):
    assert f"Did you mean {meant}?" in _refusal(given)


def test_a_config_file_is_sent_to_config():
    message = _refusal("configs/kokoro.yaml")
    assert "A config file goes in --config: mstar serve <model key> --config configs/kokoro.yaml." in message
    assert "Hugging Face" not in message and "Did you mean" not in message


def test_the_registry_lookup_imports_no_torch():
    # an unknown name is refused at once, not after torch and the model code import
    code = "import sys; from mstar.model.registry import HF_MODELS; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


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
