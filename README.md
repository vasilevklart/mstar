<p align="center"><img src="assets/mstar-logo.svg" alt="M*" width="300"></p>

<h3 align="center">A serving system for composite, any-to-any multimodal models</h3>

<p align="center"><a href="https://m-star-labs.mintlify.site/quickstart"><b>Quickstart</b></a> &nbsp;·&nbsp; <a href="https://m-star-labs.mintlify.site/models"><b>Models</b></a> &nbsp;·&nbsp; <a href="https://m-star-labs.mintlify.site"><b>Docs</b></a> &nbsp;·&nbsp; <a href="https://m-star.org/"><b>Blog</b></a> &nbsp;·&nbsp; <a href="https://arxiv.org/abs/2606.12688"><b>Paper</b></a></p>

<p align="center"><a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-7b61ff.svg" alt="License: Apache 2.0"></a> <img src="https://img.shields.io/badge/python-3.12-22d3ee.svg" alt="Python 3.12"> <a href="https://m-star-labs.mintlify.site"><img src="https://img.shields.io/badge/docs-online-3b82f6.svg" alt="Docs"></a> <a href="https://arxiv.org/abs/2606.12688"><img src="https://img.shields.io/badge/paper-arXiv-b31b1b.svg" alt="Paper (arXiv)"></a></p>

**M\*** (M-star) serves composite multimodal models, built from components such as vision encoders, transformer backbones, diffusion heads, audio codecs and action generators, with text, image, audio, video and robot actions in and out. Each model is a dataflow graph of its components, each request is a walk over that graph, and a YAML config places the components on GPUs.

## Quickstart

You need Linux, an NVIDIA GPU, Python 3.12, [uv](https://docs.astral.sh/uv/), [rustup](https://rustup.rs) and a C/C++ compiler.

```bash
git clone https://github.com/mstar-project/mstar.git
cd mstar
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install --torch-backend=auto -e ".[bagel]"

# Rust graph runtime and HTTP frontend (needs rustup)
uv pip install maturin
maturin develop --release -m rust/Cargo.toml
cargo build --release --manifest-path rust/server/Cargo.toml

mstar serve bagel
```

The first start downloads the weights and compiles the model before the server listens, which can take minutes. In a second terminal with the same `.venv` active:

```bash
curl http://localhost:8000/health
```

```python
from mstar import MStarClient

client = MStarClient("http://localhost:8000")
print(client.chat("What is the capital of France?").text)
```

- Another model: install its extra and run `mstar serve <key>`. The [model matrix](https://m-star-labs.mintlify.site/models) gives each key's GPUs and extra.
- Your own layout: `mstar serve <model> --config <path>`. See [Serving](https://m-star-labs.mintlify.site/serving).
- PyPI has `mstar-ai` 0.2.1, which has neither the Rust runtime nor `mstar-server` and knows 14 of the 61 model keys. Install from source until the next release.
- [Installation](https://m-star-labs.mintlify.site/installation) covers extras, CUDA matching, Docker and troubleshooting. Runnable scripts are in [`examples/`](examples/).

## Supported models

Every key serves the native `/generate` and `/generate/ws` endpoints and the Python SDK. The Endpoints column lists its OpenAI-compatible routes.

| Model | In → Out | Key | Endpoints |
|---|---|---|---|
| [Qwen3.5, Qwen3.8](https://m-star-labs.mintlify.site/models/qwen) | text, image, video → text | `qwen3_5_0.8b` `qwen3_5_2b` `qwen3_5_4b` `qwen3_5_9b` `qwen3_5_27b` `qwen3_8_27b` | `/v1/chat/completions` |
| [GLM-5.2, GLM-5.3-Flash](https://m-star-labs.mintlify.site/models/glm) | text → text | `glm52` `glm5_next` | `/v1/chat/completions` |
| [Command A+](https://m-star-labs.mintlify.site/models/command-a-plus) (experimental) | text, image → text | `command_a_plus` | native only |
| [Muse Glimmer 30B](https://m-star-labs.mintlify.site/models/muse-glimmer) (beta) | text, image → text | `muse_glimmer` | `/v1/chat/completions` |
| [Ripple-65B](https://m-star-labs.mintlify.site/models/ripple) (experimental) | text → text | `ripple_65b` | `/v1/chat/completions` |
| [Phi-4-multimodal](https://m-star-labs.mintlify.site/models/phi4mm) (beta) | text, image, audio → text | `phi4mm` | `/v1/chat/completions` |
| [Qwen3-Omni](https://m-star-labs.mintlify.site/models/qwen3-omni) | text, image, audio, video → text, speech | `qwen3_omni` | `/v1/chat/completions` `/v1/audio/speech` |
| [MiniCPM-o 4.5](https://m-star-labs.mintlify.site/models/minicpm-o) (beta) | text, image, audio → text, speech | `minicpm_o` | native only |
| [BAGEL](https://m-star-labs.mintlify.site/models/bagel) | text, image → text, image | `bagel` `bagel_cfg_parallel` | `/v1/chat/completions` `/v1/images/generations` `/v1/images/edits` |
| [Kokoro-82M](https://m-star-labs.mintlify.site/models/kokoro) | text → speech | `kokoro` | `/v1/audio/speech` |
| [Orpheus](https://m-star-labs.mintlify.site/models/orpheus) | text → speech | `orpheus` | `/v1/audio/speech` |
| [Chatterbox](https://m-star-labs.mintlify.site/models/chatterbox) | text, reference audio → speech | `chatterbox` `chatterbox_multilingual` `chatterbox_turbo` | `/v1/audio/speech` |
| [Fun-CosyVoice3](https://m-star-labs.mintlify.site/models/cosyvoice3) | text, reference audio → speech | `cosyvoice3` | `/v1/audio/speech` |
| [Qwen3-TTS](https://m-star-labs.mintlify.site/models/qwen3-tts) | text, voice instruction or reference audio → speech | `qwen3_tts` `qwen3_tts_1p7b` `qwen3_tts_voicedesign` `qwen3_tts_base` | `/v1/audio/speech` |
| [Fish Audio S2-Pro](https://m-star-labs.mintlify.site/models/fish-s2-pro) | text, reference audio → speech | `fish_s2_pro` | `/v1/audio/speech` |
| [VoxCPM2](https://m-star-labs.mintlify.site/models/voxcpm2) | text, reference audio → speech | `voxcpm2` | `/v1/audio/speech` |
| [OmniVoice](https://m-star-labs.mintlify.site/models/omnivoice) | text, reference audio → speech | `omnivoice` | `/v1/audio/speech` |
| [NemotronLabs VoiceChat](https://m-star-labs.mintlify.site/models/nemotron-voicechat) (beta) | speech → text, speech (full duplex) | `nemotron_duplex` | `/v1/realtime` |
| [Whisper, Qwen3-ASR](https://m-star-labs.mintlify.site/models/whisper-qwen-asr) | speech → text | `whisper_large` `whisper_large_v3_turbo` `qwen3_asr` `qwen3_asr_realtime` | `/v1/audio/transcriptions`, and `/v1/realtime` for Qwen3-ASR |
| [Canary, Parakeet, Nemotron ASR](https://m-star-labs.mintlify.site/models/fastconformer-asr) | speech → text | `canary` `canary_qwen` `parakeet` `nemotron_asr` | `/v1/audio/transcriptions` |
| [Omnilingual ASR](https://m-star-labs.mintlify.site/models/omnilingual-asr) | speech → text | `omnilingual_asr_ctc_300m` `omnilingual_asr_ctc_1b` `omnilingual_asr_ctc_3b` `omnilingual_asr_ctc_7b` | `/v1/audio/transcriptions` |
| [Higgs Audio v3 STT](https://m-star-labs.mintlify.site/models/higgs-audio) (beta) | speech → text | `higgs_audio` | `/v1/audio/transcriptions` |
| [FLUX.2 klein, Z-Image-Turbo](https://m-star-labs.mintlify.site/models/flux-z-image) | text, image → image | `flux2_klein` `flux2_klein_9b` `z_image_turbo` | `/v1/images/generations`, and `/v1/images/edits` for FLUX.2 klein |
| [Cosmos3](https://m-star-labs.mintlify.site/models/cosmos3) | text, image, video → image, video, robot actions | `cosmos3` `cosmos3_droid` `cosmos3_edge` `cosmos3_edge_droid` `cosmos3_super` `cosmos3_super_i2v_4step` `cosmos3_super_t2i_4step` | `/v1/images/generations` `/v1/videos/generations`, and `/v1/chat/completions` for Edge |
| [Wan2.2-TI2V-5B](https://m-star-labs.mintlify.site/models/wan22) | text, image → video | `wan22` | `/v1/videos/generations` |
| [LTX-2.5](https://m-star-labs.mintlify.site/models/ltx2-5) (beta) | text → video with audio | `ltx2_5` | `/v1/videos/generations` |
| [MiniMax-H3](https://m-star-labs.mintlify.site/models/minimax-h3) (beta) | text, image → video with audio | `minimax_h3` | `/v1/videos/generations` |
| [Waypoint-1.5, LingBot-World](https://m-star-labs.mintlify.site/models/world-models) | text, image, controller input or camera path → video | `waypoint` `lingbot_world_1_3b` | native only |
| [V-JEPA 2, V-JEPA 2-AC](https://m-star-labs.mintlify.site/models/world-models) | video, actions → latents, rollouts | `vjepa2` `vjepa2_ac` | native only |
| [Pi0.5](https://m-star-labs.mintlify.site/models/world-models) | text, images, robot state → robot actions | `pi05` | native only; openpi WebSocket with `pi05_droid.yaml` |

Hardware: M\* runs on Linux with NVIDIA GPUs. BAGEL also runs on Intel GPUs (XPU), and AMD Instinct support is partial. [Supported hardware](https://m-star-labs.mintlify.site/hardware) lists what is tested on each platform.

## How it works

A model declares a graph of its components and named walks over that graph, such as BAGEL's `prefill_text`, `decode` and `image_gen`.
The conductor turns each request into walks and schedules them across workers. Each worker runs the nodes placed on its GPU and sends tensors directly to the next worker.
The config's `node_groups` map nodes to GPU ranks, so the same model runs on one GPU or split across several without code changes.
`mstar-server`, a Rust binary, serves the HTTP routes, and the graph runtime is the Rust `mstar_rust` extension.
See [Architecture](https://m-star-labs.mintlify.site/architecture) and the [paper](https://arxiv.org/abs/2606.12688).

## Contributing

Issues and pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). To add a model, follow [Adding a model](https://m-star-labs.mintlify.site/adding_models).

## Citation

```bibtex
@article{mstar2026,
  title  = {M*: A Modular, Extensible, Serving System for Multimodal Models},
  author = {Jha, Atindra and Sagan, Naomi and Kamahori, Keisuke and Sivgin, Irmak and
            Sanda, Rohan and Gao, Steven and Horowitz, Mark and Zettlemoyer, Luke and
            Hsu, Olivia and Leskovec, Jure and Kasikci, Baris and Wang, Stephanie},
  year   = {2026},
  eprint = {2606.12688},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG}
}
```

From Stanford University and the University of Washington. Correspondence: `atindra@cs.stanford.edu`.

M\* builds on [vLLM](https://github.com/vllm-project/vllm) (paged attention, continuous batching), [FlashInfer](https://github.com/flashinfer-ai/flashinfer) kernels, VoxServe (streaming speech serving) and [Mooncake](https://github.com/kvcache-ai/Mooncake) (RDMA tensor transport).

## License

[Apache License 2.0](LICENSE).
