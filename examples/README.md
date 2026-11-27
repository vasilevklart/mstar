# Examples

Clients for a running M\* server. Start the server from the repository root (paths such as `configs/...` are relative to it), wait until `curl -f localhost:8000/health` succeeds, then run the example from another shell, for example `python examples/sdk_chat.py`.

| File | Start the server | Endpoint | GPUs |
|---|---|---|---|
| `curl.sh` | `mstar serve bagel`, then `MODEL=bagel bash examples/curl.sh`; the script's header lists the other keys it covers | `/health`, `/v1/models`, the model's `/v1` routes, and `/generate` for the text and speech keys | the model's |
| `sdk_chat.py` | `mstar serve bagel` | `/generate` | 1 |
| `sdk_image.py` | `mstar serve bagel` | `/generate` | 1 |
| `sdk_image_edit.py` | `mstar serve flux2_klein` | `/generate` | 1 |
| `sdk_tts.py` | `mstar serve orpheus` | `/generate` | 1 |
| `sdk_tts_qwen3.py` | `mstar serve qwen3_tts_1p7b` (argument `custom_voice`, the default), `qwen3_tts_voicedesign` (`voice_design`) or `qwen3_tts_base` (`voice_clone`) | `/generate` | 1 |
| `sdk_transcribe.py` | `mstar serve whisper_large_v3_turbo` | `/generate` | 1 |
| `openai_chat.py` | `mstar serve qwen3_omni` | `/v1/chat/completions` | 2 |
| `openai_tts.py` | `mstar serve orpheus` | `/v1/audio/speech` | 1 |
| `openai_transcribe.py` | `mstar serve whisper_large_v3_turbo` | `/v1/audio/transcriptions` | 1 |
| `livekit_kokoro.py` | `mstar serve kokoro` | `/v1/audio/speech` | 1 |
| `pipecat_kokoro.py` | `mstar serve kokoro` | `/v1/audio/speech` | 1 |
| `cosmos3_action_ws_client.py` | `mstar serve cosmos3_droid`, or `mstar serve cosmos3_edge_droid` and the client's `--json-prompt` | `/generate/ws` | 1 |
| `cosmos3_stream_video_ws_client.py` | `mstar serve cosmos3 --config configs/cosmos3_nano_ar.yaml` | `/generate/ws` | 1 |
| `livekit/` | three servers, `whisper_large`, `qwen3_5_4b` and `kokoro`, on ports 8001 to 8003 ([its README](livekit/README.md)) | `/v1/audio/transcriptions`, `/v1/chat/completions`, `/v1/audio/speech` | 1, shared |
| `observability/` | `mstar serve kokoro --enable-metrics` ([its README](observability/README.md)) | `/metrics` | 1 |

GPUs is the device count of the key's default config; `mstar serve <key> --gpus 0,1` picks which devices.

- `openai_chat.py` works against any key with a chat adapter, for example `mstar serve qwen3_5_4b` on one GPU. The `model` field in its requests is read only when the server serves LoRA adapters, so the name it sends does not matter. The keys with a chat route are in the OpenAI coverage table of [`docs/mintlify/reference/http-api.mdx`](../docs/mintlify/reference/http-api.mdx).
- The `openai_*` examples need `pip install openai`, `livekit_kokoro.py` needs `pip install "livekit-agents[openai]"`, and `pipecat_kokoro.py` needs `pip install "pipecat-ai[openai]"`.
- `sdk_transcribe.py` and `openai_transcribe.py` take a speech file as their argument (default `speech.wav`). `sdk_tts_qwen3.py voice_clone` reads `reference.wav`.
- The default `cosmos3` config leaves windowed video off, so the streaming video client needs `configs/cosmos3_nano_ar.yaml` (or a `cosmos3_edge` config).
- With `--api-key` on the server, `curl.sh` and the SDK examples read `MSTAR_API_KEY` from the environment. The `openai_*` examples send `api_key="none"`; put the key there.
