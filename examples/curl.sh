#!/bin/bash
# curl checks against a running M* server, one model at a time.
#
#   mstar serve bagel
#   MODEL=bagel bash examples/curl.sh
#
# MODEL picks the section and must name the served model:
#   bagel                                  /generate text, chat, image generation
#   qwen3_5_0.8b qwen3_5_2b qwen3_5_4b qwen3_5_9b qwen3_5_27b qwen3_8_27b
#   muse_glimmer phi4mm glm52 glm5_next ripple_65b
#                                          /generate text, chat
#   orpheus kokoro                         /generate audio, speech, voices
#   whisper_large whisper_large_v3_turbo qwen3_asr qwen3_asr_realtime
#                                          transcription of $AUDIO (default speech.wav)
#   flux2_klein flux2_klein_9b z_image_turbo
#                                          image generation
# HOST and PORT pick the server; MSTAR_API_KEY, when set, is sent as a bearer token.
set -euo pipefail
MODEL=${MODEL:-bagel}
HOST=${HOST:-localhost}
PORT=${PORT:-8000}
BASE="http://${HOST}:${PORT}"

# -f: an HTTP error status exits nonzero instead of saving the error body as output
c() { curl -fsS ${MSTAR_API_KEY:+-H "Authorization: Bearer ${MSTAR_API_KEY}"} "$@"; }
json() { c -H 'Content-Type: application/json' "$@"; }
# the first b64_json image of a /v1/images response on stdin, written to $1
save_image() {
  python3 -c 'import base64, json, sys; open(sys.argv[1], "wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))' "$1"
}

echo "== GET /health =="
c "${BASE}/health"
echo
echo "== GET /v1/models =="
c "${BASE}/v1/models"
echo

case "$MODEL" in
  bagel|qwen3_5_0.8b|qwen3_5_2b|qwen3_5_4b|qwen3_5_9b|qwen3_5_27b|qwen3_8_27b|muse_glimmer|phi4mm|glm52|glm5_next|ripple_65b)
    echo "== POST /generate (text in, text out) =="
    c "${BASE}/generate" -F 'text=Hello, how are you?' -F 'output_modalities=text' -F 'streaming=false'
    echo
    echo "== POST /v1/chat/completions =="
    json "${BASE}/v1/chat/completions" \
      -d "{\"model\": \"${MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}"
    echo
    if [ "$MODEL" = bagel ]; then
      echo "== POST /v1/images/generations -> out.png =="
      json "${BASE}/v1/images/generations" -d '{"model": "bagel", "prompt": "a cat in a hat"}' | save_image out.png
      echo "wrote out.png"
    fi
    ;;
  orpheus|kokoro)
    if [ "$MODEL" = orpheus ]; then VOICE=tara; else VOICE=af_heart; fi
    echo "== POST /generate (text in, audio out) -> generate.json =="
    c "${BASE}/generate" -F 'text=hello there' -F 'output_modalities=audio' \
      -F "model_kwargs={\"voice\": \"${VOICE}\"}" -F 'streaming=false' -o generate.json
    echo "wrote generate.json (base64 16-bit PCM in outputs.audio)"
    echo "== POST /v1/audio/speech -> out.wav =="
    json "${BASE}/v1/audio/speech" \
      -d "{\"model\": \"${MODEL}\", \"input\": \"hello there\", \"voice\": \"${VOICE}\", \"response_format\": \"wav\"}" \
      -o out.wav
    echo "wrote out.wav"
    echo "== GET /v1/audio/voices =="
    c "${BASE}/v1/audio/voices"
    echo
    ;;
  whisper_large|whisper_large_v3_turbo|qwen3_asr|qwen3_asr_realtime)
    AUDIO=${AUDIO:-speech.wav}
    if [ ! -f "$AUDIO" ]; then
      echo "no audio file at ${AUDIO}; set AUDIO to a speech recording" >&2
      exit 2
    fi
    echo "== POST /v1/audio/transcriptions (${AUDIO}) =="
    c "${BASE}/v1/audio/transcriptions" -F "file=@${AUDIO}" -F "model=${MODEL}"
    echo
    ;;
  flux2_klein|flux2_klein_9b|z_image_turbo)
    echo "== POST /v1/images/generations -> out.png =="
    json "${BASE}/v1/images/generations" \
      -d "{\"model\": \"${MODEL}\", \"prompt\": \"a cat in a hat\", \"seed\": 0}" | save_image out.png
    echo "wrote out.png"
    ;;
  *)
    echo "MODEL=${MODEL} has no section in this script; the list is at its top" >&2
    exit 2
    ;;
esac
