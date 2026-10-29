#!/usr/bin/env bash
# Start the avatar chat demo:  ./run.sh            (env: AVATAR_LLM, AVATAR_ROLE, AVATAR_INTENSITY, AVATAR_PASSWORD, HOST, PORT)
set -e
cd "$(dirname "$0")"
PY=${PY:-/home/ty/anaconda3/envs/uika/bin/python}
export AVATAR_LLM=${AVATAR_LLM:-exaone3.5:7.8b}
export AVATAR_ROLE=${AVATAR_ROLE:-taeyoung}
# AVATAR_ASSET: folder of web/assets. user = RGBAvatar trained on the person's own video (default), custom = the UIKA avatar
export AVATAR_ASSET=${AVATAR_ASSET:-user}
# 0.8 for the synthetic presets of the UIKA avatar; the RGBAvatar emotions are the person's own: 1.0
export AVATAR_INTENSITY=${AVATAR_INTENSITY:-$([ "$AVATAR_ASSET" = custom ] && echo 0.8 || echo 1.0)}
curl -sf http://127.0.0.1:11434/api/tags >/dev/null || { echo "Ollama is not running (systemctl start ollama)"; exit 1; }
echo "open http://localhost:${PORT:-8000}/   (phone on the same network: http://$(hostname -I | awk '{print $1}'):${PORT:-8000}/)"
exec "$PY" -m uvicorn server.app:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
