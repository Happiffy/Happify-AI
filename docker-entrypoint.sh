#!/bin/sh
set -eu
mkdir -p "$VOICE_AUDIO_CACHE_DIR" "$VOICE_TEMP_DIR"
chown -R happify:happify "$VOICE_AUDIO_CACHE_DIR" "$VOICE_TEMP_DIR"
exec gosu happify uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
