FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git gosu && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

ARG STT_MODEL_SIZE=base
ENV HF_HOME=/app/.hf_cache \
    STT_MODEL_SIZE=${STT_MODEL_SIZE} \
    STT_MODEL_PATH=/models/whisper
RUN mkdir -p /models/whisper /app/.hf_cache \
    && python -c "from faster_whisper.utils import download_model; download_model('${STT_MODEL_SIZE}', output_dir='/models/whisper')"

COPY main.py .
COPY mood_analysis.py .
COPY docker-entrypoint.sh .
RUN chmod 755 docker-entrypoint.sh
COPY knowledge ./knowledge
COPY prompts ./prompts
RUN mkdir -p /data/audio_cache /tmp/happify \
    && useradd --system --uid 10001 --create-home happify \
    && chown -R happify:happify /app /models /data /tmp/happify

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    OLLAMA_API_URL= \
    VOICE_LANGUAGE=en \
    VOICE_TTS_VOICE=en-US-JennyNeural \
    VOICE_TTS_RATE=-10% \
    VOICE_AUDIO_CACHE_DIR=/data/audio_cache \
    VOICE_TEMP_DIR=/tmp/happify \
    KNOWLEDGE_MANIFEST_PATH=/app/knowledge/manifest.v1.json \
    PROMPT_REGISTRY_PATH=/app/prompts/registry.v1.json \
    STT_DEVICE=auto

ENTRYPOINT ["/app/docker-entrypoint.sh"]
