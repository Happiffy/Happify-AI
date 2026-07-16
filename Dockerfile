# Use python-slim as a lightweight, clean base image
FROM python:3.11-slim

# Install system dependencies (ffmpeg is required by Faster-Whisper for audio decoding)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy dependency registry
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and mock files
COPY main.py .
COPY test_cases.json .

# Create cache directories
RUN mkdir -p /app/audio_cache /app/temp_audio /app/.hf_cache

# Expose FastAPI port
EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV OLLAMA_API_URL=
ENV VOICE_LANGUAGE=id
ENV VOICE_TTS_VOICE=id-ID-GadisNeural
ENV VOICE_TTS_RATE=-10%
ENV VOICE_AUDIO_CACHE_DIR=/app/audio_cache
ENV HF_HOME=/app/.hf_cache
ENV STT_MODEL_SIZE=base
ENV STT_DEVICE=auto

# Command to run the gateway server
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
