# Happify AI Service

Happify AI is the service behind Happify Companion voice and journal wellbeing features. It accepts authenticated requests from Happify Backend and returns structured transcription, supportive responses, mood analysis, risk signals, journal reflections, and optional response audio.

## Overview

Happify AI is designed as an early-support system. It can help users recognize and reflect on emotional patterns, but it is not a psychologist, psychiatrist, medical device, diagnostic tool, or emergency service.

## Technology Stack

| Area | Stack |
| --- | --- |
| Runtime | Python 3.11 |
| Framework | FastAPI, Uvicorn |
| Speech-to-Text | Faster-Whisper |
| Text-to-Speech | Edge TTS |
| Audio Processing | FFmpeg |
| LLM Integration | Optional Ollama |
| Validation | Pydantic |
| Knowledge Retrieval | Local governed lexical retrieval |
| Deployment | Docker |

## Features

- **Speech-to-Text** — Converts user audio to transcripts with Faster-Whisper.
- **Companion Responses** — Generates supportive conversational responses.
- **Mood Analysis** — Recognizes calm, happy, neutral, sad, anxious, and distressed signals.
- **Risk Policy** — Applies a deterministic risk floor before and after LLM processing.
- **Journal Analysis** — Returns reflection, mood, risk level, and suggested actions from journal entries.
- **Governed Knowledge** — Uses local knowledge sources verified by hash at startup.
- **Versioned Prompts** — Uses a versioned prompt registry with hash metadata.
- **Optional TTS** — Generates response audio with Edge TTS.
- **Recording Quality** — Measures audio quality without making clinical claims.
- **Multimodal Fusion Contract** — Accepts extracted observation signals from callers.
- **Privacy Boundaries** — Does not receive or store raw images in fusion endpoints.
- **Fallback Mode** — Provides local responses and mood analysis when Ollama is unavailable.

## API Routes

| Method | Path | Authentication | Description |
| --- | --- | --- | --- |
| `GET` | `/health/live` | None | Process liveness check |
| `GET` | `/health/ready` | None | Readiness check for authentication, FFmpeg, assets, and STT |
| `GET` | `/health` | None | Readiness alias |
| `POST` | `/api/process-audio` | Bearer | Transcription, response, mood, risk, and optional TTS |
| `POST` | `/api/analyze-journal` | Bearer | Journal reflection and analysis |
| `POST` | `/api/fuse-observations` | Bearer | Transcript and extracted-observation fusion |
| `GET` | `/api/audio/{filename}` | Bearer | Protected cached TTS audio |

## Request Flow

```text
User voice
    |
    v
Happify Backend
    |
    v
Happify AI
    |
    +--> Normalize audio with FFmpeg
    +--> Transcribe with Faster-Whisper
    +--> Apply deterministic risk policy
    +--> Retrieve governed support context
    +--> Generate a response or use a local fallback
    +--> Analyze mood and confidence
    +--> Optionally generate TTS audio
    |
    v
Structured response to Happify Backend
```

## Environment Variables

Create `.env` from `.env.example`.

```env
PORT=8000
LOG_LEVEL=INFO

STT_MODEL_SIZE=base
STT_MODEL_PATH=/models/whisper
STT_DEVICE=cpu

OLLAMA_API_URL=
OLLAMA_API_KEY=
OLLAMA_MODEL_NAME=qwen2.5:1.5b

VOICE_LANGUAGE=en
VOICE_TTS_VOICE=en-US-JennyNeural
VOICE_TTS_RATE=-10%
VOICE_AUDIO_CACHE_DIR=/data/audio_cache
VOICE_TEMP_DIR=/tmp/happify

AI_SERVICE_TOKEN=
MAX_AUDIO_BYTES=6291456
MAX_CONCURRENT_TURNS=2
CV_FUSION_MAX_OBSERVATIONS=10

KNOWLEDGE_MANIFEST_PATH=/app/knowledge/manifest.v1.json
PROMPT_REGISTRY_PATH=/app/prompts/registry.v1.json
CACHE_CLEANUP_INTERVAL_SECONDS=900
TTS_CACHE_TTL_SECONDS=86400
TEMP_FILE_TTL_SECONDS=3600
```

| Variable | Description |
| --- | --- |
| `PORT` | HTTP service port. |
| `AI_SERVICE_TOKEN` | Shared bearer token required by Happify Backend. |
| `STT_MODEL_SIZE` | Faster-Whisper model size; `base` is suitable for an MVP. |
| `STT_MODEL_PATH` | Path to the local or containerized model. |
| `STT_DEVICE` | Inference device, such as `cpu`. |
| `OLLAMA_API_URL` | Optional Ollama URL; an empty value enables the local fallback path. |
| `VOICE_AUDIO_CACHE_DIR` | Cached TTS-audio directory. |
| `VOICE_TEMP_DIR` | Temporary audio-normalization directory. |
| `KNOWLEDGE_MANIFEST_PATH` | Hash-verified knowledge manifest. |
| `PROMPT_REGISTRY_PATH` | Versioned prompt registry. |
| `MAX_AUDIO_BYTES` | Audio upload size limit. |
| `MAX_CONCURRENT_TURNS` | Maximum parallel voice-processing turns. |

Happify Backend should call this service through one `AI_SERVICE_BASE_URL`. Do not create separate voice and journal service URLs.

## Getting Started

### Prerequisites

- Python `3.11`
- FFmpeg for local execution
- Dependencies from `requirements.lock`
- A Faster-Whisper model or a build process that provides one
- Optional Ollama access for external LLM responses

### Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.lock
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

### Development

```bash
python main.py
```

The default local service URL is `http://localhost:8000`.

### Docker

```bash
docker build -t happify-ai .
docker run --env-file .env -p 8000:8000 happify-ai
```

For local Docker Compose:

```bash
docker compose up --build
```

## Verification

```bash
python -m py_compile main.py mood_analysis.py
```

Optional Docker verification:

```bash
docker build -t happify-ai:local .
```

## Safety Boundaries

- The service does not provide psychological diagnosis.
- Deterministic risk policy cannot be lowered by an LLM result.
- High-risk and crisis cases use safety-oriented responses.
- Raw images are not accepted by multimodal fusion.
- Knowledge sources are verified with SHA-256 at startup.
- Response audio is available only through authenticated routes.
- Never commit `AI_SERVICE_TOKEN`, provider keys, or local audio caches.

## Project Structure

```text
main.py                         # FastAPI application and runtime pipeline
mood_analysis.py                # Local mood fallback analysis
docker-entrypoint.sh            # Runtime user and Uvicorn entry point
Dockerfile                      # Production image and Whisper model setup
requirements.lock               # Locked Python dependencies
knowledge
|-- manifest.v1.json            # Knowledge manifest and source hashes
|-- sources                     # Governed support guidance
prompts
|-- registry.v1.json            # Versioned prompt registry
```
