# Happify AI Service

AI service untuk Happify Companion dan fitur kesehatan mental berbasis voice. Service ini menangani transkripsi audio, response percakapan, analisis mood, analisis risiko, analisis journal, dan optional text-to-speech.

---

## Overview

AI-Happify menerima request terautentikasi dari BE-Happify dan mengembalikan contract terstruktur untuk aplikasi Happify.

Service ini dirancang sebagai early-support system. AI membantu pengguna mengenali dan memahami kondisi emosional, tetapi bukan psikolog, psikiater, alat diagnosis, atau layanan emergency.

---

## Tech Stack

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
| Deployment | Docker, Railway |

---

## Features

- **Speech-to-Text** - mengubah audio pengguna menjadi transcript dengan Faster-Whisper.
- **AI Companion Response** - menghasilkan response suportif untuk percakapan voice.
- **Mood Analysis** - mendeteksi mood seperti calm, happy, neutral, sad, anxious, dan distressed.
- **Risk Policy** - menerapkan deterministic risk floor sebelum dan sesudah pemrosesan LLM.
- **Journal Analysis** - memberikan reflection, mood, risk level, dan suggested action dari journal.
- **Governed Knowledge** - memakai knowledge source lokal yang diverifikasi hash saat startup.
- **Versioned Prompts** - memakai prompt registry berversi dengan metadata hash.
- **Optional TTS** - membuat response audio dengan Edge TTS.
- **Recording Quality** - mengukur kualitas audio tanpa mengklaim kondisi klinis.
- **Multimodal Fusion Contract** - menerima hasil observasi kamera yang sudah diekstrak oleh caller.
- **Privacy Boundaries** - tidak menerima atau menyimpan raw image pada endpoint fusion.
- **Fallback Mode** - tetap dapat memberikan response dan mood analysis lokal tanpa Ollama.

---

## API Routes

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| `GET` | `/health/live` | None | Liveness process |
| `GET` | `/health/ready` | None | Readiness auth, ffmpeg, assets, dan STT |
| `GET` | `/health` | None | Alias readiness |
| `POST` | `/api/process-audio` | Bearer | Transkripsi, response, mood, risk, dan optional TTS |
| `POST` | `/api/analyze-journal` | Bearer | Journal reflection dan analysis |
| `POST` | `/api/fuse-observations` | Bearer | Fusion transcript dengan extracted observations |
| `GET` | `/api/audio/{filename}` | Bearer | Protected cached TTS audio |

---

## Request Flow

```text
User voice
    |
    v
BE-Happify
    |
    v
AI-Happify
    |
    +--> Normalize audio with FFmpeg
    +--> Transcribe with Faster-Whisper
    +--> Apply deterministic risk policy
    +--> Retrieve governed support context
    +--> Generate response or local fallback
    +--> Analyze mood and confidence
    +--> Optionally generate TTS audio
    |
    v
Structured response to BE-Happify
```

---

## Environment Variables

Buat file `.env` dari `.env.example`. Untuk Railway, set variables pada service AI-Happify.

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

AI_SERVICE_TOKEN=your_shared_ai_service_token
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
| `PORT` | Port HTTP service. Railway memakai port `8000` secara default. |
| `AI_SERVICE_TOKEN` | Bearer token yang harus sama persis dengan token pada BE-Happify. |
| `STT_MODEL_SIZE` | Ukuran model Faster-Whisper. `base` digunakan untuk MVP. |
| `STT_MODEL_PATH` | Lokasi model yang dibake ke Docker image. |
| `STT_DEVICE` | Device inference. Railway CPU memakai `cpu`. |
| `OLLAMA_API_URL` | Optional URL Ollama. Kosong berarti memakai fallback lokal. |
| `VOICE_AUDIO_CACHE_DIR` | Lokasi cache TTS. Railway memakai persistent volume `/data`. |
| `VOICE_TEMP_DIR` | Lokasi file normalisasi audio sementara. |
| `KNOWLEDGE_MANIFEST_PATH` | Manifest knowledge source yang diverifikasi hash. |
| `PROMPT_REGISTRY_PATH` | Registry prompt berversi. |
| `MAX_AUDIO_BYTES` | Batas ukuran audio upload. |
| `MAX_CONCURRENT_TURNS` | Batas concurrency voice processing. |

BE-Happify memanggil service ini dengan satu URL:

```env
AI_SERVICE_BASE_URL=https://happify-ai-production.up.railway.app
```

AI-Happify tidak membutuhkan `AI_VOICE_BASE_URL` atau `AI_JOURNAL_BASE_URL` karena URL tersebut adalah konfigurasi milik BE-Happify.

---

## Getting Started

### Prerequisites

- Python `3.11`
- FFmpeg untuk local run
- Dependency pada `requirements.lock`
- Model Faster-Whisper atau koneksi untuk download model saat build
- Optional Ollama jika ingin memakai LLM response eksternal

### Installation

```bash
pip install -r requirements.lock
```

### Development

```bash
python main.py
```

Service berjalan pada `http://localhost:8000` secara default.

### Docker

Build image:

```bash
docker build -t happify-ai .
```

Run container:

```bash
docker run --env-file .env -p 8000:8000 happify-ai
```

Local Docker Compose:

```bash
docker compose up --build
```

---

## Railway Deployment

Production AI service menggunakan Railway dengan Dockerfile dari branch `main`.

| Environment | URL |
| --- | --- |
| Local | `http://localhost:8000` |
| Production | `https://happify-ai-production.up.railway.app` |

Railway healthcheck memakai:

```text
GET /health/ready
```

Attach persistent volume pada `/data` agar cached TTS audio tidak hilang ketika service restart atau redeploy. Set `STT_DEVICE=cpu` untuk service Railway tanpa GPU.

---

## Verification

```bash
python -m py_compile main.py mood_analysis.py
```

Docker verification:

```bash
docker build -t happify-ai:local .
```

---

## Safety Boundaries

- AI tidak melakukan diagnosis psikologis.
- Risk policy deterministic tidak boleh diturunkan oleh hasil LLM.
- Kondisi high atau crisis memakai safety response yang deterministic.
- Raw image tidak diterima oleh endpoint multimodal fusion.
- Knowledge source diverifikasi dengan SHA-256 saat startup.
- Response audio hanya dapat diakses melalui route terautentikasi.
- `AI_SERVICE_TOKEN` tidak boleh ditulis ke repository atau README.

---

## Project Structure

```txt
main.py                         # FastAPI application dan runtime pipeline
mood_analysis.py                # Local mood fallback analysis
docker-entrypoint.sh            # Runtime user dan Uvicorn entrypoint
Dockerfile                      # Production image dan baked Whisper model
railway.json                    # Railway Docker dan healthcheck config
requirements.lock               # Python dependency lockfile
knowledge
|-- manifest.v1.json            # Knowledge manifest dan source hashes
|-- sources                     # Governed support guidance
prompts
|-- registry.v1.json            # Versioned prompt registry
```
