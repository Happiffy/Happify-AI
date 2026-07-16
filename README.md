# Happify AI Service

FastAPI voice service for local speech transcription, deterministic transcript risk rules, governed lexical retrieval, optional Ollama responses, recording-quality metrics, Edge TTS, and optional fusion of caller-supplied extracted camera observations.

This service provides supportive software behavior only. It is not a diagnostic system, emergency service, clinically validated product, or hardware validation tool. Recording-quality fields describe the uploaded recording and do not infer health conditions. The observation-fusion endpoint accepts structured outputs already extracted by caller-declared hardware/models; it does not accept raw images and does not provide or claim computer vision.

## Runtime flow

1. Authenticate `/api/*` with `Authorization: Bearer <AI_SERVICE_TOKEN>`.
2. Accept a bounded audio upload and normalize it with ffmpeg to mono 16 kHz PCM WAV.
3. Extract stdlib-only recording metrics: duration, peak/RMS dBFS, clipping ratio, silence ratio, DC offset, and descriptive quality flags.
4. Transcribe locally with Faster-Whisper.
5. Apply deterministic transcript risk and trigger rules before LLM processing.
6. Retrieve governed source entries with deterministic lexical matching and return structured citations.
7. Render prompts from a versioned registry and expose registry and prompt hashes.
8. Generate an English response with Ollama or local fallback behavior. Generated classification can raise risk but cannot lower deterministic severity.
9. Replace high/crisis generated text with deterministic safety wording and optionally synthesize TTS.

## Governed assets

`knowledge/manifest.v1.json` declares every allowed source file, source version, and SHA-256. Startup fails if metadata or hashes do not match. To update knowledge, add or version a source file, calculate its SHA-256, and update the manifest deliberately.

`prompts/registry.v1.json` contains prompt IDs, semantic versions, and templates. The service computes the registry hash and each template hash at startup and returns the applicable metadata with responses.

Retrieval is local lexical overlap only. Citations identify the manifest version, source ID/title/version, entry ID/title, and lexical score. Citations indicate which governed entries were supplied as context, not that generated wording is validated or clinically endorsed.

## API

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /health/live` | None | Process liveness |
| `GET /health/ready` | None | Readiness for auth, ffmpeg, governed assets, and STT |
| `GET /health` | None | Readiness-compatible alias |
| `POST /api/process-audio` | Bearer token | Canonical voice turn plus compatibility fields |
| `POST /api/analyze-journal` | Bearer token | English reflection with deterministic risk floor and citations |
| `POST /api/fuse-observations` | Bearer token | Fuse transcript safety with already-extracted camera observations |
| `POST /api/test-tts` | Bearer token | TTS preview |
| `GET /api/audio/{filename}` | Bearer token | Cached TTS audio |

`POST /api/process-audio` accepts `audio/wav`, `audio/x-wav`, `audio/mpeg`, `audio/mp4`, `audio/webm`, or `audio/ogg`. Useful headers are:

- `X-Request-ID`: optional caller correlation ID; validated or replaced.
- `X-Turn-ID`: optional turn correlation ID; validated or replaced.
- `X-Voice-Language`: `en` or `id` STT hint. User-facing output remains English.
- `X-Voice-TTS-Voice`: optional Edge TTS voice.
- `X-Voice-Rate`: signed percentage such as `-10%`.
- `X-Voice-Enabled`: set to `false` to skip TTS.
- `X-User-Name`: optional preferred name.
- `X-Voice-Context`: untrusted application context, explicitly isolated from system instructions.

The canonical response is versioned by `contract_version` and contains `request_id`, `turn_id`, typed `transcript`, `response`, `emotion`, `intent`, `risk_policy`, `recording_quality`, `citations`, `prompt`, and `latency`. Existing top-level fields such as `text`, `message`, `audio_url`, `audioUrl`, `response_source`, `responseSource`, and `latency_ms` remain for current consumers.

`POST /api/fuse-observations` accepts transcript text or an upstream transcript risk floor plus one or more already-extracted observations containing state, confidence, risk, face presence, eye contact, normalized expression probabilities, provider/model/version, and observation timestamp. Unknown fields are rejected, so raw image payloads are not part of the contract. Observation risk can raise final severity but can never lower the deterministic transcript floor. `CV_FUSION_MAX_OBSERVATIONS` bounds observations per request.

## Configuration

Copy `.env.example` to `.env` and set at least:

```env
AI_SERVICE_TOKEN=use-a-long-random-secret
STT_MODEL_SIZE=base
STT_DEVICE=auto
OLLAMA_API_URL=http://localhost:11434/api/chat
```

Ollama is optional; the service remains available with deterministic fallback responses. `AI_SERVICE_TOKEN`, ffmpeg, the prompt registry, the knowledge manifest, and the STT model are required for readiness.

Cache settings:

- `CACHE_CLEANUP_INTERVAL_SECONDS`: periodic cleanup interval.
- `TTS_CACHE_TTL_SECONDS`: inactivity lifetime for cached TTS files.
- `TEMP_FILE_TTL_SECONDS`: lifetime for stale normalization files.
- `VOICE_AUDIO_CACHE_DIR` and `VOICE_TEMP_DIR`: runtime directories. Railway should mount a persistent volume at `/data` and use `/data/audio_cache` for TTS files; temporary normalized audio should remain under `/tmp`.

## Railway release configuration

Set these values in the Railway service, not in Git:

- `AI_SERVICE_TOKEN`: long random token matching the backend `AI_SERVICE_TOKEN`.
- `VOICE_AUDIO_CACHE_DIR=/data/audio_cache`.
- `VOICE_TEMP_DIR=/tmp/happify`.
- `STT_MODEL_SIZE=base` unless a larger model is deliberately provisioned.
- `STT_MODEL_PATH=/models/whisper` for the Docker image's baked model.
- `STT_DEVICE=cpu` for a CPU Railway service.

Attach a persistent volume at `/data`. Without it, protected audio can disappear after a restart or redeploy. The service runs as the `happify` user; configure the Railway volume/runtime UID according to the platform permissions for non-root containers.

Startup performs cleanup, then a periodic task removes expired `tts_*.mp3` and `temp_*` files. Active files are uniquely named and request cleanup also runs in `finally`.

## Run

```bash
pip install -r requirements.txt
python main.py
```

Docker:

```bash
AI_SERVICE_TOKEN=use-a-long-random-secret docker compose up --build
```

The Docker image copies the governed knowledge and prompt assets and includes ffmpeg.

## Evaluation

Place the audio files referenced by `test_cases.json` under `test_audio/`, start the service, export the same token used by the service, then run:

```bash
AI_SERVICE_TOKEN=use-a-long-random-secret python test_suite.py
```

On PowerShell:

```powershell
$env:AI_SERVICE_TOKEN = "use-a-long-random-secret"
python test_suite.py
```

The suite sends the bearer token, unique request and turn IDs, disables TTS for repeatable evaluation, checks the canonical contract, measures transcription similarity and latency, and compares `intent.trigger` with every `expected_trigger`, including expected `null` values. Missing audio cases are reported as skipped and do not count as passes.

## Verification without test files

```bash
python -m py_compile main.py test_suite.py
python -m compileall -q main.py test_suite.py
```
