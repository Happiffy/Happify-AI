# Eldora Voice Service API Gateway

Eldora is an integrated, low-latency, privacy-first voice pipeline designed specifically as a safety, assistance, and recovery companion for the elderly population in Southeast Asia. 

This repository hosts the **Eldora Voice Gateway** (built with FastAPI). It orchestrates local Speech-to-Text (STT), conversational reasoning via Large Language Models (LLM) with custom emotion analysis, and Text-to-Speech (TTS) synthesis optimized for senior auditory and cognitive requirements.

---

## 🏗️ Pipeline Architecture

```
                                  +------------------------------+
                                  |   FastAPI Central Gateway    |
                                  |                              |
[🎙️ Senior Voice Input] ---------> |  1. Faster-Whisper STT (CPU) |
                                  |             |                |
                                  |   (Parallel Execution)       |
                                  |    /                  \      |
                                  |  2. LLM (Ollama)   3. Emotion|
                                  |    (Local GPU)      (Local)  |
                                  |    \                  /      |
                                  |             |                |
[🔊 Paced Speech Output] <-------- |  4. Edge-TTS Audio Cache     |
                                  +------------------------------+
```

1. **Layer 0: Audio STT (Local CPU)**: Dynamic English/Indonesian auto-transcription using a local, multi-threaded `Faster-Whisper` model (optimized for CPU with 4 execution threads).
2. **Layer 1: Conversational Response (Local GPU via Ollama)**: Empathic response generation powered by Qwen2.5 (configured for `qwen2.5:1.5b` or your fine-tuned `eldora-bot` model) offloaded to the local Ollama Windows daemon to utilize the GPU (e.g., NVIDIA GeForce RTX 3050).
3. **Layer 3: Emotion Metrics (Local GPU via Ollama)**: Unsupervised sentiment analysis determining the senior's emotional state (e.g., *calm, happy, sad, anxious, distressed*) run in parallel with response generation.
4. **TTS Speech Synthesis (Local Client)**: Natural-pacing `edge-tts` utilizing dynamic voice packs (warm Javanese-Indonesian female voice `"id-ID-GadisNeural"` or English `"en-US-JennyNeural"`) with speech rates slowed by **10%** for clear senior comprehension.
5. **Telemetry Logs**: Engagement tracking metrics written asynchronously to a local JSON log file.

---

## 📁 Repository Structure

| File / Folder | Purpose |
| :--- | :--- |
| [main.py](file:///C:/ai-stuff/dora-bot-backend/main.py) | Main FastAPI gateway server integrating Faster-Whisper, local Ollama, and Edge-TTS caching. |
| [test_suite.py](file:///C:/ai-stuff/dora-bot-backend/test_suite.py) | Diagnostics test client that evaluates transcription accuracy, cognitive load limits, latency SLAs, and exports metrics charts. |
| [test_cases.json](file:///C:/ai-stuff/dora-bot-backend/test_cases.json) | Database containing 10 representative mock dialogues and metadata for verification. |
| [convert_hf_to_gguf.py](file:///C:/ai-stuff/dora-bot-backend/convert_hf_to_gguf.py) | Utility script to convert Hugging Face PyTorch weights into GGUF format for Ollama import. |
| [requirements.txt](file:///C:/ai-stuff/dora-bot-backend/requirements.txt) | Python dependencies specifying exact package versions used in the environment. |
| [.gitignore](file:///C:/ai-stuff/dora-bot-backend/.gitignore) | Git ignore configurations for large binaries, runtime caches, and logs. |
| [old/](file:///C:/ai-stuff/dora-bot-backend/old/) | Archived scripts and documentation folder containing: <ul><li>`train_lora.py` (CPU-compatible LoRA training script)</li><li>`merge_and_export.py` (Model merging script for CPU/GPU)</li><li>`server_api.py` (Cloud Gemini API version)</li><li>`server_ollama.py` (Legacy server version)</li><li>Roadmaps and slide assets</li></ul> |

---

## 🚀 Installation & Setup

### 1. Environment Activation
Activate the pre-configured virtual environment and install the required dependencies:
```powershell
# Activate environment
.\env_dorabot\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Ollama (Local LLM Execution)
To bypass Python CUDA package limitations and leverage GPU-accelerated inference:
1. Download and install **[Ollama for Windows](https://ollama.com)**.
2. Run the Ollama daemon and pull the base model:
   ```powershell
   ollama run qwen2.5:1.5b
   ```

---

## 🏃 Running the Gateway Server

Start the FastAPI gateway:
```powershell
python main.py
```
The gateway will start on `http://127.0.0.1:8000`. It automatically handles:
- Audio uploads via `/api/process-audio`.
- TTS previews via `/api/test-tts`.
- Auto-caching of generated TTS speech under `audio_cache/`.
- Per-request voice switching through headers: `X-Voice-Language`, `X-Voice-TTS-Voice`, `X-Voice-Rate`, `X-Voice-Enabled`.
- Asynchronous logging of senior emotional status.

---

## 🚄 Railway Deployment

Railway uses the included `Dockerfile` and `railway.json`.

Required / recommended environment variables:
```env
PORT=8000
STT_MODEL_SIZE=base
STT_DEVICE=cpu
VOICE_LANGUAGE=id
VOICE_TTS_VOICE=id-ID-GadisNeural
VOICE_TTS_RATE=-10%
VOICE_AUDIO_CACHE_DIR=/app/audio_cache
OLLAMA_API_URL=https://your-ollama-service.example.com/api/chat
OLLAMA_API_KEY=
OLLAMA_MODEL_NAME=qwen2.5:1.5b
```

Notes:
- `OLLAMA_API_URL` must point to a reachable hosted Ollama-compatible gateway for AI responses on Railway.
- Set `OLLAMA_API_KEY` when the Ollama gateway requires bearer-token auth.
- If `OLLAMA_API_URL` is empty/unreachable, the service still runs with local STT + fallback responses + TTS.
- Backend should point to this Railway service:
```env
VOICE_AUDIO_PROCESSOR_URL=https://your-ai-eldora.up.railway.app/api/process-audio
VOICE_AUDIO_BASE_URL=https://your-ai-eldora.up.railway.app
```

---

## 🐳 Running with Docker

You can run the Eldora Voice Gateway inside a container. This handles all system dependencies (like `ffmpeg`) automatically.

### 1. Build and Start the Container
Use Docker Compose to build and start the service:
```bash
docker compose up --build
```
This starts the FastAPI gateway on `http://127.0.0.1:8000`.

### 2. Persistence & Caching
The [docker-compose.yml](file:///C:/ai-stuff/dora-bot-backend/docker-compose.yml) is configured to map two volumes to your host directory:
* `./audio_cache`: Persists synthesized Edge-TTS speech files.
* `./.hf_cache`: Persists downloaded Faster-Whisper models (so they aren't re-downloaded when the container restarts).

### 3. GPU Acceleration in Docker
* To run STT on the GPU inside Docker, the host machine must have the **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)** installed.
* The compose file includes GPU reservation parameters. If your host does not have a GPU or NVIDIA Docker runtime, simply remove the `deploy:` block from the [docker-compose.yml](file:///C:/ai-stuff/dora-bot-backend/docker-compose.yml) file.

---

## 🧪 Testing & Metrics Evaluation

Eldora includes an automated evaluation suite to verify performance against key quality of service guidelines:

1. Ensure target audio files (`1.wav` to `10.wav`) are placed in the `test_audio/` folder.
2. Run the test suite:
   ```powershell
   python test_suite.py
   ```

### Quality Metrics Checked:
- **STT Accuracy**: Normalized Levenshtein similarity comparing Faster-Whisper output with ground-truth transcripts.
- **Cognitive Sentence Constraint**: Checks that responses do not exceed **2 sentences** to prevent cognitive overload for senior users.
- **Latency SLAs**: Measures components breakdown (STT, LLM, Emotion Analysis, TTS) against the strict target **1.5s total latency SLA**.
- **Visualization Charts**: Automatically outputs diagnostic charts to the workspace root:
  - **[latency_metrics.png](file:///C:/ai-stuff/dora-bot-backend/latency_metrics.png)**: Bar chart illustrating latency per stage vs SLA limits.
  - **[accuracy_metrics.png](file:///C:/ai-stuff/dora-bot-backend/accuracy_metrics.png)**: Donut chart displaying transcription accuracy groups.

---

## 🏋️ Fine-Tuning Your Custom LoRA Adapter

To train a custom empathic adapter and deploy it locally:

1. **Train the LoRA Model**:
   ```powershell
   python old/train_lora.py
   ```
   *Note: Runs with GPU acceleration if PyTorch CUDA is available, otherwise automatically falls back to CPU-mode for a quick test run. Output weights save to `./eldercare_adapter`.*
2. **Merge Weights**:
   ```powershell
   python old/merge_and_export.py
   ```
   *Combines your fine-tuned weights with the base model to output a merged PyTorch directory in `./merged_eldercare_qwen2`.*
3. **Convert to GGUF**:
   ```powershell
   python convert_hf_to_gguf.py --model-dir ./merged_eldercare_qwen2 --output-file ./eldora_bot.gguf
   ```
4. **Register in Ollama**:
   Create a `Modelfile` containing:
   ```dockerfile
   FROM ./eldora_bot.gguf
   TEMPLATE "{{ .System }}\n<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant\n"
   SYSTEM "You are Eldora, a supportive and clear companion for seniors. Speak slowly, keep answers under 2 sentences, and show empathy."
   ```
   Build the model:
   ```powershell
   ollama create eldora-bot -f Modelfile
   ```
5. Update `OLLAMA_MODEL_NAME = "eldora-bot"` inside [main.py](file:///C:/ai-stuff/dora-bot-backend/main.py) to activate your custom model.

---

## 🛡️ Git Ignore Guidelines

To keep the repository clean and optimized for version control, the following files and folders are ignored via [.gitignore](file:///C:/ai-stuff/dora-bot-backend/.gitignore):

* **Virtual Environments**: `env_dorabot/` (Local Python runtime).
* **Model Files**: `*.gguf`, `lora_checkpoints/`, `eldercare_adapter/`, and `merged_eldercare_qwen2/` (Large binary model files and weights).
* **Caches**: `.hf_cache/` (Hugging Face cache directory) and `__pycache__/` (Python compiled files).
* **Runtime Assets / Outputs**:
  * `audio_cache/` (Edge-TTS cached voice files).
  * `temp_audio/` (FastAPI temporary incoming audio files).
  * `elder_input.wav` and `dorabot_output.mp3` (Local interactive audio files).
  * `wellness_signals_log.json` (Local telemetry data logs).
  * `test_audio/*.wav` (User-recorded speech files for the test suite, while preserving folder structure via `!test_audio/.gitkeep`).
* **Evaluation Metrics**: `latency_metrics.png` and `accuracy_metrics.png` (Charts generated locally during testing).
