import asyncio
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Add NVIDIA library paths to DLL search path and system PATH for CUDA 12 on Windows
import site
for prefix in site.getsitepackages():
    nvidia_dir = Path(prefix) / "nvidia"
    if nvidia_dir.exists():
        for root, dirs, files in os.walk(nvidia_dir):
            if any(f.endswith(".dll") for f in files):
                try:
                    os.add_dll_directory(root)
                except Exception:
                    pass
                os.environ["PATH"] = f"{root};" + os.environ["PATH"]

import httpx
from faster_whisper import WhisperModel
import edge_tts

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
STT_MODEL_SIZE = os.getenv("STT_MODEL_SIZE", "base")
STT_DEVICE = os.getenv("STT_DEVICE", "auto").lower()
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "").strip()
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:1.5b")
DEFAULT_VOICES = {
    "id": "id-ID-GadisNeural",
    "en": "en-US-JennyNeural",
}

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eldora.ai")


def log_event(event: str, **fields) -> None:
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, default=str, ensure_ascii=False))


def ollama_headers() -> dict[str, str]:
    if not OLLAMA_API_KEY:
        return {}
    return {"Authorization": f"Bearer {OLLAMA_API_KEY}"}

# ==========================================
# GATEWAY RESPONSE MODELS
# ==========================================
@dataclass
class VoiceRequestConfig:
    language: str
    tts_voice: str
    tts_rate: str
    enabled: bool

class EmotionData(BaseModel):
    state: str = "neutral"
    confidence: float = 0.0

class LatencyBreakdown(BaseModel):
    audio_ms: float
    stt_ms: float
    ai_ms: float
    tts_ms: float
    total_ms: float

class ProcessAudioResponse(BaseModel):
    text: str
    message: str
    audio_url: Optional[str] = None
    audioUrl: Optional[str] = None
    language: str = "en"
    confidence: float = 0.0
    response_source: str
    responseSource: str
    emotion: EmotionData
    latency_ms: float
    latency: LatencyBreakdown

class TestTTSRequest(BaseModel):
    text: Optional[str] = None

# ==========================================
# VOICE PROCESSOR (WHISPER + OLLAMA API + EDGE-TTS)
# ==========================================
class VoiceProcessor:
    def __init__(self) -> None:
        self.language = os.getenv("VOICE_LANGUAGE", "id")
        self.tts_voice = os.getenv("VOICE_TTS_VOICE", "id-ID-GadisNeural")
        self.tts_rate = os.getenv("VOICE_TTS_RATE", "-10%")
        self.audio_cache_dir = Path(os.getenv("VOICE_AUDIO_CACHE_DIR", "./audio_cache"))
        self.audio_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.stt_model = None
        self.stt_device = "cpu"

    def load(self) -> None:
        if STT_DEVICE in {"cuda", "gpu"}:
            log_event("stt_load_start", device="cuda", model=STT_MODEL_SIZE)
            self.stt_model = WhisperModel(STT_MODEL_SIZE, device="cuda", compute_type="float16")
            self.stt_device = "cuda"
            log_event("stt_load_success", device="cuda", model=STT_MODEL_SIZE)
            return

        if STT_DEVICE == "cpu":
            log_event("stt_load_start", device="cpu", model=STT_MODEL_SIZE)
            self.stt_model = WhisperModel(STT_MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=4)
            self.stt_device = "cpu"
            log_event("stt_load_success", device="cpu", model=STT_MODEL_SIZE)
            return

        try:
            log_event("stt_load_start", device="cuda", model=STT_MODEL_SIZE)
            self.stt_model = WhisperModel(STT_MODEL_SIZE, device="cuda", compute_type="float16")
            self.stt_device = "cuda"
            log_event("stt_load_success", device="cuda", model=STT_MODEL_SIZE)
        except Exception as e:
            log_event("stt_load_failed", device="cuda", model=STT_MODEL_SIZE, error=str(e))
            log_event("stt_load_start", device="cpu", model=STT_MODEL_SIZE)
            self.stt_model = WhisperModel(STT_MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=4)
            self.stt_device = "cpu"
            log_event("stt_load_success", device="cpu", model=STT_MODEL_SIZE)

    def config_from_request(self, request: Request) -> VoiceRequestConfig:
        return VoiceRequestConfig(
            language=request.headers.get("x-voice-language", self.language),
            tts_voice=request.headers.get("x-voice-tts-voice", ""),
            tts_rate=request.headers.get("x-voice-rate", self.tts_rate),
            enabled=request.headers.get("x-voice-enabled", "true").lower() != "false",
        )

    # ── Layer 0: STT (Local Faster-Whisper) ───────────────────────────────────
    async def transcribe(self, audio_bytes: bytes, target_language: Optional[str] = None) -> tuple[str, str, float]:
        if self.stt_model is None:
            raise RuntimeError("STT model is not initialized")
            
        temp_file = self.audio_cache_dir / f"temp_{time.time()}_transcribe.wav"
        with open(temp_file, "wb") as f:
            f.write(audio_bytes)
            
        try:
            loop = asyncio.get_running_loop()
            def run_whisper():
                # Pass the explicit target language if provided to avoid detection errors
                segments, info = self.stt_model.transcribe(str(temp_file), beam_size=3, language=target_language)
                transcript = " ".join([segment.text for segment in segments]).strip()
                return transcript, info.language, info.language_probability

            transcript, detected_lang, confidence = await loop.run_in_executor(None, run_whisper)
            
            if confidence < 0.7:
                log_event("stt_low_confidence", confidence=round(confidence, 4), detected_language=detected_lang)
                if detected_lang != "en":
                    detected_lang = "en"
                    
            if detected_lang not in ["id", "en"]:
                detected_lang = "en"
                
            return transcript, detected_lang, confidence
        finally:
            if temp_file.exists():
                temp_file.unlink()

    # ── Layer 1: Conversational Response (Ollama API) ────────────────────────
    async def response_for(self, transcript: str, language: str) -> tuple[str, str]:
        text = self._normalize(transcript)
        if not text:
            fallback = "Maaf, saya tidak mendengar dengan jelas. Bisa diulangi?" if language == "id" else "I'm sorry, I didn't catch that. Could you please repeat slowly?"
            return fallback, "fallback"
            
        if not OLLAMA_API_URL:
            return self._fallback_response_for(transcript, language), "fallback"

        try:
            # Set system prompts based on language
            if language == "id":
                system_instruction = (
                    "You are Eldora, a warm voice companion for elderly users. "
                    "Speak warmly, clearly, and concisely in Bahasa Indonesia. "
                    "You must respond in EXACTLY 1 or 2 short sentences. Do NOT write more than 2 sentences under any circumstance. "
                    "If there is an emergency, a fall, pain, or a request for help, reassure the user and let them know their caregiver will be notified."
                )
            else:
                system_instruction = (
                    "You are Eldora, a warm voice companion for elderly users. "
                    "Speak warmly, clearly, and concisely in English. "
                    "You must respond in EXACTLY 1 or 2 short sentences. Do NOT write more than 2 sentences under any circumstance. "
                    "If there is an emergency, a fall, pain, or a request for help, reassure the user and let them know their caregiver will be notified."
                )
                
            payload = {
                "model": OLLAMA_MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": transcript}
                ],
                "options": {
                    "temperature": 0.4,
                    "num_predict": 50
                },
                "keep_alive": -1,
                "stream": False
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(OLLAMA_API_URL, json=payload, headers=ollama_headers(), timeout=15.0)
                response.raise_for_status()
                data = response.json()
                content = self._normalize_response(data["message"]["content"] or "")
                if content:
                    return content[:260], "ollama_qwen"
                    
        except Exception as e:
            log_event("ollama_response_failed", error=str(e), language=language, model=OLLAMA_MODEL_NAME)
            
        return self._fallback_response_for(transcript, language), "fallback"

    # ── Layer 3: Emotion Metrics (Ollama API) ─────────────────────────
    async def analyze_emotion(self, transcript: str, language: str) -> EmotionData:
        if not transcript.strip() or not OLLAMA_API_URL:
            return EmotionData(state="neutral", confidence=0.0)
            
        try:
            prompt = (
                f'Analyze the emotional state expressed in this speech transcript: "{transcript}"\n\n'
                'Respond with a JSON object only (no markdown, no explanation):\n'
                '{"state": "<calm|happy|sad|anxious|distressed>", "confidence": <0.0-1.0>}'
            )
            
            payload = {
                "model": OLLAMA_MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are a professional emotion analysis agent. Respond strictly in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                "options": {
                    "temperature": 0.1,
                    "num_predict": 45
                },
                "keep_alive": -1,
                "stream": False
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(OLLAMA_API_URL, json=payload, headers=ollama_headers(), timeout=10.0)
                response.raise_for_status()
                data = response.json()
                result_text = data["message"]["content"].strip()
                
                match = re.search(r"\{.*?\}", result_text, re.DOTALL)
                if match:
                    data_json = json.loads(match.group())
                    return EmotionData(
                        state=str(data_json.get("state", "neutral")),
                        confidence=float(data_json.get("confidence", 0.5)),
                    )
        except Exception as e:
            log_event("ollama_emotion_failed", error=str(e), language=language, model=OLLAMA_MODEL_NAME)
            
        return EmotionData(state="neutral", confidence=0.5)

    # ── Local TTS (Edge-TTS) ──────────────────────────────────────────────────
    async def generate_audio(self, text: str, cfg: Optional[VoiceRequestConfig] = None, language: str = "id") -> Optional[str]:
        if cfg and not cfg.enabled:
            return None

        clean_text = self._clean_for_tts(text)
        if not clean_text:
            return None
            
        header_voice = cfg.tts_voice if cfg else None
        configured_default = self.tts_voice or DEFAULT_VOICES.get(language, DEFAULT_VOICES["en"])
        voice = header_voice or configured_default
        if not header_voice and language in DEFAULT_VOICES:
            voice = DEFAULT_VOICES[language]
            
        rate = cfg.tts_rate if cfg else self.tts_rate
        
        cache_key = f"{voice}_{rate}_{clean_text}"
        filename = f"tts_{hashlib.md5(cache_key.encode()).hexdigest()[:12]}.mp3"
        path = self.audio_cache_dir / filename
        
        if not path.exists():
            communicate = edge_tts.Communicate(clean_text, voice, rate=rate)
            await communicate.save(str(path))
            
        return f"/api/audio/{filename}"

    # ── Fallbacks & Helpers ───────────────────────────────────────────────────
    def _fallback_response_for(self, transcript: str, language: str) -> str:
        text = self._normalize(transcript)
        if not text:
            return "Maaf, saya tidak mendengar dengan jelas. Bisa diulangi?" if language == "id" else "I'm sorry, I didn't catch that. Could you please repeat slowly?"
            
        if language == "id":
            if any(word in text for word in ["jatuh", "terpleset", "roboh"]):
                return "Saya akan segera menghubungi pengasuh Anda. Harap tetap tenang dan jangan banyak bergerak."
            if any(word in text for word in ["tolong", "bantuan", "sakit", "sesak", "nyeri"]):
                return "Saya akan segera memberitahu pengasuh Anda. Harap tetap tenang, bantuan sedang dikirim."
            if any(word in text for word in ["minum", "haus", "air"]):
                return "Baik, saya akan memberitahu pengasuh Anda bahwa Anda memerlukan air minum."
            if any(word in text for word in ["obat", "pil", "kapsul"]):
                return "Baik, saya akan menyampaikan permintaan obat Anda kepada pengasuh."
            if any(word in text for word in ["kesepian", "takut", "sedih"]):
                return "Saya di sini menemani Anda. Tarik napas perlahan, Anda tidak sendirian."
            return "Saya mendengar Anda. Saya akan meneruskan kebutuhan Anda kepada pengasuh jika diperlukan."
        else:
            if any(word in text for word in ["fall", "fell", "fallen"]):
                return "I am contacting your caregiver right now. Please stay calm and try not to move too much."
            if any(word in text for word in ["help", "emergency", "hurts", "pain", "can't breathe"]):
                return "I will notify your caregiver immediately. Please stay calm, help is on the way."
            if any(word in text for word in ["water", "drink", "thirsty"]):
                return "Got it, I will let your caregiver know that you need some water."
            if any(word in text for word in ["medicine", "medication", "pill"]):
                return "Got it, I will pass your medicine request to your caregiver."
            if any(word in text for word in ["lonely", "scared", "afraid", "sad"]):
                return "I am right here with you. Take a slow breath, you are not alone."
            return "I hear you. I will help pass your needs to your caregiver if needed."

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    def _normalize_response(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().strip('"')

    def _clean_for_tts(self, text: str) -> str:
        return re.sub(r"[^\w\s.,!?;:'\-()À-ɏ]", "", text).strip()

# ==========================================
# FASTAPI GATEWAY ENGINE
# ==========================================
processor = VoiceProcessor()
app = FastAPI(title="Eldora Voice Service (Ollama Gateway)", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logger(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as error:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log_event("request_failed", method=request.method, path=request.url.path, duration_ms=duration_ms, error=str(error))
        raise

    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    log_event("request_completed", method=request.method, path=request.url.path, status=response.status_code, duration_ms=duration_ms)
    return response


@app.on_event("startup")
async def startup() -> None:
    log_event("startup", stt_model=STT_MODEL_SIZE, stt_device=STT_DEVICE, ollama_configured=bool(OLLAMA_API_URL), ollama_api_key_configured=bool(OLLAMA_API_KEY))
    processor.load()
    log_event("startup_ready", active_stt_device=processor.stt_device, default_language=processor.language, default_voice=processor.tts_voice)

@app.get("/health")
async def health() -> dict:
    status = "not_configured"
    if OLLAMA_API_URL:
        try:
            async with httpx.AsyncClient() as client:
                res = await client.get(OLLAMA_API_URL.split("/api")[0] + "/", headers=ollama_headers(), timeout=1.0)
                if res.status_code == 200:
                    status = "healthy"
                else:
                    status = "unhealthy_ollama_not_responding"
        except Exception:
            status = "ollama_not_reachable"
        
    return {
        "status": "healthy" if processor.stt_model is not None and status == "healthy" else "degraded",
        "ollama_status": status,
        "stt_model": f"Faster-Whisper (Local {processor.stt_device.upper()})",
        "llm_model": f"Ollama ({OLLAMA_MODEL_NAME})",
        "language": processor.language,
    }

@app.post("/api/process-audio", response_model=ProcessAudioResponse)
async def process_audio(request: Request) -> ProcessAudioResponse:
    cfg = processor.config_from_request(request)
    start = time.perf_counter()
    audio_bytes = await request.body()
    audio_ms = round((time.perf_counter() - start) * 1000, 2)

    if len(audio_bytes) < 1000:
        log_event("voice_rejected", reason="audio_too_short", bytes=len(audio_bytes), language=cfg.language)
        raise HTTPException(status_code=400, detail="Audio stream is too short")

    log_event("voice_processing_started", bytes=len(audio_bytes), language=cfg.language, voice=cfg.tts_voice or processor.tts_voice, rate=cfg.tts_rate, voice_enabled=cfg.enabled)

    try:
        # Layer 0: STT — must complete before response generation
        stt_start = time.perf_counter()
        transcript, language, confidence = await processor.transcribe(audio_bytes, cfg.language)
        stt_ms = round((time.perf_counter() - stt_start) * 1000, 2)

        # Layer 1 + Layer 3 run in parallel
        ai_start = time.perf_counter()
        (message, response_source), emotion = await asyncio.gather(
            processor.response_for(transcript, language),
            processor.analyze_emotion(transcript, language),
        )
        ai_ms = round((time.perf_counter() - ai_start) * 1000, 2)

        # TTS Synthesis — use per-request settings and pass language for auto voice packs
        tts_start = time.perf_counter()
        audio_url = await processor.generate_audio(message, cfg, language)
        tts_ms = round((time.perf_counter() - tts_start) * 1000, 2)

    except Exception as error:
        log_event("voice_processing_failed", error=str(error), language=cfg.language, voice=cfg.tts_voice or processor.tts_voice)
        raise HTTPException(status_code=500, detail=str(error)) from error

    total_ms = round((time.perf_counter() - start) * 1000, 2)
    latency = LatencyBreakdown(
        audio_ms=audio_ms,
        stt_ms=stt_ms,
        ai_ms=ai_ms,
        tts_ms=tts_ms,
        total_ms=total_ms,
    )
    log_event(
        "voice_processing_completed",
        stt_ms=stt_ms,
        ai_ms=ai_ms,
        tts_ms=tts_ms,
        total_ms=total_ms,
        response_source=response_source,
        emotion=emotion.state,
        emotion_confidence=emotion.confidence,
        language=language,
        confidence=confidence,
        voice=cfg.tts_voice or processor.tts_voice,
        audio_cached=bool(audio_url),
    )
    return ProcessAudioResponse(
        text=transcript,
        message=message,
        audio_url=audio_url,
        audioUrl=audio_url,
        language=language,
        confidence=confidence,
        response_source=response_source,
        responseSource=response_source,
        emotion=emotion,
        latency_ms=total_ms,
        latency=latency,
    )

@app.post("/api/test-tts")
async def test_tts(request: Request, body: TestTTSRequest) -> dict:
    cfg = processor.config_from_request(request)
    text = body.text or "Hello! I am Eldora, your voice companion. I am here whenever you need me."
    audio_url = await processor.generate_audio(text, cfg, cfg.language)
    log_event("tts_preview_generated", language=cfg.language, voice=cfg.tts_voice or processor.tts_voice, rate=cfg.tts_rate, audio_cached=bool(audio_url))
    return {"audio_url": audio_url, "audioUrl": audio_url, "text": text}

@app.get("/api/audio/{filename}")
def get_audio(filename: str) -> FileResponse:
    path = processor.audio_cache_dir / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(str(path), media_type="audio/mpeg", filename=filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
