import asyncio
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from zoneinfo import ZoneInfo

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


def format_datetime(value: Optional[str]) -> str:
    if not value:
        return "waktu yang diminta"
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.strftime("%H:%M")
    except Exception:
        return value

# ==========================================
# GATEWAY RESPONSE MODELS
# ==========================================
@dataclass
class VoiceRequestConfig:
    language: str
    tts_voice: str
    tts_rate: str
    enabled: bool
    elder_name: str
    timezone: str
    context: str

class EmotionData(BaseModel):
    state: str = "neutral"
    confidence: float = 0.0

class IntentData(BaseModel):
    name: str = "general"
    confidence: float = 0.0
    notify_caregiver: bool = False
    call_family: bool = False

class ReminderPlan(BaseModel):
    action: str = "none"
    title: Optional[str] = None
    message: Optional[str] = None
    due_at: Optional[str] = None
    dueAt: Optional[str] = None
    timezone: Optional[str] = None
    recurrence_rule: Optional[str] = None
    recurrenceRule: Optional[str] = None
    confidence: float = 0.0
    clarification_question: Optional[str] = None
    clarificationQuestion: Optional[str] = None

class LatencyBreakdown(BaseModel):
    audio_ms: float
    stt_ms: float
    response_ms: float
    emotion_ms: float
    reminder_ms: float
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
    intent: IntentData
    reminder: ReminderPlan
    latency_ms: float
    latency: LatencyBreakdown

class TestTTSRequest(BaseModel):
    text: Optional[str] = None

# ==========================================
# VOICE PROCESSOR (WHISPER + OLLAMA API + EDGE-TTS)
# ==========================================
class VoiceProcessor:
    def __init__(self) -> None:
        self.language = os.getenv("VOICE_LANGUAGE", "en")
        self.tts_voice = os.getenv("VOICE_TTS_VOICE", "en-US-JennyNeural")
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
            elder_name=request.headers.get("x-elder-name", "").strip(),
            timezone=request.headers.get("x-elder-timezone", "Asia/Jakarta").strip() or "Asia/Jakarta",
            context=request.headers.get("x-voice-context", "").strip(),
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
    async def response_for(self, transcript: str, language: str, elder_name: str = "", context: str = "") -> tuple[str, str]:
        text = self._normalize(transcript)
        if not text:
            return "I'm sorry, I didn't catch that. Could you please repeat slowly?", "fallback"
            
        if not OLLAMA_API_URL:
            return self._fallback_response_for(transcript, language), "fallback"

        try:
            name_instruction = f"The elder's preferred name is {elder_name}. Use it naturally, not in every sentence. " if elder_name else ""
            context_instruction = f"Relevant care context:\n{context}\n" if context else ""
            system_instruction = (
                "You are Eldora, a warm voice companion for elderly users. "
                f"{name_instruction}"
                f"{context_instruction}"
                "Speak warmly, clearly, and concisely in English or Indonesian based on the user's language. "
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

    # ── Layer 2: Intent Metrics (local rules) ─────────────────────────────────
    def analyze_intent(self, transcript: str, language: str) -> IntentData:
        text = self._normalize(transcript)
        if not text:
            return IntentData()

        call_family_terms_id = ["panggil anak", "hubungi anak", "telepon anak", "telpon anak", "panggil keluarga", "hubungi keluarga", "telepon keluarga", "telpon keluarga", "panggil caregiver", "hubungi caregiver", "panggil pengasuh", "hubungi pengasuh"]
        call_family_terms_en = ["call my child", "call my son", "call my daughter", "call my family", "contact my family", "call caregiver", "contact caregiver"]
        if any(term in text for term in call_family_terms_id + call_family_terms_en):
            return IntentData(name="call_family", confidence=0.95, notify_caregiver=True, call_family=True)

        if any(word in text for word in ["jatuh", "terpleset", "roboh", "fall", "fell", "fallen"]):
            return IntentData(name="fall_detected", confidence=0.9, notify_caregiver=True)
        if any(word in text for word in ["tolong", "bantuan", "darurat", "sakit", "sesak", "nyeri", "help", "emergency", "hurts", "pain", "can't breathe"]):
            return IntentData(name="help_request", confidence=0.86, notify_caregiver=True)
        if any(word in text for word in ["minum", "haus", "air", "water", "drink", "thirsty"]):
            return IntentData(name="water_request", confidence=0.78, notify_caregiver=True)
        if any(word in text for word in ["obat", "pil", "kapsul", "medicine", "medication", "pill"]):
            return IntentData(name="medicine_request", confidence=0.82, notify_caregiver=True)
        if any(word in text for word in ["kesepian", "takut", "sedih", "lonely", "scared", "afraid", "sad"]):
            return IntentData(name="emotional_support", confidence=0.75, notify_caregiver=False)
        return IntentData(name="general", confidence=0.5)

    # ── Layer 4: Reminder Planner (fast local + optional Ollama JSON) ─────────
    async def plan_reminder(self, transcript: str, language: str, timezone: str) -> ReminderPlan:
        text = self._normalize(transcript)
        if not self._looks_like_reminder_request(text):
            return ReminderPlan()

        local_plan = self._parse_local_reminder(text, timezone)
        if local_plan.action != "none":
            return local_plan

        if not OLLAMA_API_URL:
            return ReminderPlan(
                action="clarify_reminder",
                confidence=0.4,
                clarification_question="Jam berapa saya harus mengingatkan?",
            )

        try:
            prompt = (
                f'Transcript: "{transcript}"\n'
                f'Timezone: "{timezone}"\n'
                f'Current time: "{self._now(timezone).isoformat()}"\n\n'
                'Extract an elder reminder request. Respond JSON only:\n'
                '{"action":"none|create_reminder|clarify_reminder", "title":"short title", "message":"reminder message", '
                '"dueAt":"ISO-8601 datetime or null", "timezone":"timezone", "recurrenceRule":null, '
                '"confidence":0.0, "clarificationQuestion":"question or null"}'
            )
            payload = {
                "model": OLLAMA_MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You extract reminder commands for elderly care. Respond strictly in JSON."},
                    {"role": "user", "content": prompt},
                ],
                "options": {"temperature": 0.1, "num_predict": 100},
                "keep_alive": -1,
                "stream": False,
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(OLLAMA_API_URL, json=payload, headers=ollama_headers(), timeout=3.0)
                response.raise_for_status()
                data = response.json()
                result_text = data["message"]["content"].strip()
                match = re.search(r"\{.*?\}", result_text, re.DOTALL)
                if not match:
                    return ReminderPlan(action="clarify_reminder", confidence=0.4, clarification_question="Bisa ulangi pengingatnya untuk jam berapa?")
                parsed = json.loads(match.group())
                action = str(parsed.get("action", "none"))
                due_at = parsed.get("dueAt") or parsed.get("due_at")
                return ReminderPlan(
                    action=action,
                    title=parsed.get("title"),
                    message=parsed.get("message"),
                    due_at=due_at,
                    dueAt=due_at,
                    timezone=parsed.get("timezone") or timezone,
                    recurrence_rule=parsed.get("recurrenceRule"),
                    recurrenceRule=parsed.get("recurrenceRule"),
                    confidence=float(parsed.get("confidence", 0.5)),
                    clarification_question=parsed.get("clarificationQuestion"),
                    clarificationQuestion=parsed.get("clarificationQuestion"),
                )
        except Exception as e:
            log_event("ollama_reminder_failed", error=str(e), language=language, model=OLLAMA_MODEL_NAME)
            return ReminderPlan(action="clarify_reminder", confidence=0.4, clarification_question="Bisa ulangi pengingatnya untuk jam berapa?")

    def _looks_like_reminder_request(self, text: str) -> bool:
        terms = [
            "ingatkan", "ingetin", "remind me", "reminder", "jangan lupa", "nanti ingatkan",
        ]
        return any(term in text for term in terms)

    def _now(self, timezone: str) -> datetime:
        try:
            return datetime.now(ZoneInfo(timezone))
        except Exception:
            return datetime.now(ZoneInfo("Asia/Jakarta"))

    def _parse_local_reminder(self, text: str, timezone: str) -> ReminderPlan:
        now = self._now(timezone)
        recurrence_rule = None
        if any(term in text for term in ["setiap hari", "tiap hari", "daily", "every day"]):
            recurrence_rule = "FREQ=DAILY"

        match = re.search(r"(?:jam|pukul|at)\s*(\d{1,2})(?:[:.](\d{2}))?\s*(pagi|siang|sore|malam|am|pm)?", text)
        if not match:
            return ReminderPlan(action="clarify_reminder", confidence=0.55, clarification_question="Jam berapa saya harus mengingatkan?")

        hour = int(match.group(1))
        minute = int(match.group(2) or "0")
        period = match.group(3)
        if period in {"siang", "sore", "malam", "pm"} and hour < 12:
            hour += 12
        if period == "pagi" and hour == 12:
            hour = 0
        if not period and hour <= 7 and "nanti" in text and now.hour >= hour:
            hour += 12
        if hour > 23 or minute > 59:
            return ReminderPlan(action="clarify_reminder", confidence=0.45, clarification_question="Jamnya belum jelas. Bisa sebutkan ulang waktunya?")

        due = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if any(term in text for term in ["besok", "tomorrow"]):
            due += timedelta(days=1)
        elif due <= now and not recurrence_rule:
            due += timedelta(days=1)

        message = self._reminder_message_from_text(text)
        return ReminderPlan(
            action="create_reminder",
            title=message[:80],
            message=message,
            due_at=due.isoformat(),
            dueAt=due.isoformat(),
            timezone=timezone,
            recurrence_rule=recurrence_rule,
            recurrenceRule=recurrence_rule,
            confidence=0.82,
        )

    def _reminder_message_from_text(self, text: str) -> str:
        cleaned = re.sub(r"\b(eldora|tolong|please|ya)\b", " ", text)
        cleaned = re.sub(r"\b(ingatkan|ingetin|remind me|reminder|nanti|besok|setiap hari|tiap hari|daily|every day)\b", " ", cleaned)
        cleaned = re.sub(r"(?:jam|pukul|at)\s*\d{1,2}(?:[:.]\d{2})?\s*(?:pagi|siang|sore|malam|am|pm)?", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")
        return cleaned.capitalize() if cleaned else "Pengingat dari DoraBot"

    # ── Fallbacks & Helpers ───────────────────────────────────────────────────
    def _fallback_response_for(self, transcript: str, language: str) -> str:
        text = self._normalize(transcript)
        if not text:
            return "I'm sorry, I didn't catch that. Could you please repeat slowly?"

        if any(term in text for term in ["call my child", "call my son", "call my daughter", "call my family", "contact my family", "call caregiver", "contact caregiver"]):
            return "Okay, I will notify your family right away. Please stay calm."
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

        # Layer 1 + Layer 3 + Layer 4 run after transcript; Layer 2 stays local/sync
        intent = processor.analyze_intent(transcript, language)
        ai_start = time.perf_counter()

        async def timed_response():
            start_at = time.perf_counter()
            result = await processor.response_for(transcript, language, cfg.elder_name, cfg.context)
            return result, round((time.perf_counter() - start_at) * 1000, 2)

        async def timed_emotion():
            start_at = time.perf_counter()
            result = await processor.analyze_emotion(transcript, language)
            return result, round((time.perf_counter() - start_at) * 1000, 2)

        async def timed_reminder():
            start_at = time.perf_counter()
            result = await processor.plan_reminder(transcript, language, cfg.timezone)
            return result, round((time.perf_counter() - start_at) * 1000, 2)

        ((message, response_source), response_ms), (emotion, emotion_ms), (reminder, reminder_ms) = await asyncio.gather(
            timed_response(),
            timed_emotion(),
            timed_reminder(),
        )
        ai_ms = round((time.perf_counter() - ai_start) * 1000, 2)

        if intent.notify_caregiver:
            reminder = ReminderPlan()
        elif reminder.action == "clarify_reminder" and reminder.clarification_question:
            message = reminder.clarification_question
            response_source = "reminder_planner"
        elif reminder.action == "create_reminder" and reminder.due_at:
            name_prefix = f", {cfg.elder_name}" if cfg.elder_name else ""
            due_text = format_datetime(reminder.due_at)
            message = f"Baik{name_prefix}, saya catat ya. Nanti jam {due_text}, saya akan mengingatkan tentang {reminder.message or reminder.title or 'pengingat ini'}."
            response_source = "reminder_planner"

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
        response_ms=response_ms,
        emotion_ms=emotion_ms,
        reminder_ms=reminder_ms,
        ai_ms=ai_ms,
        tts_ms=tts_ms,
        total_ms=total_ms,
    )
    log_event(
        "voice_processing_completed",
        stt_ms=stt_ms,
        response_ms=response_ms,
        emotion_ms=emotion_ms,
        reminder_ms=reminder_ms,
        ai_ms=ai_ms,
        tts_ms=tts_ms,
        total_ms=total_ms,
        response_source=response_source,
        emotion=emotion.state,
        emotion_confidence=emotion.confidence,
        intent=intent.name,
        intent_confidence=intent.confidence,
        reminder_action=reminder.action,
        reminder_confidence=reminder.confidence,
        notify_caregiver=intent.notify_caregiver,
        call_family=intent.call_family,
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
        intent=intent,
        reminder=reminder,
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
