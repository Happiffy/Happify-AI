import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import site
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

for prefix in site.getsitepackages():
    nvidia_dir = Path(prefix) / "nvidia"
    if nvidia_dir.exists():
        for root, _, files in os.walk(nvidia_dir):
            if any(file.endswith(".dll") for file in files):
                try:
                    os.add_dll_directory(root)
                except Exception:
                    pass
                os.environ["PATH"] = f"{root};" + os.environ["PATH"]

import edge_tts
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from faster_whisper import WhisperModel
from pydantic import BaseModel

STT_MODEL_SIZE = os.getenv("STT_MODEL_SIZE", "base")
STT_DEVICE = os.getenv("STT_DEVICE", "auto").lower()
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "").strip()
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:1.5b")
AI_SERVICE_TOKEN = os.getenv("AI_SERVICE_TOKEN", "").strip()
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", "6291456"))
MAX_CONCURRENT_TURNS = int(os.getenv("MAX_CONCURRENT_TURNS", "2"))
ALLOWED_AUDIO_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp4",
    "audio/webm",
    "audio/ogg",
}
DEFAULT_VOICES = {"id": "id-ID-GadisNeural", "en": "en-US-JennyNeural"}

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("happify.ai")


def log_event(event: str, **fields) -> None:
    logger.info(json.dumps({"event": event, **fields}, default=str, ensure_ascii=False))


def ollama_headers() -> dict[str, str]:
    if not OLLAMA_API_KEY:
        return {}
    return {"Authorization": f"Bearer {OLLAMA_API_KEY}"}


@dataclass
class VoiceRequestConfig:
    language: str
    tts_voice: str
    tts_rate: str
    enabled: bool
    preferred_name: str
    context: str


class EmotionData(BaseModel):
    state: str = "neutral"
    confidence: float = 0.0
    risk_level: str = "low"
    requires_referral: bool = False


class IntentData(BaseModel):
    name: str = "general"
    confidence: float = 0.0
    requires_sos: bool = False
    requires_referral: bool = False


class LatencyBreakdown(BaseModel):
    audio_ms: float
    stt_ms: float
    response_ms: float
    emotion_ms: float
    ai_ms: float
    tts_ms: float
    total_ms: float


class ProcessAudioResponse(BaseModel):
    text: str
    message: str
    audio_url: Optional[str] = None
    audioUrl: Optional[str] = None
    language: str = "id"
    confidence: float = 0.0
    response_source: str
    responseSource: str
    emotion: EmotionData
    intent: IntentData
    latency_ms: float
    latency: LatencyBreakdown


class TestTTSRequest(BaseModel):
    text: Optional[str] = None


class AnalyzeJournalRequest(BaseModel):
    content: str
    language: str = "id"


class AnalyzeJournalResponse(BaseModel):
    reflection: str
    emotion: EmotionData
    suggested_action: str


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
            self.stt_model = WhisperModel(
                STT_MODEL_SIZE, device="cuda", compute_type="float16"
            )
            self.stt_device = "cuda"
            return
        if STT_DEVICE == "cpu":
            self.stt_model = WhisperModel(
                STT_MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=4
            )
            self.stt_device = "cpu"
            return
        try:
            self.stt_model = WhisperModel(
                STT_MODEL_SIZE, device="cuda", compute_type="float16"
            )
            self.stt_device = "cuda"
        except Exception as error:
            log_event("stt_cuda_failed", error=str(error))
            self.stt_model = WhisperModel(
                STT_MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=4
            )
            self.stt_device = "cpu"

    def config_from_request(self, request: Request) -> VoiceRequestConfig:
        return VoiceRequestConfig(
            language=request.headers.get("x-voice-language", self.language)
            if request.headers.get("x-voice-language", self.language) in {"id", "en"}
            else self.language,
            tts_voice="",
            tts_rate=self.tts_rate,
            enabled=request.headers.get("x-voice-enabled", "true").lower() != "false",
            preferred_name=request.headers.get("x-user-name", "").strip()[:80],
            context=request.headers.get("x-voice-context", "").strip()[:2000],
        )

    async def transcribe(
        self, audio_bytes: bytes, target_language: Optional[str] = None
    ) -> tuple[str, str, float]:
        if self.stt_model is None:
            raise RuntimeError("STT model is not initialized")
        temp_file = self.audio_cache_dir / f"temp_{time.time()}_transcribe.wav"
        with open(temp_file, "wb") as file:
            file.write(audio_bytes)
        try:
            loop = asyncio.get_running_loop()

            def run_whisper():
                segments, info = self.stt_model.transcribe(
                    str(temp_file), beam_size=3, language=target_language
                )
                transcript = " ".join([segment.text for segment in segments]).strip()
                return transcript, info.language, info.language_probability

            transcript, detected_language, confidence = await loop.run_in_executor(
                None, run_whisper
            )
            if detected_language not in ["id", "en"]:
                detected_language = "id"
            return transcript, detected_language, confidence
        finally:
            if temp_file.exists():
                temp_file.unlink()

    async def ollama_chat(
        self,
        system_instruction: str,
        user_content: str,
        *,
        temperature: float,
        num_predict: int,
        timeout: float,
    ) -> str:
        if not OLLAMA_API_URL:
            return ""
        payload = {
            "model": OLLAMA_MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content},
            ],
            "options": {"temperature": temperature, "num_predict": num_predict},
            "keep_alive": -1,
            "stream": False,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                OLLAMA_API_URL, json=payload, headers=ollama_headers(), timeout=timeout
            )
            response.raise_for_status()
            data = response.json()
            return self._normalize_response(data["message"]["content"] or "")

    async def response_for(
        self,
        transcript: str,
        language: str,
        preferred_name: str = "",
        context: str = "",
    ) -> tuple[str, str]:
        text = self._normalize(transcript)
        if not text:
            return (
                "Sorry, I did not catch that. Could you repeat it slowly?",
                "fallback",
            )
        if not OLLAMA_API_URL:
            return self._fallback_response_for(transcript), "fallback"
        name_instruction = (
            f"The user's preferred name is {preferred_name}. Use it naturally, not in every sentence. "
            if preferred_name
            else ""
        )
        context_instruction = f"Relevant app context:\n{context}\n" if context else ""
        system_instruction = (
            "You are Happify, a warm AI mental-health companion for teenagers and university students. "
            f"{name_instruction}{context_instruction}"
            "You are not a therapist and you must not diagnose. "
            "Respond in English. "
            "Use a friendly, non-judgmental, emotionally safe tone. "
            "Keep the response to 1 or 2 short sentences. "
            "If the user mentions self-harm, suicide, abuse, panic, or immediate danger, gently encourage contacting trusted people or emergency/professional help."
        )
        try:
            content = await self.ollama_chat(
                system_instruction,
                transcript,
                temperature=0.35,
                num_predict=70,
                timeout=15.0,
            )
            if content:
                return content[:360], "ollama"
        except Exception as error:
            log_event(
                "ollama_response_failed",
                error=str(error),
                language=language,
                model=OLLAMA_MODEL_NAME,
            )
        return self._fallback_response_for(transcript), "fallback"

    async def analyze_emotion(self, content: str, language: str) -> EmotionData:
        if not content.strip():
            return EmotionData()
        local_intent = self.analyze_intent(content, language)
        if not OLLAMA_API_URL:
            return EmotionData(
                state="distressed" if local_intent.requires_sos else "neutral",
                confidence=local_intent.confidence,
                risk_level="high" if local_intent.requires_referral else "low",
                requires_referral=local_intent.requires_referral,
            )
        prompt = (
            f'Analyze this mental-health expression: "{content}"\n'
            "Return JSON only with this exact shape: "
            '{"state":"calm|happy|neutral|sad|anxious|distressed","confidence":0.0,"risk_level":"low|medium|high|crisis","requires_referral":false}'
        )
        try:
            result_text = await self.ollama_chat(
                "You are a safety-focused mental-health emotion analysis agent. Return strict JSON only.",
                prompt,
                temperature=0.1,
                num_predict=90,
                timeout=10.0,
            )
            match = re.search(r"\{.*?\}", result_text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                return EmotionData(
                    state=str(parsed.get("state", "neutral")),
                    confidence=float(parsed.get("confidence", 0.5)),
                    risk_level=str(parsed.get("risk_level", "low")),
                    requires_referral=bool(parsed.get("requires_referral", False)),
                )
        except Exception as error:
            log_event(
                "ollama_emotion_failed",
                error=str(error),
                language=language,
                model=OLLAMA_MODEL_NAME,
            )
        return EmotionData(
            state="distressed" if local_intent.requires_sos else "neutral",
            confidence=local_intent.confidence,
            risk_level="high" if local_intent.requires_referral else "low",
            requires_referral=local_intent.requires_referral,
        )

    def analyze_intent(self, transcript: str, language: str) -> IntentData:
        text = self._normalize(transcript)
        if not text:
            return IntentData()
        crisis_terms = [
            "bunuh diri",
            "mengakhiri hidup",
            "mati aja",
            "self harm",
            "suicide",
            "kill myself",
            "end my life",
        ]
        panic_terms = [
            "panic",
            "panik",
            "sesak",
            "can't breathe",
            "tidak bisa napas",
            "cemas banget",
            "darurat",
            "tolong",
        ]
        loneliness_terms = [
            "kesepian",
            "sendirian",
            "lonely",
            "alone",
            "tak ada yang peduli",
            "nobody cares",
        ]
        if any(term in text for term in crisis_terms):
            return IntentData(
                name="crisis_risk",
                confidence=0.98,
                requires_sos=True,
                requires_referral=True,
            )
        if any(term in text for term in panic_terms):
            return IntentData(
                name="panic_or_emergency",
                confidence=0.88,
                requires_sos=True,
                requires_referral=False,
            )
        if any(term in text for term in loneliness_terms):
            return IntentData(
                name="loneliness_support",
                confidence=0.76,
                requires_sos=False,
                requires_referral=False,
            )
        return IntentData(name="general", confidence=0.5)

    async def generate_audio(
        self, text: str, cfg: Optional[VoiceRequestConfig] = None, language: str = "id"
    ) -> Optional[str]:
        if cfg and not cfg.enabled:
            return None
        clean_text = self._clean_for_tts(text)
        if not clean_text:
            return None
        header_voice = cfg.tts_voice if cfg else None
        voice = (
            header_voice
            or self.tts_voice
            or DEFAULT_VOICES.get(language, DEFAULT_VOICES["id"])
        )
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

    async def analyze_journal(
        self, content: str, language: str
    ) -> AnalyzeJournalResponse:
        emotion = await self.analyze_emotion(content, language)
        if OLLAMA_API_URL:
            system_instruction = (
                "You are Happify, a mental-health journaling reflection assistant. "
                "You are not a therapist and must not diagnose. "
                "Give a short, warm reflection in English, plus one gentle next step."
            )
            try:
                reflection = await self.ollama_chat(
                    system_instruction,
                    content,
                    temperature=0.35,
                    num_predict=130,
                    timeout=12.0,
                )
            except Exception as error:
                log_event("ollama_journal_failed", error=str(error))
                reflection = "It sounds like today meant a lot to you. Thank you for writing about it honestly."
        else:
            reflection = "It sounds like today meant a lot to you. Thank you for writing about it honestly."
        suggested_action = (
            "Open SOS and contact someone you trust now."
            if emotion.risk_level in {"high", "crisis"}
            else "Take one slow breath, then note one small thing you can do next."
        )
        return AnalyzeJournalResponse(
            reflection=reflection[:600],
            emotion=emotion,
            suggested_action=suggested_action,
        )

    def _fallback_response_for(self, transcript: str) -> str:
        text = self._normalize(transcript)
        if not text:
            return "I did not catch that. Could you repeat it slowly?"
        if any(
            term in text
            for term in [
                "bunuh diri",
                "mengakhiri hidup",
                "suicide",
                "kill myself",
                "end my life",
            ]
        ):
            return "I am concerned about your safety. Contact someone you trust or emergency services now; you do not have to face this alone."
        if any(term in text for term in ["panik", "panic", "sesak", "can't breathe"]):
            return "Let us slow down together. Breathe in for four counts, pause, then breathe out gently."
        if any(term in text for term in ["kesepian", "lonely", "sendirian", "alone"]):
            return "I am here with you. This feeling is heavy, but you do not have to face it alone."
        return "I hear you. We can take this one small step at a time."

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    def _normalize_response(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().strip('"')

    def _clean_for_tts(self, text: str) -> str:
        return re.sub(r"[^\w\s.,!?;:'\-()À-ɏ]", "", text).strip()


processor = VoiceProcessor()
turn_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TURNS)
app = FastAPI(title="Happify AI Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
    allow_methods=[],
    allow_headers=[],
)


@app.middleware("http")
async def request_logger(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        authorization = request.headers.get("authorization", "")
        provided_token = (
            authorization[7:] if authorization.startswith("Bearer ") else ""
        )
        if not AI_SERVICE_TOKEN or not secrets.compare_digest(
            provided_token, AI_SERVICE_TOKEN
        ):
            return Response(
                content='{"detail":"Unauthorized"}',
                status_code=401,
                media_type="application/json",
            )
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as error:
        log_event(
            "request_failed",
            method=request.method,
            path=request.url.path,
            duration_ms=round((time.perf_counter() - start) * 1000, 2),
            error=str(error),
        )
        raise
    log_event(
        "request_completed",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round((time.perf_counter() - start) * 1000, 2),
    )
    return response


@app.on_event("startup")
async def startup() -> None:
    log_event(
        "startup",
        stt_model=STT_MODEL_SIZE,
        stt_device=STT_DEVICE,
        ollama_configured=bool(OLLAMA_API_URL),
    )
    processor.load()
    log_event(
        "startup_ready",
        active_stt_device=processor.stt_device,
        default_language=processor.language,
        default_voice=processor.tts_voice,
    )


@app.get("/health")
async def health() -> dict:
    status = "not_configured"
    if OLLAMA_API_URL:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    OLLAMA_API_URL.split("/api")[0] + "/",
                    headers=ollama_headers(),
                    timeout=1.0,
                )
                status = (
                    "healthy"
                    if response.status_code == 200
                    else "unhealthy_ollama_not_responding"
                )
        except Exception:
            status = "ollama_not_reachable"
    return {
        "status": "healthy" if processor.stt_model is not None else "degraded",
        "ollama_status": status,
        "stt_model": f"Faster-Whisper ({processor.stt_device})",
        "llm_model": OLLAMA_MODEL_NAME,
    }


@app.post("/api/process-audio", response_model=ProcessAudioResponse)
async def process_audio(request: Request) -> ProcessAudioResponse:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported audio type")
    content_length = int(request.headers.get("content-length", "0") or 0)
    if content_length > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio stream is too large")
    cfg = processor.config_from_request(request)
    start = time.perf_counter()
    audio_bytes = await request.body()
    audio_ms = round((time.perf_counter() - start) * 1000, 2)
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio stream is too large")
    if len(audio_bytes) < 1000:
        raise HTTPException(status_code=400, detail="Audio stream is too short")
    try:
        await turn_semaphore.acquire()
        stt_start = time.perf_counter()
        transcript, language, confidence = await processor.transcribe(
            audio_bytes, cfg.language
        )
        stt_ms = round((time.perf_counter() - stt_start) * 1000, 2)
        intent = processor.analyze_intent(transcript, language)
        ai_start = time.perf_counter()

        async def timed_response():
            started = time.perf_counter()
            result = await processor.response_for(
                transcript, language, cfg.preferred_name, cfg.context
            )
            return result, round((time.perf_counter() - started) * 1000, 2)

        async def timed_emotion():
            started = time.perf_counter()
            result = await processor.analyze_emotion(transcript, language)
            return result, round((time.perf_counter() - started) * 1000, 2)

        (
            ((message, response_source), response_ms),
            (emotion, emotion_ms),
        ) = await asyncio.gather(timed_response(), timed_emotion())
        ai_ms = round((time.perf_counter() - ai_start) * 1000, 2)
        if intent.requires_referral and emotion.risk_level not in {"high", "crisis"}:
            emotion.risk_level = "high"
            emotion.requires_referral = True
        if emotion.risk_level == "crisis" or intent.name == "crisis_risk":
            message = "I am concerned about your safety. Contact someone you trust or emergency services now; you do not have to face this alone."
            response_source = "safety_policy"
        elif emotion.risk_level == "high":
            message = "This sounds very difficult. Contact someone you trust today, and use professional or SOS support if you feel unsafe."
            response_source = "safety_policy"
        tts_start = time.perf_counter()
        audio_url = await processor.generate_audio(message, cfg, language)
        tts_ms = round((time.perf_counter() - tts_start) * 1000, 2)
    except Exception as error:
        log_event(
            "voice_processing_failed", error=type(error).__name__, language=cfg.language
        )
        raise HTTPException(
            status_code=500, detail="Voice processing failed"
        ) from error
    finally:
        turn_semaphore.release()
    total_ms = round((time.perf_counter() - start) * 1000, 2)
    latency = LatencyBreakdown(
        audio_ms=audio_ms,
        stt_ms=stt_ms,
        response_ms=response_ms,
        emotion_ms=emotion_ms,
        ai_ms=ai_ms,
        tts_ms=tts_ms,
        total_ms=total_ms,
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
        latency_ms=total_ms,
        latency=latency,
    )


@app.post("/api/analyze-journal", response_model=AnalyzeJournalResponse)
async def analyze_journal(body: AnalyzeJournalRequest) -> AnalyzeJournalResponse:
    if not body.content.strip():
        raise HTTPException(status_code=400, detail="Journal content is required")
    return await processor.analyze_journal(body.content, body.language)


@app.post("/api/test-tts")
async def test_tts(request: Request, body: TestTTSRequest) -> dict:
    cfg = processor.config_from_request(request)
    text = (
        body.text or "Hello, I am Happify. I am here to support you, one step at a time."
    )
    audio_url = await processor.generate_audio(text, cfg, cfg.language)
    return {"audio_url": audio_url, "audioUrl": audio_url, "text": text}


@app.get("/api/audio/{filename}")
def get_audio(filename: str) -> FileResponse:
    if not re.fullmatch(r"tts_[a-f0-9]{12}\.mp3", filename):
        raise HTTPException(status_code=404, detail="Audio not found")
    cache_root = processor.audio_cache_dir.resolve()
    path = (cache_root / filename).resolve()
    if path.parent != cache_root or not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(str(path), media_type="audio/mpeg", filename=filename)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
