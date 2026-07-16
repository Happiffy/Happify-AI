import asyncio
import os
import torch
import json
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from faster_whisper import WhisperModel
import edge_tts


BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct"
ADAPTER_DIR = "./eldercare_adapter"  
STT_MODEL_SIZE = "base"

# whisper in cpu & lower compute
stt_model = WhisperModel(STT_MODEL_SIZE, device="cpu", compute_type="int8")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, 
    quantization_config=bnb_config, 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

try:
    if os.path.exists(ADAPTER_DIR):
        print("found weights")
        model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    else:
        print("No adapter")
        model = base_model
except Exception as e:
    print(f"Could not load adapter: {e}. Defaulting to base framework.")
    model = base_model

print("All good")


# behavior
def send_wellness_signal(interaction_type: str, user_text: str, detected_language: str, confidence: float, trigger_detected: str = None):
    log_file = "wellness_signals_log.json"
    
    signal = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "interaction_type": interaction_type,
        "language": detected_language,
        "stt_confidence": round(confidence, 4),
        "text_length": len(user_text),
        "trigger_fired": trigger_detected,
        "raw_text_preview": user_text[:60] + "..." if len(user_text) > 60 else user_text
    }
    
    try:
        signals = []
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                try:
                    signals = json.load(f)
                except json.JSONDecodeError:
                    signals = []
        signals.append(signal)
        with open(log_file, "w") as f:
            json.dump(signals, f, indent=4)
        print(f"Behavioral wellness signal logged. (Total engagements: {len(signals)})")
    except Exception as e:
        print(f"Failed to record wellness signal: {e}")


# voice
def speech_to_text(audio_path: str) -> tuple[str, str, float]:
    if not os.path.exists(audio_path):
        print(f"audio input file not found -> {audio_path}")
        return "", "id", 0.0

    print(f"Processing audio -> {audio_path}")
    
    # b indo, coba bing juga
    segments, info = stt_model.transcribe(audio_path, beam_size=3) 
    
    transcript = " ".join([segment.text for segment in segments]).strip()
    detected_lang = info.language
    confidence = info.language_probability
    
    print(f"User Transcribed: '{transcript}' (Detected: '{detected_lang}', Confidence: {confidence:.2f})")
    
    if confidence < 0.7:
        print(f"STT confidence ({confidence:.2f}) is below 0.7!")
        if detected_lang != "en":
            print("Low confidence in primary language. Falling back to English ")
            detected_lang = "en"
            
    if detected_lang not in ["id", "en"]:
        detected_lang = "en"
        
    return transcript, detected_lang, confidence


def generate_empathic_response(user_text: str, language: str = "id") -> str:
    if language == "id":
        system_prompt = (
            "You are DoraBot, a gentle, companionable care assistant for elders. "
            "Speak warmly, clearly, and concisely in Bahasa Indonesia. "
            "If a physical emergency is stated (e.g., fall, severe pain), response text must strictly include 'TRIGGER: emergency_call'. "
            "If medication logs are updated or requested, response text must include 'TRIGGER: medication_log'. "
            "If the elder requests to contact their family or caregiver, response text must include 'TRIGGER: family_call'. "
            "Safety Guideline: Do not diagnose medical conditions, do not prescribe drugs, and do not offer professional medical counsel. "
            "Keep your tone patient, respectful, comforting, and companionable."
        )
    else:
        system_prompt = (
            "You are DoraBot, a gentle, companionable care assistant for elders. "
            "Speak warmly, clearly, and concisely in English or Indonesian based on the given sentence."
            "If a physical emergency is stated (e.g., fall, severe pain), response text must strictly include 'TRIGGER: emergency_call'. "
            "If medication logs are updated or requested, response text must include 'TRIGGER: medication_log'. "
            "If the elder requests to contact their family or caregiver, response text must include 'TRIGGER: family_call'. "
            "Safety Guideline: Do not diagnose medical conditions, do not prescribe drugs, and do not offer professional medical counsel. "
            "Keep your tone patient, respectful, comforting, and companionable."
        )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text}
    ]
    
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=128, 
        temperature=0.6,
        top_p=0.9
    )
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# tts
async def text_to_speech(text_target: str, output_path: str, language: str = "id"):    
    clean_text = (text_target
                  .replace("TRIGGER: emergency_call", "")
                  .replace("TRIGGER: medication_log", "")
                  .replace("TRIGGER: family_call", "")
                  .strip())
    
    if language == "id":
        voice = "id-ID-GadisNeural"
    else:
        voice = "en-US-JennyNeural"
    
    communicate = edge_tts.Communicate(clean_text, voice, rate="-10%")
    await communicate.save(output_path)
    print(f"done")

# final
async def run_dorabot_pipeline(input_audio_file: str, output_audio_file: str):
    user_transcript, detected_lang, confidence = speech_to_text(input_audio_file)
    if not user_transcript.strip():
        print("blank string")
        return

    ai_response = generate_empathic_response(user_transcript, detected_lang)
    print(f"Response: {ai_response}")

    active_trigger = None
    if "TRIGGER: emergency_call" in ai_response:
        active_trigger = "emergency_call"
        # TODO
    elif "TRIGGER: medication_log" in ai_response:
        active_trigger = "medication_log"
    elif "TRIGGER: family_call" in ai_response:
        active_trigger = "family_call"

    send_wellness_signal(
        interaction_type="voice_dialogue",
        user_text=user_transcript,
        detected_language=detected_lang,
        confidence=confidence,
        trigger_detected=active_trigger
    )

    await text_to_speech(ai_response, output_audio_file, detected_lang)

if __name__ == "__main__":
    sample_input = "elder_input.wav"
    sample_output = "dorabot_output.mp3"
    
    if not os.path.exists(sample_input):
        print(f"Tip: Please place a real audio file at '{sample_input}' for real transcription testing.")
    else:
        asyncio.run(run_dorabot_pipeline(sample_input, sample_output))