import json
import mimetypes
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx

BASE_DIR = Path(__file__).resolve().parent
SERVER_URL = os.getenv("AI_TEST_SERVER_URL", "http://127.0.0.1:8000").rstrip("/")
SERVICE_TOKEN = os.getenv("AI_SERVICE_TOKEN", "").strip()
PROCESS_AUDIO_ENDPOINT = f"{SERVER_URL}/api/process-audio"
TEST_CASES_FILE = Path(
    os.getenv("AI_TEST_CASES_FILE", str(BASE_DIR / "test_cases.json"))
)
CONTENT_TYPES = {
    ".mp3": "audio/mpeg",
    ".mp4": "audio/mp4",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
}


def calculate_levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        return calculate_levenshtein_distance(right, left)
    if not right:
        return len(left)
    previous_row = list(range(len(right) + 1))
    for index, left_character in enumerate(left):
        current_row = [index + 1]
        for offset, right_character in enumerate(right):
            current_row.append(
                min(
                    previous_row[offset + 1] + 1,
                    current_row[offset] + 1,
                    previous_row[offset] + (left_character != right_character),
                )
            )
        previous_row = current_row
    return previous_row[-1]


def normalize_text(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def stt_accuracy(reference: str, hypothesis: str) -> float:
    reference_normalized = normalize_text(reference)
    hypothesis_normalized = normalize_text(hypothesis)
    if not reference_normalized and not hypothesis_normalized:
        return 1.0
    if not reference_normalized or not hypothesis_normalized:
        return 0.0
    distance = calculate_levenshtein_distance(
        reference_normalized, hypothesis_normalized
    )
    return max(
        0.0,
        1.0 - distance / max(len(reference_normalized), len(hypothesis_normalized)),
    )


def count_sentences(text: str) -> int:
    return len([sentence for sentence in re.split(r"[.!?]+", text) if sentence.strip()])


def safe_text(value: object) -> str:
    text = str(value)
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


def content_type_for(path: Path) -> str:
    return CONTENT_TYPES.get(
        path.suffix.lower(), mimetypes.guess_type(path.name)[0] or "audio/wav"
    )


def run_evaluation() -> int:
    if not SERVICE_TOKEN:
        print("AI_SERVICE_TOKEN is required to run the authenticated evaluation suite.")
        return 2
    if not TEST_CASES_FILE.exists():
        print(f"Test case file not found: {TEST_CASES_FILE}")
        return 2
    with TEST_CASES_FILE.open("r", encoding="utf-8") as source:
        test_cases: list[dict[str, Any]] = json.load(source)
    try:
        health = httpx.get(f"{SERVER_URL}/health/ready", timeout=5.0)
        if health.status_code != 200:
            print(f"Service is not ready: {safe_text(health.text)}")
            return 2
    except httpx.HTTPError as error:
        print(f"Cannot reach the service: {safe_text(error)}")
        return 2
    stt_scores: list[float] = []
    trigger_results: list[bool] = []
    contract_results: list[bool] = []
    correlation_results: list[bool] = []
    sentence_results: list[bool] = []
    total_latencies: list[float] = []
    skipped = 0
    request_failures = 0
    for index, case in enumerate(test_cases, start=1):
        audio_path = (BASE_DIR / str(case.get("audio_file", ""))).resolve()
        expected_trigger = case.get("expected_trigger")
        if not audio_path.exists():
            skipped += 1
            print(f"[{index}] Skipped missing audio: {audio_path.name}")
            continue
        request_id = str(uuid.uuid4())
        turn_id = str(uuid.uuid4())
        headers = {
            "Authorization": f"Bearer {SERVICE_TOKEN}",
            "Content-Type": content_type_for(audio_path),
            "X-Request-ID": request_id,
            "X-Turn-ID": turn_id,
            "X-Voice-Language": str(case.get("expected_language", "en")),
            "X-Voice-Enabled": "false",
        }
        try:
            response = httpx.post(
                PROCESS_AUDIO_ENDPOINT,
                content=audio_path.read_bytes(),
                headers=headers,
                timeout=90.0,
            )
        except httpx.HTTPError as error:
            request_failures += 1
            print(f"[{index}] Request failed: {safe_text(error)}")
            continue
        if response.status_code != 200:
            request_failures += 1
            print(f"[{index}] HTTP {response.status_code}: {safe_text(response.text)}")
            continue
        data = response.json()
        transcript = str(data.get("transcript", {}).get("text", data.get("text", "")))
        message = str(data.get("response", {}).get("text", data.get("message", "")))
        intent = data.get("intent", {})
        actual_trigger = intent.get("trigger")
        trigger_matches = "trigger" in intent and actual_trigger == expected_trigger
        score = stt_accuracy(str(case.get("ground_truth", "")), transcript)
        sentence_count = count_sentences(message)
        sentence_matches = 1 <= sentence_count <= 2
        total_latency = float(data.get("latency", {}).get("total_ms", 0.0))
        identifiers_match = (
            data.get("request_id") == request_id and data.get("turn_id") == turn_id
        )
        contract_matches = data.get("contract_version") == "1.0.0"
        trigger_results.append(trigger_matches)
        contract_results.append(contract_matches)
        correlation_results.append(identifiers_match)
        stt_scores.append(score)
        sentence_results.append(sentence_matches)
        total_latencies.append(total_latency)
        print(
            f"[{index}] trigger={actual_trigger!r} expected={expected_trigger!r} "
            f"match={trigger_matches} stt={score * 100:.1f}% sentences={sentence_count} "
            f"ids={identifiers_match} contract={contract_matches} total_ms={total_latency:.2f}"
        )
    processed = len(total_latencies)
    if not processed:
        print("No audio cases were processed.")
        return 2
    trigger_pass_rate = sum(trigger_results) / len(trigger_results) * 100
    stt_mean = sum(stt_scores) / len(stt_scores) * 100
    sentence_pass_rate = sum(sentence_results) / len(sentence_results) * 100
    latency_mean = sum(total_latencies) / processed
    print(
        f"Processed={processed} skipped={skipped} request_failures={request_failures} "
        f"trigger_pass={trigger_pass_rate:.1f}% stt_mean={stt_mean:.1f}% "
        f"sentence_pass={sentence_pass_rate:.1f}% latency_mean_ms={latency_mean:.2f}"
    )
    failed = (
        request_failures
        or not all(trigger_results)
        or not all(contract_results)
        or not all(correlation_results)
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run_evaluation())
