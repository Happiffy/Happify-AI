import os
import json
import time
import httpx
from typing import List, Dict, Any

# Target FastAPI Server URL
SERVER_URL = "http://127.0.0.1:8000"
PROCESS_AUDIO_ENDPOINT = f"{SERVER_URL}/api/process-audio"
TEST_CASES_FILE = "test_cases.json"

# ==========================================
# EVALUATION METRIC HELPER FUNCTIONS
# ==========================================

def calculate_levenshtein_distance(s1: str, s2: str) -> int:
    """Calculates Levenshtein Distance between two strings for accuracy mapping."""
    if len(s1) < len(s2):
        return calculate_levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def get_stt_accuracy(ref: str, hyp: str) -> float:
    """Computes Character-Level Transcription Accuracy."""
    ref_norm = re_normalize(ref)
    hyp_norm = re_normalize(hyp)
    if not ref_norm and not hyp_norm:
        return 1.0
    if not ref_norm or not hyp_norm:
        return 0.0
    dist = calculate_levenshtein_distance(ref_norm, hyp_norm)
    return max(0.0, 1.0 - (dist / max(len(ref_norm), len(hyp_norm))))


def re_normalize(text: str) -> str:
    """Helper to remove punctuation and lower text for clean comparison."""
    import re
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def count_sentences(text: str) -> int:
    """Counts the number of sentences in the response to evaluate cognitive load."""
    # Split by common sentence terminal symbols: ., ?, !
    sentences = [s for s in re_split(r"[.!?]+", text) if s.strip()]
    return len(sentences)


def re_split(pattern: str, text: str) -> List[str]:
    import re
    return re.split(pattern, text)


def safe_str(text: str) -> str:
    """Encodes a string to the console's encoding, replacing unmappable characters to avoid crashes."""
    import sys
    encoding = sys.stdout.encoding or "utf-8"
    try:
        return text.encode(encoding, errors="replace").decode(encoding)
    except Exception:
        return text.encode("ascii", errors="replace").decode("ascii")


# ==========================================
# MAIN TEST SUITE EXECUTION
# ==========================================

def run_evaluation():
    print(" Starting Eldora Voice Gateway Test Suite...")
    
    # 1. Load Test Cases
    if not os.path.exists(TEST_CASES_FILE):
        print(f" Error: Test database '{TEST_CASES_FILE}' not found. Please create it first.")
        return
        
    with open(TEST_CASES_FILE, "r") as f:
        test_cases: List[Dict[str, Any]] = json.load(f)
        
    print(f" Loaded {len(test_cases)} test case(s) from '{TEST_CASES_FILE}'.\n")
    
    # 2. Check server health
    print(f" Pinging FastAPI health check at {SERVER_URL}/health ...")
    try:
        res = httpx.get(f"{SERVER_URL}/health", timeout=3.0)
        if res.status_code == 200:
            print(f" Gateway Server is active. Engine details: {res.json()}\n")
        else:
            print(f" Health check returned status: {res.status_code}. Proceeding anyway.")
    except Exception as e:
        print(f" Connection error: Could not reach the server at {SERVER_URL}. Is it running?")
        print("Tip: Run the server first using: python main.py")
        return

    # Metrics registries
    stt_scores = []
    trigger_matches = []
    sentence_counts = []
    cognitive_compliances = []  # Strict <= 2 sentences constraint
    
    latencies = {
        "stt": [],
        "ai": [],
        "tts": [],
        "total": []
    }
    
    # 3. Process test cases
    for idx, case in enumerate(test_cases):
        audio_file = case.get("audio_file")
        ground_truth = case.get("ground_truth", "")
        expected_trigger = case.get("expected_trigger")
        expected_lang = case.get("expected_language", "en")
        
        print(f" [Test {idx + 1}] Processing file: {audio_file}")
        if not os.path.exists(audio_file):
            print(f"    Audio file '{audio_file}' not found. Skipping case.")
            continue
            
        # Send raw audio bytes to endpoint
        headers = {
            "x-voice-language": expected_lang,
            "x-voice-tts-voice": "id-ID-GadisNeural" if expected_lang == "id" else "en-US-JennyNeural"
        }
        
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
            
        try:
            # Make the HTTP POST call
            response = httpx.post(
                PROCESS_AUDIO_ENDPOINT, 
                content=audio_bytes, 
                headers=headers, 
                timeout=90.0
            )
            
            if response.status_code != 200:
                print(f"    Endpoint returned error {response.status_code}: {safe_str(response.text)}")
                continue
                
            data = response.json()
            
            # Extract returned values
            transcribed_text = data.get("text", "")
            response_msg = data.get("message", "")
            detected_lang = data.get("language", "en")
            emotion_state = data.get("emotion", {}).get("state", "neutral")
            
            # Extract latency parameters
            lat = data.get("latency", {})
            stt_ms = lat.get("stt_ms", 0.0)
            ai_ms = lat.get("ai_ms", 0.0)
            tts_ms = lat.get("tts_ms", 0.0)
            total_ms = data.get("latency_ms", 0.0)
            
            # EVALUATION 1: STT Transcription Accuracy
            stt_acc = get_stt_accuracy(ground_truth, transcribed_text)
            stt_scores.append(stt_acc)
            
            # EVALUATION 2: Trigger Accuracy
            # Note: For our gateway version, triggers are logged downstream,
            # but we can analyze trigger warnings printed in response or logged
            # Or in this suite, check if we parsed expected behavior in fallback/responses
            # (Mock checks or parsed keywords if any)
            # We check the trigger status in our test case
            actual_trigger = data.get("response_source", "gemini") # Or fallback
            trigger_success = True
            if expected_trigger:
                # If we expect a trigger, check if fallback was hit due to keywords
                # or if the response source was marked
                pass
            trigger_matches.append(trigger_success)
            
            # EVALUATION 3: Cognitive load sentence counts
            s_count = count_sentences(response_msg)
            sentence_counts.append(s_count)
            is_compliant = s_count <= 2
            cognitive_compliances.append(is_compliant)
            
            # Record Latencies
            latencies["stt"].append(stt_ms)
            latencies["ai"].append(ai_ms)
            latencies["tts"].append(tts_ms)
            latencies["total"].append(total_ms)
            
            # Print individual test case results
            print(f"     Transcript: '{safe_str(transcribed_text)}'")
            print(f"    Ground Truth: '{safe_str(ground_truth)}'")
            print(f"    STT Accuracy: {stt_acc * 100:.1f}%")
            print(f"    Response  : '{safe_str(response_msg)}' ({s_count} sentences, Compliant: {is_compliant})")
            print(f"    Emotion   : {safe_str(emotion_state)}")
            print(f"    Latencies : STT={stt_ms}ms | AI={ai_ms}ms | TTS={tts_ms}ms | Total={total_ms}ms")
            print("-" * 60)
            
        except Exception as e:
            print(f"    Request failed: {safe_str(str(e))}")
            print("-" * 60)

    # 4. Final Aggregated Evaluation Reports
    if not latencies["total"]:
        print(" No successful tests run. Cannot compile metrics.")
        return
        
    num_cases = len(latencies["total"])
    avg_stt_acc = (sum(stt_scores) / len(stt_scores)) * 100 if stt_scores else 0.0
    cognitive_pass_rate = (sum(1 for x in cognitive_compliances if x) / len(cognitive_compliances)) * 100 if cognitive_compliances else 0.0
    avg_sentence_len = sum(sentence_counts) / len(sentence_counts) if sentence_counts else 0.0
    
    # Calculate Latency KPIs
    avg_stt = sum(latencies["stt"]) / num_cases
    avg_ai = sum(latencies["ai"]) / num_cases
    avg_tts = sum(latencies["tts"]) / num_cases
    avg_total = sum(latencies["total"]) / num_cases
    sla_compliance = (sum(1 for t in latencies["total"] if t <= 1500.0) / num_cases) * 100
    
    print("\n============================================================")
    print(" ELDORA VOICE GATEWAY ACCURACY & EFFICIENCY METRICS REPORT")
    print("============================================================")
    print(f"Total Test Cases Processed: {num_cases}")
    print(f"STT Language Auto-Detection: Enabled (Bahasa Indonesia / English)")
    print("-" * 60)
    print(f" ACCURACY PERFORMANCE:")
    print(f"   - Mean STT Transcription Accuracy (Char-level): {avg_stt_acc:.2f}%")
    print(f"   - Seniors Cognitive Length Compliance (<= 2 sentences): {cognitive_pass_rate:.1f}% Pass Rate")
    print(f"   - Average Output Sentences: {avg_sentence_len:.2f}")
    print("-" * 60)
    print(f" LATENCY & EFFICIENCY (SLA Target: <= 1.5 seconds / 1500ms):")
    print(f"   - Mean Audio Stream Read Latency: {sum(latencies['total']) - sum(latencies['stt']) - sum(latencies['ai']) - sum(latencies['tts']):.2f} ms")
    print(f"   - Mean STT Processing Latency    : {avg_stt:.2f} ms")
    print(f"   - Mean LLM AI Generation Latency : {avg_ai:.2f} ms")
    print(f"   - Mean TTS Synthesis Latency     : {avg_tts:.2f} ms")
    print(f"   - Mean Total Pipeline Latency    : {avg_total:.2f} ms")
    print(f"   - Latency SLA Target Pass Rate   : {sla_compliance:.1f}%")
    print("============================================================\n")

    # Generate Charts
    generate_visualization(latencies, num_cases, stt_scores)

def generate_visualization(latencies: dict, num_cases: int, stt_scores: list):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 1. LATENCY METRICS DASHBOARD
        print(" Generating latency visualization chart (latency_metrics.png)...")
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        indices = np.arange(1, num_cases + 1)
        width = 0.4
        
        stt_data = np.array(latencies["stt"]) / 1000.0  # Convert to seconds
        ai_data = np.array(latencies["ai"]) / 1000.0
        tts_data = np.array(latencies["tts"]) / 1000.0
        
        p1 = ax1.bar(indices, stt_data, width, label='STT Latency (Faster-Whisper)', color='#4c72b0')
        p2 = ax1.bar(indices, ai_data, width, bottom=stt_data, label='AI Generation (Local LLM)', color='#c44e52')
        p3 = ax1.bar(indices, tts_data, width, bottom=stt_data+ai_data, label='TTS Synthesis (Edge-TTS)', color='#55a868')
        
        ax1.axhline(y=1.5, color='r', linestyle='--', linewidth=1.5, label='SLA Target (1.5s)')
        ax1.set_xlabel('Test Case Number', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Latency (Seconds)', fontsize=11, fontweight='bold')
        ax1.set_title('Pipeline Latency Breakdown per Test Case', fontsize=12, fontweight='bold', pad=10)
        ax1.set_xticks(indices)
        ax1.legend(loc='upper right')
        
        avg_stt = np.mean(stt_data)
        avg_ai = np.mean(ai_data)
        avg_tts = np.mean(tts_data)
        avg_total = np.mean(np.array(latencies["total"]) / 1000.0)
        
        categories = ['STT Avg', 'AI Avg', 'TTS Avg', 'Total Avg']
        averages = [avg_stt, avg_ai, avg_tts, avg_total]
        colors = ['#4c72b0', '#c44e52', '#55a868', '#8172b3']
        
        bars = ax2.barh(categories, averages, color=colors, height=0.5)
        ax2.axvline(x=1.5, color='r', linestyle='--', linewidth=1.5, label='SLA Target (1.5s)')
        
        for bar in bars:
            width = bar.get_width()
            ax2.text(width + 0.05, bar.get_y() + bar.get_height()/2, f'{width:.2f}s', 
                     va='center', ha='left', fontsize=10, fontweight='bold')
            
        ax2.set_xlabel('Latency (Seconds)', fontsize=11, fontweight='bold')
        ax2.set_title('Overall Average Latencies vs SLA Target', fontsize=12, fontweight='bold', pad=10)
        ax2.set_xlim(0, max(avg_total + 0.5, 2.0))
        ax2.legend(loc='lower right')
        
        plt.tight_layout()
        latency_image = "latency_metrics.png"
        plt.savefig(latency_image, dpi=150)
        plt.close()
        print(f" Latency visualization saved successfully as '{latency_image}'!")

        # 2. ACCURACY PIE CHART
        print(" Generating STT accuracy distribution chart (accuracy_metrics.png)...")
        # Classify the scores based on compliance bands
        perfect_count = sum(1 for s in stt_scores if s >= 0.999)
        high_count = sum(1 for s in stt_scores if 0.90 <= s < 0.999)
        acceptable_count = sum(1 for s in stt_scores if 0.70 <= s < 0.90)
        verify_count = sum(1 for s in stt_scores if s < 0.70)
        
        labels = []
        sizes = []
        colors = []
        
        if perfect_count > 0:
            labels.append(f'Perfect Match (100%): {perfect_count}')
            sizes.append(perfect_count)
            colors.append('#2ecc71')  # Flat Green
        if high_count > 0:
            labels.append(f'High Accuracy (90-99%): {high_count}')
            sizes.append(high_count)
            colors.append('#a3e4d7')  # Pale teal
        if acceptable_count > 0:
            labels.append(f'Acceptable (70-89%): {acceptable_count}')
            sizes.append(acceptable_count)
            colors.append('#f39c12')  # Flat Orange
        if verify_count > 0:
            labels.append(f'Needs Verification (<70%): {verify_count}')
            sizes.append(verify_count)
            colors.append('#e74c3c')  # Flat Red
            
        # Draw pie chart
        fig, ax = plt.subplots(figsize=(7, 7))
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=labels, 
            colors=colors, 
            autopct='%1.1f%%',
            startangle=140, 
            textprops=dict(color="black", weight="bold"),
            wedgeprops=dict(width=0.4, edgecolor='white') # Donut shape
        )
        
        ax.set_title('STT Audio Transcription Accuracy Distribution', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        accuracy_image = "accuracy_metrics.png"
        plt.savefig(accuracy_image, dpi=150)
        plt.close()
        print(f" Accuracy visualization saved successfully as '{accuracy_image}'!")
        
    except ImportError:
        print("\n Tip: To generate visualization graphs, install matplotlib:")
        print("   pip install matplotlib")
    except Exception as e:
        print(f" Failed to generate charts: {e}")

if __name__ == "__main__":
    run_evaluation()
