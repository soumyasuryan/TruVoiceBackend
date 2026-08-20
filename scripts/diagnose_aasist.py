import argparse
import os
import sys
import io
import wave
import numpy as np
import torch
import librosa

# Add project root directory to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.ai_voice.aasist_detector import AASISTDetector
from app.ai_voice.preprocessing import preprocess_audio_waveform
from app.config import settings


def pcm16_to_mulaw(pcm16_bytes: bytes) -> bytes:
    try:
        import audioop
        return audioop.lin2ulaw(pcm16_bytes, 2)
    except Exception:
        samples = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        sign = np.sign(samples)
        abs_s = np.abs(samples)
        mulaw = sign * np.log(1.0 + 255.0 * abs_s) / np.log(256.0)
        return (mulaw * 127.0).astype(np.int8).tobytes()


def mulaw_to_pcm16(mulaw_bytes: bytes) -> bytes:
    try:
        import audioop
        return audioop.ulaw2lin(mulaw_bytes, 2)
    except Exception:
        ulaw = ~np.frombuffer(mulaw_bytes, dtype=np.uint8)
        sign = ulaw & 0x80
        exponent = (ulaw >> 4) & 0x07
        mantissa = ulaw & 0x0F
        sample = ((mantissa << 3) + 132) << exponent
        sample = sample - 132
        sample = np.where(sign != 0, -sample, sample)
        return (sample * 4).astype(np.int16).tobytes()


def run_diagnostics(audio_path: str, model_path: str = None):
    model_file = model_path or settings.AASIST_MODEL_PATH
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at '{audio_path}'")
        sys.exit(1)

    print("==================================================")
    print("TRUVOICE AASIST PIPELINE DIAGNOSTIC AUDIT")
    print("==================================================")
    print(f"Audio File: {os.path.basename(audio_path)}")
    print(f"Model Checkpoint: {model_file}")

    detector = AASISTDetector(model_path=model_file, device="auto")

    audio_full, sr_orig = librosa.load(audio_path, sr=None, mono=True)
    duration_orig = len(audio_full) / sr_orig
    rms_orig = float(np.sqrt(np.mean(audio_full**2)))

    print("\n--- ORIGINAL AUDIO PROPERTIES ---")
    print(f"Sample Rate: {sr_orig} Hz")
    print(f"Channels: 1 (Mono)")
    print(f"Samples: {len(audio_full)}")
    print(f"Duration: {duration_orig:.3f} s")
    print(f"RMS Energy: {rms_orig:.6f}")
    print(f"Min Amplitude: {np.min(audio_full):.4f}")
    print(f"Max Amplitude: {np.max(audio_full):.4f}")

    # TEST A: Direct 16kHz Processing
    waveform_A, _ = preprocess_audio_waveform(audio_path, target_sample_rate=16000, target_num_samples=64600)
    waveform_A_dev = waveform_A.to(detector.device)
    with torch.no_grad():
        _, logits_A = detector.model(waveform_A_dev)
        probs_A = torch.softmax(logits_A, dim=1)
        # Verified ASVspoof 2019 class mapping: Class 0 = SPOOF (AI), Class 1 = BONAFIDE (Human)
        spoof_prob_A = float(probs_A[0, 0].item())
        human_prob_A = float(probs_A[0, 1].item())

    print("\n--------------------------------------------------")
    print("TEST A: DIRECT 16kHz AUDIO -> AASIST (Original)")
    print("--------------------------------------------------")
    print(f"AASIST Raw Logits [Class 0 (Spoof), Class 1 (Human)]: [{logits_A[0,0].item():.4f}, {logits_A[0,1].item():.4f}]")
    print(f"Human Probability (Class 1): {human_prob_A * 100:.2f}%")
    print(f"AI Probability    (Class 0): {spoof_prob_A * 100:.2f}%")
    print(f"Prediction: {'SPOOF / AI' if spoof_prob_A >= detector.threshold else 'BONAFIDE / HUMAN'}")

    # TEST B: Simulated Telephony Pipeline (8kHz mu-law -> 16kHz)
    audio_8k = librosa.resample(audio_full, orig_sr=sr_orig, target_sr=8000)
    pcm16_8k = (np.clip(audio_8k, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    mulaw_bytes = pcm16_to_mulaw(pcm16_8k)
    pcm16_decoded = mulaw_to_pcm16(mulaw_bytes)

    wav_buf = io.BytesIO()
    with wave.open(wav_buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(pcm16_decoded)

    waveform_B, _ = preprocess_audio_waveform(wav_buf.getvalue(), target_sample_rate=16000, target_num_samples=64600)
    waveform_B_dev = waveform_B.to(detector.device)
    with torch.no_grad():
        _, logits_B = detector.model(waveform_B_dev)
        probs_B = torch.softmax(logits_B, dim=1)
        spoof_prob_B = float(probs_B[0, 0].item())
        human_prob_B = float(probs_B[0, 1].item())

    print("\n--------------------------------------------------")
    print("TEST B: AGORA/TELEPHONY 8kHz MU-LAW -> 16kHz RESAMPLE -> AASIST")
    print("--------------------------------------------------")
    print(f"AASIST Raw Logits [Class 0 (Spoof), Class 1 (Human)]: [{logits_B[0,0].item():.4f}, {logits_B[0,1].item():.4f}]")
    print(f"Human Probability (Class 1): {human_prob_B * 100:.2f}%")
    print(f"AI Probability    (Class 0): {spoof_prob_B * 100:.2f}%")
    print(f"Prediction: {'SPOOF / AI' if spoof_prob_B >= detector.threshold else 'BONAFIDE / HUMAN'}")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AASIST TruVoice Pipeline Diagnostic Tool")
    parser.add_argument("audio_path", type=str, help="Path to input audio file")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model checkpoint")
    args = parser.parse_args()
    run_diagnostics(args.audio_path, args.model_path)
