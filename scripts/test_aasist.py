import argparse
import os
import sys
import time
from pathlib import Path

# Add project root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import soundfile as sf
from app.ai_voice.aasist_detector import AASISTDetector
from app.config import settings


def main():
    parser = argparse.ArgumentParser(description="AASIST AI Voice Detector Smoke Test")
    parser.add_argument("audio_path", type=str, help="Path to input audio file (.wav, .mp3, .flac)")
    parser.add_argument(
        "--model_path",
        type=str,
        default=settings.AASIST_MODEL_PATH,
        help="Path to AASIST model checkpoint (.pth)",
    )
    args = parser.parse_args()

    audio_file = args.audio_path
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found at '{audio_file}'")
        sys.exit(1)

    # Read audio metadata instantly with soundfile
    try:
        info = sf.info(audio_file)
        duration = info.duration
        sr = info.samplerate
    except Exception:
        duration = 0.0
        sr = 16000

    print("==================================================")
    print("AASIST INFERENCE TEST")
    print("==================================================")
    print(f"File: {os.path.basename(audio_file)}")
    print(f"Duration: {duration:.2f} s")
    print(f"Sample rate: {sr} Hz")
    print()

    detector = AASISTDetector(model_path=args.model_path, device="auto", threshold=settings.AASIST_THRESHOLD)

    start_time = time.perf_counter()
    result = detector.predict(audio_file, sample_rate=sr)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    print(f"Bonafide probability: {result['bonafide_probability']:.4f}")
    print(f"Spoof probability: {result['spoof_probability']:.4f}")
    print(f"Prediction: {result['prediction'].upper()}")
    print(f"Inference time: {latency_ms:.2f} ms")
    print("==================================================")


if __name__ == "__main__":
    main()
