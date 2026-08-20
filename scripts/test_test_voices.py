import argparse
import os
import sys
import time

# Add project root directory to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.ai_voice.aasist_detector import AASISTDetector
from app.utils.pipeline import get_pipeline
from app.config import settings


def test_voice_folder(folder_path: str, model_path: str = None):
    target_model = model_path or settings.AASIST_MODEL_PATH
    if not os.path.exists(folder_path):
        print(f"Error: Directory '{folder_path}' does not exist.")
        sys.exit(1)

    print("==========================================================================================")
    print("TRUVOICE AI VOICE DETECTION — MULTI-SAMPLE EVALUATION RUNNER")
    print("==========================================================================================")
    print(f"Target Directory : {folder_path}")
    print(f"Model Checkpoint : {target_model}")
    print("------------------------------------------------------------------------------------------")

    # Load AASIST detector
    detector = AASISTDetector(model_path=target_model, device="auto")
    pipeline = get_pipeline()

    audio_extensions = (".wav", ".mp3", ".m4a", ".flac", ".mpeg", ".ogg")
    files = [f for f in sorted(os.listdir(folder_path)) if f.lower().endswith(audio_extensions)]

    if not files:
        print(f"No audio files found in '{folder_path}' with extensions {audio_extensions}")
        return

    header_fmt = "{:<20} | {:<12} | {:<12} | {:<12} | {:<16} | {:<12}"
    print(header_fmt.format("File Name", "Human Prob", "AI Prob", "Prediction", "Unified Risk", "Risk Level"))
    print("-" * 94)

    results = []

    for file_name in files:
        full_path = os.path.join(folder_path, file_name)
        try:
            start_t = time.time()
            aasist_res = detector.predict(full_path)
            pipeline_res = pipeline.analyze_audio_sample(full_path)
            elapsed_ms = (time.time() - start_t) * 1000

            human_prob_pct = aasist_res["bonafide_probability"] * 100
            ai_prob_pct = aasist_res["spoof_probability"] * 100
            prediction = aasist_res["prediction"].upper()
            unified_risk = pipeline_res["unified_risk_score"]
            risk_level = pipeline_res["risk_level"]

            print(
                header_fmt.format(
                    file_name[:20],
                    f"{human_prob_pct:6.2f}%",
                    f"{ai_prob_pct:6.2f}%",
                    prediction,
                    f"{unified_risk:6.2f}%",
                    risk_level,
                )
            )

            results.append(
                {
                    "file_name": file_name,
                    "human_prob": human_prob_pct,
                    "ai_prob": ai_prob_pct,
                    "prediction": prediction,
                    "unified_risk": unified_risk,
                    "risk_level": risk_level,
                    "inference_time_ms": elapsed_ms,
                }
            )
        except Exception as err:
            print(f"{file_name:<20} | ERROR: {err}")

    print("=" * 94)
    print(f"Evaluated {len(results)} audio samples successfully.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AASIST AI Voice Detector on test audio samples.")
    default_dir = os.path.join(root_dir, "test_voice")
    if not os.path.exists(default_dir) and os.path.exists(os.path.join(root_dir, "test_voices")):
        default_dir = os.path.join(root_dir, "test_voices")

    parser.add_argument("--folder", type=str, default=default_dir, help="Directory containing test audio files")
    parser.add_argument("--model", type=str, default=None, help="Path to AASIST model checkpoint")
    args = parser.parse_args()

    test_voice_folder(args.folder, args.model)
