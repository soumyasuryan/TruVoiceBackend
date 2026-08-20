import logging
import os
import sys
import joblib
import librosa
import numpy as np
from app.config import settings
from src.scam_detector import GroqScamDetector

logger = logging.getLogger(__name__)


class UnifiedPipelineTester:
    def __init__(self, model_path: str = "model/voice_detector_model.pkl"):
        if not os.path.exists(model_path):
            error_msg = f"Model file not found at '{model_path}'."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        self.acoustic_model = joblib.load(model_path)

        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is missing.")

        self.scam_detector = GroqScamDetector(api_key=settings.GROQ_API_KEY)

    def extract_acoustic_features(self, audio_array: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extracts 47 acoustic features (F0 pitch statistics, spectral centroid, spectral rolloff, ZCR, MFCCs)
        matching the exact feature structure expected by the XGBoost acoustic model.
        """
        # Peak normalize waveform
        max_val = float(np.max(np.abs(audio_array))) if len(audio_array) > 0 else 0.0
        if max_val > 1.0:
            audio_array = audio_array / max_val
        elif 0.0 < max_val < 0.1:
            audio_array = audio_array / (max_val + 1e-8)

        # F0 pitch estimation with fallback for unvoiced / noisy speech frames
        try:
            f0, _, _ = librosa.pyin(
                audio_array,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
            )
            f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        except Exception:
            f0_clean = np.array([])

        if len(f0_clean) > 0:
            pitch_mean = float(np.mean(f0_clean))
            pitch_std = float(np.std(f0_clean))
            pitch_max = float(np.max(f0_clean))
            pitch_min = float(np.min(f0_clean))
        else:
            # Human vocal pitch baseline (130 Hz) for unvoiced or quiet speech frames
            pitch_mean = 130.0
            pitch_std = 15.0
            pitch_max = 180.0
            pitch_min = 90.0

        mfcc = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio_array, sr=sr)))
        spec_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=audio_array, sr=sr)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=audio_array)))

        features = np.hstack([pitch_mean, pitch_std, pitch_max, pitch_min, spec_centroid, spec_rolloff, zcr, mfcc_mean, mfcc_std])
        return features.reshape(1, -1)

    def analyze_audio_sample(self, audio_path: str) -> dict:
        """
        Executes XGBoost acoustic AI voice detection + Groq NLP scam intent detection.
        Applies RMS energy silence gate and probability calibration for live call streams.
        """
        audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
        rms_energy = float(np.sqrt(np.mean(audio_array**2))) if len(audio_array) > 0 else 0.0

        # Silence / ambient noise gate (RMS < 0.008 -> 0% AI Voice probability)
        if rms_energy < 0.008:
            ai_voice_prob = 0.0
        else:
            X_features = self.extract_acoustic_features(audio_array, sr=sr)
            raw_prob = float(self.acoustic_model.predict_proba(X_features)[0][1])

            # Probability calibration for live microphone stream audio
            # Softens uncalibrated raw scores (<0.65 -> scaled down to prevent false deepfake alerts)
            if raw_prob < 0.65:
                ai_voice_prob = raw_prob * 0.35
            else:
                ai_voice_prob = raw_prob

        # Groq NLP scam intent analysis (Unchanged)
        nlp_result = self.scam_detector.run(audio_path)
        scam_text_score = float(nlp_result.get("scam_score", 0.0))

        # Unified risk score calculation (Unchanged)
        unified_risk = (0.5 * ai_voice_prob) + (0.5 * scam_text_score)
        risk_level = "CRITICAL RISK" if unified_risk >= 0.7 else "MEDIUM RISK" if unified_risk >= 0.35 else "LOW RISK"

        return {
            "file_name": os.path.basename(audio_path),
            "transcript": nlp_result.get("transcript", ""),
            "ai_voice_probability": round(ai_voice_prob * 100, 2),
            "scam_intent_score": round(scam_text_score * 100, 2),
            "unified_risk_score": round(unified_risk * 100, 2),
            "risk_level": risk_level,
            "scam_category": nlp_result.get("category", "N/A"),
            "flagged_keywords": nlp_result.get("risk_keywords", []),
            "reasoning": nlp_result.get("reasoning", ""),
        }


# Singleton instance
pipeline_engine = None


def get_pipeline():
    global pipeline_engine
    if pipeline_engine is None:
        pipeline_engine = UnifiedPipelineTester()
    return pipeline_engine