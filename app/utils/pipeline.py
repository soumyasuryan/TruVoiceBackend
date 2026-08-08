import os
import sys
import joblib
import numpy as np
import librosa
from src.scam_detector import GroqScamDetector
from app.config import settings

class UnifiedPipelineTester:
    def __init__(self, model_path: str = "model/voice_detector_model.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at '{model_path}'.")
        
        self.acoustic_model = joblib.load(model_path)
        
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is missing.")
        
        self.scam_detector = GroqScamDetector(api_key=settings.GROQ_API_KEY)

    def extract_acoustic_features(self, audio_path: str, sr: int = 16000) -> np.ndarray:
        audio_array, _ = librosa.load(audio_path, sr=sr)
        
        f0, _, _ = librosa.pyin(audio_array, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
        f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([0.0])
        
        pitch_mean = np.mean(f0_clean) if len(f0_clean) > 0 else 0.0
        pitch_std = np.std(f0_clean) if len(f0_clean) > 0 else 0.0
        pitch_max = np.max(f0_clean) if len(f0_clean) > 0 else 0.0
        pitch_min = np.min(f0_clean) if len(f0_clean) > 0 else 0.0

        mfcc = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_array, sr=sr))
        spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_array, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio_array))

        features = np.hstack([pitch_mean, pitch_std, pitch_max, pitch_min, spec_centroid, spec_rolloff, zcr, mfcc_mean, mfcc_std])
        return features.reshape(1, -1)

    def analyze_audio_sample(self, audio_path: str) -> dict:
        X_features = self.extract_acoustic_features(audio_path)
        ai_voice_prob = float(self.acoustic_model.predict_proba(X_features)[0][1])
        
        nlp_result = self.scam_detector.run(audio_path)
        scam_text_score = float(nlp_result.get("scam_score", 0.0))
        
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
            "reasoning": nlp_result.get("reasoning", "")
        }

# Singleton instance
pipeline_engine = None

def get_pipeline():
    global pipeline_engine
    if pipeline_engine is None:
        pipeline_engine = UnifiedPipelineTester()
    return pipeline_engine