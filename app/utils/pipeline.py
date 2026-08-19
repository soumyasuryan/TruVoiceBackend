import logging
import os
from app.ai_voice.aasist_detector import AASISTDetector
from app.config import settings
from src.scam_detector import GroqScamDetector

logger = logging.getLogger(__name__)


class UnifiedPipelineTester:
    def __init__(self, model_path: str = None):
        target_model_path = model_path or settings.AASIST_MODEL_PATH

        if not os.path.exists(target_model_path):
            error_msg = f"AASIST model checkpoint not found: {target_model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Initialize AASIST Deepfake Voice Detector (replaces XGBoost)
        self.aasist_detector = AASISTDetector(
            model_path=target_model_path,
            device="auto",
            threshold=settings.AASIST_THRESHOLD,
        )

        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is missing.")

        # Initialize Groq NLP Scam Intent Detector (UNCHANGED)
        self.scam_detector = GroqScamDetector(api_key=settings.GROQ_API_KEY)

    def analyze_audio_sample(self, audio_path: str) -> dict:
        """
        Executes dual AI analysis pipeline:
        1. AASIST PyTorch AI-Voice Detector (audio waveform -> spoof probability)
        2. Groq Whisper + LLM Scam Intent Detector (transcript -> scam score)
        Combines scores using unchanged unified risk formula: (0.5 * ai_voice_prob) + (0.5 * scam_text_score).
        """
        # Step 1: AASIST AI-Voice Detection
        aasist_result = self.aasist_detector.predict(audio_path)
        ai_voice_prob = float(aasist_result["spoof_probability"])  # 0.0 (bonafide) to 1.0 (spoof)

        # Step 2: Groq Scam Intent Detection (Unchanged)
        nlp_result = self.scam_detector.run(audio_path)
        scam_text_score = float(nlp_result.get("scam_score", 0.0))

        # Step 3: Unified Risk Score (Unchanged formula)
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