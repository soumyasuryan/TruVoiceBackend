import os
import json
import re
from groq import Groq
from src.exception import CustomException
from src.logger import logger
import sys

class GroqScamDetector:
    def __init__(self, api_key: str = None):
        """
        Initializes the Groq client for Speech-to-Text and Intent Detection.
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required for GroqScamDetector.")
        
        self.client = Groq(api_key=self.api_key)
        self.whisper_model = "whisper-large-v3-turbo"
        self.llm_model = "llama-3.3-70b-versatile"

    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribes audio using Groq Whisper API.
        """
        try:
            with open(audio_path, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=(os.path.basename(audio_path), file.read()),
                    model=self.whisper_model,
                    response_format="text",
                    language="en"
                )
            return str(transcription).strip()
        except Exception as e:
            logger.error(f"Error during audio transcription: {e}")
            raise CustomException(e, sys)

    def analyze_scam_intent(self, transcript: str) -> dict:
        """
        Analyzes the transcript using Llama-3.3-70B to evaluate scam intent, category, and risk keywords.
        """
        if not transcript or len(transcript.strip()) < 3:
            return {
                "scam_score": 0.0,
                "category": "Insufficient Audio / Silent Call",
                "risk_keywords": [],
                "reasoning": "Audio clip too short or contains no recognizable speech."
            }

        prompt = f"""
You are an expert Cybersecurity AI system analyzing phone call transcripts for phishing, scams, and financial fraud.

Analyze the following phone call transcript:
"{transcript}"

Provide a JSON output containing:
1. "scam_score": Float between 0.0 (100% legitimate) and 1.0 (100% scam/phishing).
2. "category": String describing the threat category (e.g., "Bank OTP Fraud", "KYC Scam", "Tech Support Fraud", "Urgent Legal Threat", "Safe Call").
3. "risk_keywords": List of suspicious words/phrases spotted in the transcript (e.g., ["OTP", "CVV", "urgent", "account blocked"]).
4. "reasoning": A concise 1-2 sentence explanation of why this was flagged or marked safe.

Return strictly valid JSON only. Do not include markdown headers or extra conversation.
JSON format:
{{
  "scam_score": 0.85,
  "category": "Bank OTP Fraud",
  "risk_keywords": ["OTP", "account suspended"],
  "reasoning": "The caller claims to be from a bank and asks for an urgent OTP."
}}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a JSON-only response bot for scam detection."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            raw_content = response.choices[0].message.content.strip()
            parsed_json = json.loads(raw_content)
            
            return {
                "scam_score": float(parsed_json.get("scam_score", 0.0)),
                "category": str(parsed_json.get("category", "General Query")),
                "risk_keywords": list(parsed_json.get("risk_keywords", [])),
                "reasoning": str(parsed_json.get("reasoning", "No suspicious activity detected."))
            }
        except Exception as e:
            logger.error(f"Error during intent analysis: {e}")
            return {
                "scam_score": 0.0,
                "category": "Analysis Error",
                "risk_keywords": [],
                "reasoning": f"Failed to analyze transcript intent: {str(e)}"
            }

    def run(self, audio_path: str) -> dict:
        """
        Executes both transcription and intent detection.
        """
        transcript = self.transcribe_audio(audio_path)
        analysis = self.analyze_scam_intent(transcript)
        
        return {
            "transcript": transcript,
            "scam_score": analysis["scam_score"],
            "category": analysis["category"],
            "risk_keywords": analysis["risk_keywords"],
            "reasoning": analysis["reasoning"]
        }