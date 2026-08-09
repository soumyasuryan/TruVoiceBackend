import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()
class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Scam & Deepfake Detector API"
    SECRET_KEY: str = os.getenv("SECRET_KEY")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7 
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    SMTP_HOST: str = os.getenv("SMTP_HOST", "")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME: str = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    SMTP_FROM_EMAIL: str = os.getenv("SMTP_FROM_EMAIL", "")

    # Twilio Telephony Settings
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_API_KEY: str = os.getenv("TWILIO_API_KEY", "")
    TWILIO_API_SECRET: str = os.getenv("TWILIO_API_SECRET", "")
    TWILIO_TWIML_APP_SID: str = os.getenv("TWILIO_TWIML_APP_SID", "")
    TWILIO_PHONE_NUMBER: str = os.getenv("TWILIO_PHONE_NUMBER", "")

    # Server & Voice Stream Settings
    PUBLIC_SERVER_URL: str = os.getenv("PUBLIC_SERVER_URL", "http://localhost:8000")
    VOICE_ANALYSIS_CHUNK_SECONDS: float = float(os.getenv("VOICE_ANALYSIS_CHUNK_SECONDS", "3.0"))
    VOICE_ANALYSIS_INTERVAL_SECONDS: float = float(os.getenv("VOICE_ANALYSIS_INTERVAL_SECONDS", "3.0"))

settings = Settings()

