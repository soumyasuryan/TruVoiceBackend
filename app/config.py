import os
from pydantic_settings import BaseSettings
load_dotenv()
class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Scam & Deepfake Detector API"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "SUPER_SECRET_JWT_KEY_CHANGE_THIS")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 Days
    
    # Database (MongoDB / PostgreSQL URI)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./test.db")
    
    # Groq API
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

settings = Settings()