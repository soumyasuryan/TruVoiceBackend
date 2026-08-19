import logging
from contextlib import asynccontextmanager
from pathlib import Path
import sys

# Add project root directory to sys.path to allow running directly from app/ or project root
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from app.routers import analysis, auth, community, voice
from app.utils.pipeline import get_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan handler: loads and validates the AASIST PyTorch model once during startup.
    Fails startup cleanly if checkpoint or model initialization fails.
    """
    logger.info("Initializing AASIST AI-Voice detection pipeline...")
    try:
        pipeline = get_pipeline()
        detector = pipeline.aasist_detector
        logger.info(f"AASIST model loaded successfully")
        logger.info(f"AASIST device: {detector.device}")
        logger.info(f"AASIST sample rate: {detector.sample_rate}")
        logger.info(f"AASIST input samples: {detector.num_samples}")
        logger.info(f"AASIST threshold: {detector.threshold}")
    except Exception as e:
        logger.critical(f"FATAL: AASIST model startup initialization failed: {e}")
        raise RuntimeError(f"Server startup aborted due to AASIST model loading error: {e}") from e
    yield
    logger.info("Shutting down AI Voice Scam Detector API.")


app = FastAPI(
    title="AI Voice Scam Detector API",
    description="Backend API handling Phone Auth and Dual-Pipeline AI Scam Detection with PyTorch AASIST.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for React Native & Web Frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(auth.router)
app.include_router(analysis.router)
app.include_router(community.router)
app.include_router(voice.router)
app.include_router(voice.ws_router)


@app.get("/")
def root():
    return {"status": "online", "message": "AI Voice Scam Detection API operational with AASIST model."}


# AWS Lambda Handler Wrapper
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
