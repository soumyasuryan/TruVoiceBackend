import os
import shutil
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas import AudioAnalysisResponse
from app.utils.pipeline import get_pipeline

router = APIRouter(prefix="/api/v1", tags=["Audio Analysis"])

@router.post("/analyze", response_model=AudioAnalysisResponse)
async def analyze_audio(file: UploadFile = File(...)):
    if not file.filename.endswith(('.wav', '.mp3', '.m4a', '.flac')):
        raise HTTPException(status_code=400, detail="Unsupported audio format. Provide WAV or MP3.")
    
    # Save temporary file
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run Unified Pipeline
        pipeline = get_pipeline()
        result = pipeline.analyze_audio_sample(temp_path)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)