import os
import shutil
import tempfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.database import get_supabase
from app.schemas import AudioAnalysisResponse
from app.utils.auth import get_current_user_id
from app.utils.pipeline import get_pipeline

router = APIRouter(prefix="/api/v1", tags=["Audio Analysis"])


@router.post("/analyze", response_model=AudioAnalysisResponse)
async def analyze_audio(
    file: UploadFile = File(...),
    caller_number: str | None = Form(default=None),
    user_id: str = Depends(get_current_user_id),
):
    if not file.filename or not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".flac")):
        raise HTTPException(status_code=400, detail="Unsupported audio format. Provide WAV, MP3, M4A, or FLAC.")
    if caller_number:
        caller_number = caller_number.strip().replace(" ", "").replace("-", "")

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, os.path.basename(file.filename))
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        result = get_pipeline().analyze_audio_sample(temp_path)
        _save_analysis_history(user_id, caller_number, result)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(exc)}") from exc
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.get("/analysis-history")
def get_analysis_history(user_id: str = Depends(get_current_user_id)):
    result = (
        get_supabase()
        .table("call_analysis_history")
        .select("id,caller_number,file_name,transcript,ai_voice_probability,scam_intent_score,unified_risk_score,risk_level,scam_category,flagged_keywords,reasoning,created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(30)
        .execute()
    )
    return {"items": result.data or []}


def _save_analysis_history(user_id: str, caller_number: str | None, result: dict) -> None:
    db = get_supabase()
    db.table("call_analysis_history").insert(
        {
            "user_id": user_id,
            "caller_number": caller_number,
            "file_name": result["file_name"],
            "transcript": result["transcript"],
            "ai_voice_probability": result["ai_voice_probability"],
            "scam_intent_score": result["scam_intent_score"],
            "unified_risk_score": result["unified_risk_score"],
            "risk_level": result["risk_level"],
            "scam_category": result["scam_category"],
            "flagged_keywords": result["flagged_keywords"],
            "reasoning": result["reasoning"],
        }
    ).execute()

