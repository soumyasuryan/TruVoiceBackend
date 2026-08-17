import base64
import os
import shutil
import tempfile
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.database import get_supabase
from app.schemas import AudioAnalysisResponse, AudioBase64Request
from app.utils.auth import get_current_user_id
from app.utils.pipeline import get_pipeline
from app.utils.websocket_manager import manager

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


@router.post("/analysis/audio", response_model=AudioAnalysisResponse)
async def analyze_base64_audio(
    payload: AudioBase64Request,
    user_id: str = Depends(get_current_user_id),
):
    """
    Accepts a Base64-encoded audio payload (WAV buffer captured from live call stream),
    decodes bytes, saves to a temporary WAV file, executes AI analysis pipeline,
    records history, and broadcasts real-time analysis updates if call_id is supplied.
    """
    if not payload.audio_base64:
        raise HTTPException(status_code=400, detail="audio_base64 payload is required.")

    b64_data = payload.audio_base64
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]

    try:
        audio_bytes = base64.b64decode(b64_data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio encoding.") from exc

    caller_num = payload.caller_number
    if caller_num:
        caller_num = caller_num.strip().replace(" ", "").replace("-", "")

    temp_dir = tempfile.mkdtemp()
    temp_filename = f"agora_chunk_{int(time.time() * 1000)}.wav"
    temp_path = os.path.join(temp_dir, temp_filename)

    try:
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)

        result = get_pipeline().analyze_audio_sample(temp_path)
        _save_analysis_history(user_id, caller_num, result)

        if payload.call_id:
            await manager.broadcast_to_call(payload.call_id, {
                "type": "analysis_update",
                "trust_score": round(max(0.0, 100.0 - result.get("unified_risk_score", 0.0)), 1),
                "confidence": 95.0,
                "risk_level": result.get("risk_level", "LOW RISK"),
                "is_scam": result.get("scam_intent_score", 0.0) > 50.0,
                "is_ai_voice": result.get("ai_voice_probability", 0.0) > 50.0,
                "transcript": result.get("transcript", ""),
                "signals": result.get("flagged_keywords", []),
                "reasoning": result.get("reasoning", ""),
                "ai_voice_probability": result.get("ai_voice_probability", 0.0),
                "scam_intent_score": result.get("scam_intent_score", 0.0),
                "unified_risk_score": result.get("unified_risk_score", 0.0),
            })

        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Base64 pipeline execution error: {str(exc)}") from exc
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

