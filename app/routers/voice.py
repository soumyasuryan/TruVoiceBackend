import datetime
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request, Response, status, WebSocket, WebSocketDisconnect
from jose import JWTError, jwt

from app.config import settings
from app.database import get_supabase
from app.schemas import (
    OutgoingCallRequest,
    OutgoingCallResponse,
    VoiceCallResponse,
    VoiceTokenResponse,
)
from app.services.streaming_service import (
    get_or_create_stream_session,
    remove_stream_session,
)
from app.services.twilio_service import (
    generate_twiml_response,
    generate_voice_token,
    initiate_twilio_call,
)
from app.utils.auth import get_current_user_id
from app.utils.websocket_manager import manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/voice", tags=["Voice Calling & Telephony"])


@router.post("/token", response_model=VoiceTokenResponse)
def get_voice_token(user_id: str = Depends(get_current_user_id)):
    """
    Generates a secure server-side Twilio access token for the authenticated user.
    """
    identity = f"user_{user_id}"
    token = generate_voice_token(identity)
    return VoiceTokenResponse(token=token, identity=identity)


@router.post("/outgoing", response_model=OutgoingCallResponse)
def initiate_outgoing_call(
    payload: OutgoingCallRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    Creates a database call record and initiates an outgoing phone call via Twilio.
    """
    db = get_supabase()
    phone_number = payload.phone_number

    # Insert initial call record into database
    call_record = db.table("voice_calls").insert({
        "user_id": user_id,
        "phone_number": phone_number,
        "status": "initiated",
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }).execute()

    if not call_record.data:
        raise HTTPException(status_code=500, detail="Failed to initialize call record in database.")

    call_id = str(call_record.data[0]["id"])

    # Initiate Twilio call
    public_url = settings.PUBLIC_SERVER_URL
    provider_sid = initiate_twilio_call(phone_number, call_id, public_url)

    if provider_sid:
        db.table("voice_calls").update({"provider_call_sid": provider_sid}).eq("id", call_id).execute()

    return OutgoingCallResponse(
        call_id=call_id,
        status="initiated",
        phone_number=phone_number
    )


@router.post("/twiml")
@router.get("/twiml")
def get_call_twiml(call_id: str = Query(...)):
    """
    Twilio Webhook returning TwiML XML to connect the voice call to the media stream WebSocket.
    """
    public_url = settings.PUBLIC_SERVER_URL
    twiml_content = generate_twiml_response(call_id, public_url)
    return Response(content=twiml_content, media_type="application/xml")


@router.post("/status")
async def handle_call_status(request: Request):
    """
    Twilio Status Callback Webhook to track call lifecycle.
    """
    try:
        form_data = await request.form()
        call_sid = form_data.get("CallSid")
        call_status = form_data.get("CallStatus", "unknown").lower()
        duration = form_data.get("CallDuration")

        db = get_supabase()
        existing = db.table("voice_calls").select("id").eq("provider_call_sid", call_sid).execute()

        if existing.data:
            call_id = existing.data[0]["id"]
            update_data = {
                "status": call_status,
                "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            if call_status == "in-progress" or call_status == "answered":
                update_data["answered_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            elif call_status in ["completed", "busy", "failed", "no-answer", "canceled"]:
                update_data["ended_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                if duration and str(duration).isdigit():
                    update_data["duration"] = int(duration)

                # Broadcast call_ended notification to subscribers
                await manager.broadcast_to_call(str(call_id), {
                    "type": "call_ended",
                    "call_id": str(call_id),
                    "status": call_status
                })

            db.table("voice_calls").update(update_data).eq("id", call_id).execute()

        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error handling Twilio call status: {e}")
        return {"status": "error", "detail": str(e)}


@router.get("/calls", response_model=dict)
def list_user_voice_calls(user_id: str = Depends(get_current_user_id)):
    """
    Fetches historical voice calls for the authenticated user.
    """
    db = get_supabase()
    result = (
        db.table("voice_calls")
        .select("id,user_id,phone_number,provider_call_sid,status,risk_level,trust_score,confidence,is_scam,is_ai_voice,transcript,signals,duration,created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(30)
        .execute()
    )
    return {"items": result.data or []}


@router.get("/calls/{call_id}")
def get_voice_call_detail(call_id: str, user_id: str = Depends(get_current_user_id)):
    """
    Fetches details for a specific call. Enforces user ownership.
    """
    db = get_supabase()
    result = db.table("voice_calls").select("*").eq("id", call_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Call record not found.")

    record = result.data[0]
    if str(record["user_id"]) != user_id:
        raise HTTPException(status_code=403, detail="Unauthorized access to this call record.")

    return record


# WebSocket Routes setup helper
ws_router = APIRouter(tags=["WebSockets"])

@ws_router.websocket("/ws/voice-stream")
async def handle_media_stream_ws(websocket: WebSocket):
    """
    WebSocket endpoint receiving real-time audio streams from Twilio.
    """
    await websocket.accept()
    logger.info("Twilio Media Stream WebSocket connected.")

    current_call_id: Optional[str] = None
    stream_sid: Optional[str] = None

    try:
        while True:
            raw_msg = await websocket.receive_text()
            if not raw_msg:
                continue

            msg = json.loads(raw_msg)
            event_type = msg.get("event")

            if event_type == "start":
                start_data = msg.get("start", {})
                stream_sid = start_data.get("streamSid")
                custom_params = start_data.get("customParameters", {})
                current_call_id = custom_params.get("call_id") or msg.get("streamSid")

                logger.info(f"Media stream started for call_id: {current_call_id}, streamSid: {stream_sid}")
                get_or_create_stream_session(current_call_id)

                if current_call_id:
                    await manager.broadcast_to_call(current_call_id, {
                        "type": "call_started",
                        "call_id": current_call_id,
                        "status": "connected"
                    })

            elif event_type == "media":
                media_data = msg.get("media", {})
                payload = media_data.get("payload")
                if payload and current_call_id:
                    session = get_or_create_stream_session(current_call_id)
                    await session.append_mulaw_payload(payload)

            elif event_type == "stop":
                logger.info(f"Media stream stopped for call_id: {current_call_id}")
                if current_call_id:
                    remove_stream_session(current_call_id)
                    await manager.broadcast_to_call(current_call_id, {
                        "type": "call_ended",
                        "call_id": current_call_id,
                        "status": "completed"
                    })
                break

    except WebSocketDisconnect:
        logger.info(f"Media stream WebSocket disconnected for call_id: {current_call_id}")
    except Exception as e:
        logger.error(f"Error handling media stream WS: {e}")
    finally:
        if current_call_id:
            remove_stream_session(current_call_id)


@ws_router.websocket("/ws/live-analysis/{call_id}")
async def handle_live_analysis_ws(websocket: WebSocket, call_id: str, token: str = Query(...)):
    """
    WebSocket endpoint for mobile clients to receive real-time AI scam analysis updates.
    Requires JWT token and ownership validation of call_id.
    """
    # 1. Authenticate user JWT
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
    except JWTError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # 2. Check call ownership in database
    db = get_supabase()
    call_record = db.table("voice_calls").select("user_id").eq("id", call_id).execute()
    if not call_record.data or str(call_record.data[0]["user_id"]) != str(user_id):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # 3. Connect subscriber
    await manager.connect_analysis(call_id, websocket)
    try:
        while True:
            # Keep socket open and listen for ping/client messages
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        await manager.disconnect_analysis(call_id, websocket)
    except Exception as e:
        logger.error(f"Error in live analysis WS for call {call_id}: {e}")
        await manager.disconnect_analysis(call_id, websocket)
