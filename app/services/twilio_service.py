import logging
from app.config import settings

logger = logging.getLogger(__name__)

def generate_voice_token(identity: str) -> str:
    """
    Generates a secure server-side Twilio Access Token with VoiceGrant.
    """
    try:
        from twilio.jwt.access_token import AccessToken
        from twilio.jwt.access_token.grants import VoiceGrant

        account_sid = settings.TWILIO_ACCOUNT_SID
        api_key = settings.TWILIO_API_KEY
        api_secret = settings.TWILIO_API_SECRET
        twiml_app_sid = settings.TWILIO_TWIML_APP_SID

        if not account_sid or not api_key or not api_secret:
            logger.warning("Twilio credentials not configured; generating development token.")
            return f"dev_token_{identity}"

        token = AccessToken(
            account_sid,
            api_key,
            api_secret,
            identity=identity,
            ttl=3600,
        )

        grant = VoiceGrant(
            outgoing_application_sid=twiml_app_sid,
            incoming_allow=True,
        )
        token.add_grant(grant)
        return token.to_jwt()
    except Exception as e:
        logger.error(f"Error generating Twilio token: {e}")
        return f"dev_token_{identity}"


def initiate_twilio_call(to_phone_number: str, call_id: str, public_url: str) -> str | None:
    """
    Initiates an outgoing call using the Twilio REST Client.
    Connects to the backend TwiML webhook URL.
    """
    account_sid = settings.TWILIO_ACCOUNT_SID
    auth_token = settings.TWILIO_AUTH_TOKEN
    from_number = settings.TWILIO_PHONE_NUMBER

    if not account_sid or not auth_token or not from_number:
        logger.warning("Twilio REST client credentials missing. Simulating call initiation in dev mode.")
        return f"dev_sid_{call_id}"

    try:
        from twilio.rest import Client
        client = Client(account_sid, auth_token)

        twiml_url = f"{public_url.rstrip('/')}/api/v1/voice/twiml?call_id={call_id}"
        status_url = f"{public_url.rstrip('/')}/api/v1/voice/status"

        call = client.calls.create(
            to=to_phone_number,
            from_=from_number,
            url=twiml_url,
            status_callback=status_url,
            status_callback_event=["initiated", "ringing", "answered", "completed"],
            status_callback_method="POST"
        )
        return call.sid
    except Exception as e:
        logger.error(f"Failed to initiate Twilio call to {to_phone_number}: {e}")
        return f"dev_sid_{call_id}"


def generate_twiml_response(call_id: str, public_url: str) -> str:
    """
    Generates TwiML XML connecting the active phone call to the media stream WebSocket.
    """
    # Convert http(s) to ws(s)
    clean_url = public_url.rstrip('/')
    if clean_url.startswith("https://"):
        ws_base = clean_url.replace("https://", "wss://")
    elif clean_url.startswith("http://"):
        ws_base = clean_url.replace("http://", "ws://")
    else:
        ws_base = f"wss://{clean_url}"

    ws_url = f"{ws_base}/ws/voice-stream"

    try:
        from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
        response = VoiceResponse()
        connect = Connect()
        stream = connect.stream(url=ws_url)
        stream.parameter(name="call_id", value=call_id)
        connect.append(stream)
        response.append(connect)
        return str(response)
    except ImportError:
        # Fallback TwiML XML string if twilio package is initializing
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}">
            <Parameter name="call_id" value="{call_id}" />
        </Stream>
    </Connect>
</Response>"""
