import time
import logging
from app.config import settings

logger = logging.getLogger(__name__)

def generate_rtc_token(channel_name: str, user_id: str, expiration_time_in_seconds: int = 3600) -> str:
    """
    Generates a secure Agora RTC Token for joining a specified audio channel.
    Uses a stable, deterministic numeric UID computed from user_id. The JS _computeUid()
    function in agoraService.js must produce the identical value for the token to be valid.
    Token is ALWAYS built with the numeric UID — never with a string user account —
    to match what the frontend passes to joinChannel(token, channel, numericUid, ...).
    """
    app_id = settings.AGORA_APP_ID
    app_certificate = settings.AGORA_APP_CERTIFICATE

    if not app_id or not app_certificate:
        logger.warning(
            "AGORA_APP_ID or AGORA_APP_CERTIFICATE missing from settings. "
            "Both must be set in .env for Agora tokens to work. "
            "Returning placeholder — media will NOT connect in production."
        )
        return f"agora_dev_token_{channel_name}_{user_id}"

    privilege_expired_ts = int(time.time()) + expiration_time_in_seconds

    # Compute deterministic numeric UID — MUST match agoraService.js _computeUid()
    uid_value = 1
    if user_id:
        source = str(user_id)
        uid_value = (sum((i + 1) * ord(ch) for i, ch in enumerate(source)) % 1000000000) + 1

    logger.info(f"Generating Agora RTC token: channel={channel_name}, userId={user_id}, uid={uid_value}")

    try:
        from agora_token_builder import RtcTokenBuilder
        role_publisher = 1

        # Always use numeric UID to match joinChannel(token, channel, numericUid, ...)
        # DO NOT use buildTokenWithUserAccount — it creates a string-UID token that
        # mismatches the numeric UID used by the frontend, breaking Agora media auth.
        token = RtcTokenBuilder.buildTokenWithUid(
            app_id,
            app_certificate,
            channel_name,
            uid_value,
            role_publisher,
            privilege_expired_ts,
        )
        return token
    except Exception as e:
        logger.error(
            f"Error generating Agora RTC token for channel={channel_name}, uid={uid_value}: {e}. "
            f"Check that agora_token_builder is installed (pip install agora-token-builder) "
            f"and AGORA_APP_ID/AGORA_APP_CERTIFICATE are correct in .env."
        )
        return f"agora_dev_token_{channel_name}_{user_id}"
