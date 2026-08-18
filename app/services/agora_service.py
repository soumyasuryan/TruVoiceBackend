import time
import logging
from app.config import settings

logger = logging.getLogger(__name__)

def generate_rtc_token(channel_name: str, user_id: str, expiration_time_in_seconds: int = 3600) -> str:
    """
    Generates a secure Agora RTC Token for joining a specified audio channel.
    Uses a stable, user-specific UID so both peers can join the same channel without colliding.
    """
    app_id = settings.AGORA_APP_ID
    app_certificate = settings.AGORA_APP_CERTIFICATE

    if not app_id or not app_certificate:
        logger.warning("AGORA_APP_ID or AGORA_APP_CERTIFICATE missing from settings; returning development token.")
        return f"agora_dev_token_{channel_name}_{user_id}"

    privilege_expired_ts = int(time.time()) + expiration_time_in_seconds

    try:
        from agora_token_builder import RtcTokenBuilder

        role_publisher = 1
        uid_value = 1
        if user_id:
            source = str(user_id)
            uid_value = (sum((i + 1) * ord(ch) for i, ch in enumerate(source)) % 1000000000) + 1

        try:
            token = RtcTokenBuilder.buildTokenWithUid(
                app_id,
                app_certificate,
                channel_name,
                uid_value,
                role_publisher,
                privilege_expired_ts,
            )
        except Exception:
            token = RtcTokenBuilder.buildTokenWithUserAccount(
                app_id,
                app_certificate,
                channel_name,
                str(user_id),
                role_publisher,
                privilege_expired_ts,
            )
        return token
    except Exception as e:
        logger.error(f"Error generating Agora RTC token: {e}")
        return f"agora_dev_token_{channel_name}_{user_id}"
