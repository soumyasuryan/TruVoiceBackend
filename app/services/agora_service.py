import time
import logging
from app.config import settings

logger = logging.getLogger(__name__)

def generate_rtc_token(channel_name: str, user_id: str, expiration_time_in_seconds: int = 3600) -> str:
    """
    Generates a secure Agora RTC Token for joining a specified audio channel.
    Uses user_id as string user account or numeric hash.
    """
    app_id = settings.AGORA_APP_ID
    app_certificate = settings.AGORA_APP_CERTIFICATE

    if not app_id or not app_certificate:
        logger.warning("AGORA_APP_ID or AGORA_APP_CERTIFICATE missing from settings; returning development token.")
        return f"agora_dev_token_{channel_name}_{user_id}"

    privilege_expired_ts = int(time.time()) + expiration_time_in_seconds

    try:
        from agora_token_builder import RtcTokenBuilder
        # Role 1 is Publisher (can speak and listen)
        role_publisher = 1

        # Build token with uid=0 (wildcard integer UID) so client joining with uid 0 or user account works
        try:
            token = RtcTokenBuilder.buildTokenWithUid(
                app_id,
                app_certificate,
                channel_name,
                0,
                role_publisher,
                privilege_expired_ts
            )
        except Exception:
            token = RtcTokenBuilder.buildTokenWithUserAccount(
                app_id,
                app_certificate,
                channel_name,
                str(user_id),
                role_publisher,
                privilege_expired_ts
            )
        return token
    except Exception as e:
        logger.error(f"Error generating Agora RTC token: {e}")
        return f"agora_dev_token_{channel_name}_{user_id}"
