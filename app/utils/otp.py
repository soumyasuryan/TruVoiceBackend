import secrets
import smtplib
from email.message import EmailMessage

from app.config import settings


def generate_otp() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def send_email_otp(email: str, otp_code: str) -> None:
    """Send a verification OTP through the configured SMTP account."""
    required = (settings.SMTP_HOST, settings.SMTP_USERNAME, settings.SMTP_PASSWORD, settings.SMTP_FROM_EMAIL)
    if not all(required):
        raise RuntimeError("Email OTP is not configured. Set SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, and SMTP_FROM_EMAIL.")

    message = EmailMessage()
    message["Subject"] = "Your TruVoice verification code"
    message["From"] = settings.SMTP_FROM_EMAIL
    message["To"] = email
    message.set_content(f"Your TruVoice verification code is {otp_code}. It expires in 10 minutes. Do not share it with anyone.")
    with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=15) as smtp:
        smtp.starttls()
        smtp.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
        smtp.send_message(message)
