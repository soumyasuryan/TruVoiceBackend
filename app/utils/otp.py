import secrets


def generate_otp() -> str:
    """Generate a cryptographically secure six-digit OTP.

    Delivery is intentionally kept outside this helper; integrate an SMS provider
    before deploying so OTPs are never logged or returned by the API.
    """
    return f"{secrets.randbelow(1_000_000):06d}"
