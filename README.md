# TruVoice Backend — Real-Time AI Voice Scam Detector & Telephony Engine

This repository powers TruVoice's real-time phone calling and live AI scam detection engine.

## Features
- **Phone Authentication**: JWT auth, password hashing, email OTP verification (`/auth/*`).
- **Audio Analysis API**: `POST /api/v1/analyze` for recorded audio files.
- **Twilio Voice Telephony**:
  - `POST /api/v1/voice/token`: Server-side Twilio Access Token generation with `VoiceGrant`.
  - `POST /api/v1/voice/outgoing`: E.164 phone number validation, database call record tracking, and outgoing call dispatch.
  - `POST /api/v1/voice/twiml`: TwiML webhook connecting call audio to backend WebSocket.
  - `POST /api/v1/voice/status`: Call lifecycle tracking (ringing, answered, completed, duration).
- **Real-Time Media Stream WebSocket**: `/ws/voice-stream` decodes incoming μ-law 8kHz audio packets into PCM/WAV and feeds standard 3-second audio windows into the AI pipeline off the event loop.
- **Live Analysis WebSocket**: `/ws/live-analysis/{call_id}` authenticates via JWT, checks call ownership, and streams real-time `trust_score`, `risk_level`, `is_scam`, `is_ai_voice`, `transcript`, `signals`, and `risk_alert` events to mobile clients.

---

## Environment Configuration

Copy `.env.example` to `.env` and configure your credentials:

```env
# Supabase Database
SUPABASE_URL=https://your-supabase-project.supabase.co
SUPABASE_KEY=your-supabase-service-role-key

# JWT Security
SECRET_KEY=your-jwt-secret-key

# Groq AI Service
GROQ_API_KEY=gsk_your_groq_api_key

# SMTP Credentials (OTP Email Delivery)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_FROM_EMAIL=your-email@gmail.com

# Twilio Telephony Credentials
TWILIO_ACCOUNT_SID=ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_API_KEY=SKXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
TWILIO_API_SECRET=your_twilio_api_secret
TWILIO_TWIML_APP_SID=APXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
TWILIO_PHONE_NUMBER=+1234567890

# Public Server URL (e.g. Ngrok URL for Twilio Webhooks & WebSockets)
PUBLIC_SERVER_URL=https://your-ngrok-subdomain.ngrok-free.app

# Real-Time Voice Processing Settings
VOICE_ANALYSIS_CHUNK_SECONDS=3.0
VOICE_ANALYSIS_INTERVAL_SECONDS=3.0
```

---

## Database Setup & Migrations

Run the SQL migration scripts in your Supabase SQL Editor:
1. `supabase/migrations/20260805_add_auth_history_and_safety.sql`
2. `supabase/migrations/20260809_add_voice_calls.sql`

---

## Local Telephony Setup with Ngrok

1. Start local FastAPI server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
2. Start an Ngrok tunnel:
   ```bash
   ngrok http 8000
   ```
3. Set `PUBLIC_SERVER_URL` in `.env` to your Ngrok domain (`https://xxxx.ngrok-free.app`).
4. In your Twilio Console:
   - Set Voice Webhook URL to `https://xxxx.ngrok-free.app/api/v1/voice/twiml`
   - Set Status Callback URL to `https://xxxx.ngrok-free.app/api/v1/voice/status`

---

## Running Tests

Run the backend unit test suite:
```bash
python -m unittest discover tests
```
