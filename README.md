# TruVoice Backend — Agora App-to-App Voice Scam Detector & Telephony Engine

This repository powers TruVoice's **app-to-app real-time voice calling engine** using **Agora Real-Time Communication (RTC)** for audio streaming and **Agora Real-Time Messaging (RTM)** for call signaling, user availability, and invitations. Live AI scam detection is preserved via WebSockets.

## Features
- **Phone Authentication**: JWT auth, password hashing, email OTP verification (`/auth/*`).
- **Audio Analysis API**: `POST /api/v1/analyze` for recorded audio files.
- **Agora App-to-App Telephony**:
  - `POST /api/v1/voice/token`: Server-side Agora RTC Token generation (`AGORA_APP_ID`, `AGORA_APP_CERTIFICATE`) for a given `channelName`.
  - `POST /api/v1/voice/log-call`: Creates database record in `voice_calls` table tracking `channel_name`, `user_id` (caller), and `target_user_id` (callee).
  - `POST /api/v1/voice/update-call`: Logs call lifecycle status updates (`answered`, `ended`, `declined`, `busy`, duration).
  - `GET /api/v1/voice/users`: Discovers available online users for direct app-to-app calling.
  - `GET /api/v1/voice/calls`: Fetches call history where user is caller or callee.
- **Live Analysis WebSocket**: `/ws/live-analysis/{call_id}` authenticates via JWT, checks call ownership (caller or target callee), and streams real-time `trust_score`, `risk_level`, `is_scam`, `is_ai_voice`, `transcript`, `signals`, and `risk_alert` events.

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

# Agora RTC Credentials
AGORA_APP_ID=your_agora_app_id
AGORA_APP_CERTIFICATE=your_agora_app_certificate

# Public Server URL
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
3. `supabase/migrations/20260810_agora_voice_calls.sql`

---

## Running Tests

Run the backend unit test suite:
```bash
.venv\Scripts\python -m unittest discover tests
```
