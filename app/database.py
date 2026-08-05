import os
from supabase import create_client, Client
from app.config import settings

# Initialize Supabase Client
supabase_url: str = os.getenv("SUPABASE_URL", "")
supabase_key: str = os.getenv("SUPABASE_KEY", "")

if not supabase_url or not supabase_key:
    print("⚠️ Warning: SUPABASE_URL or SUPABASE_KEY is missing from environment variables.")

supabase: Client = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None

def get_supabase() -> Client:
    if not supabase:
        raise RuntimeError("Supabase client is not configured. Check environment variables.")
    return supabase