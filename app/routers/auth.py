from fastapi import APIRouter, HTTPException, status
from app.schemas import SignUpRequest, LoginRequest, VerifyOTPRequest, ResendOTPRequest, TokenResponse
from app.utils.auth import hash_password, verify_password, create_access_token
from app.utils.otp import generate_otp
from app.database import get_supabase

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/signup/initiate", status_code=status.HTTP_200_OK)
def initiate_signup(payload: SignUpRequest):
    db = get_supabase()
    
    # 1. Check duplicate mobile number in Supabase 'users' table
    existing_user = db.table("users").select("id").eq("phone_number", payload.phone_number).execute()
    if existing_user.data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mobile number already registered. Please login instead."
        )
    
    # 2. Store user credentials temporarily in pending users or upsert OTP
    hashed_pwd = hash_password(payload.password)
    otp_code = generate_otp(payload.phone_number)
    
    # Save OTP to Supabase 'otp_store' table
    db.table("otp_store").upsert({
        "phone_number": payload.phone_number,
        "otp_code": otp_code
    }).execute()
    
    # Pre-register unverified user
    db.table("users").upsert({
        "phone_number": payload.phone_number,
        "password_hash": hashed_pwd,
        "is_verified": False
    }, on_conflict="phone_number").execute()
    
    return {"message": "OTP sent successfully to mobile number.", "phone_number": payload.phone_number}


@router.post("/signup/verify", response_model=TokenResponse)
def verify_signup_otp(payload: VerifyOTPRequest):
    db = get_supabase()
    
    # Fetch stored OTP
    otp_record = db.table("otp_store").select("otp_code").eq("phone_number", payload.phone_number).execute()
    if not otp_record.data or otp_record.data[0]["otp_code"] != payload.otp:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP.")
    
    # Mark user as verified
    db.table("users").update({"is_verified": True}).eq("phone_number", payload.phone_number).execute()
    
    # Delete used OTP
    db.table("otp_store").delete().eq("phone_number", payload.phone_number).execute()
    
    # Generate Token
    token = create_access_token(data={"sub": payload.phone_number})
    return TokenResponse(access_token=token, user_id=payload.phone_number, phone_number=payload.phone_number)


@router.post("/login/initiate")
def initiate_login(payload: LoginRequest):
    db = get_supabase()
    
    # Fetch user
    user_record = db.table("users").select("*").eq("phone_number", payload.phone_number).execute()
    if not user_record.data or not verify_password(payload.password, user_record.data[0]["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid mobile number or password.")
    
    # Generate & store OTP
    otp_code = generate_otp(payload.phone_number)
    db.table("otp_store").upsert({
        "phone_number": payload.phone_number,
        "otp_code": otp_code
    }).execute()
    
    return {"message": "Credentials verified. OTP sent to mobile number.", "phone_number": payload.phone_number}


@router.post("/login/verify", response_model=TokenResponse)
def verify_login_otp(payload: VerifyOTPRequest):
    db = get_supabase()
    
    # Fetch OTP
    otp_record = db.table("otp_store").select("otp_code").eq("phone_number", payload.phone_number).execute()
    if not otp_record.data or otp_record.data[0]["otp_code"] != payload.otp:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP.")
    
    # Consume OTP
    db.table("otp_store").delete().eq("phone_number", payload.phone_number).execute()
    
    token = create_access_token(data={"sub": payload.phone_number})
    return TokenResponse(access_token=token, user_id=payload.phone_number, phone_number=payload.phone_number)


@router.post("/resend-otp")
def resend_otp(payload: ResendOTPRequest):
    db = get_supabase()
    otp_code = generate_otp(payload.phone_number)
    
    db.table("otp_store").upsert({
        "phone_number": payload.phone_number,
        "otp_code": otp_code
    }).execute()
    
    return {"message": "OTP resent successfully."}


@router.post("/logout")
def logout():
    return {"message": "Logged out successfully."}