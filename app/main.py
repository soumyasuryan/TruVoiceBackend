from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from app.routers import analysis, auth, community

app = FastAPI(
    title="AI Voice Scam Detector API",
    description="Backend API handling Phone Auth and Dual-Pipeline AI Scam Detection.",
    version="1.0.0"
)

# Enable CORS for React Native & Web Frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(auth.router)
app.include_router(analysis.router)
app.include_router(community.router)

@app.get("/")
def root():
    return {"status": "online", "message": "AI Voice Scam Detection API operational."}

# AWS Lambda Handler Wrapper
handler = Mangum(app)
