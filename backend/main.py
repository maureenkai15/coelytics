from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from backend.api.coe_routes import router as coe_router
from backend.models.database import init_db

app = FastAPI(
    title="COElytics API",
    description="Singapore COE market intelligence platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
def startup():
    init_db()

# Include routes
app.include_router(coe_router)

@app.get("/")
def health_check():
    return {
        "status": "running",
        "app": "COElytics API",
        "version": "1.0.0",
        "docs": "/docs"
    }