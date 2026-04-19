"""MP3 Cutter API — FastAPI backend for audio processing."""

import os
import shutil
import subprocess
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routers.audio import router as audio_router
from utils.file_helpers import cleanup_old_temp_dirs


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(
            [settings.ffmpeg_path, "-version"],
            capture_output=True,
            timeout=5,
        )
        return True
    except Exception:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    os.makedirs(settings.temp_dir, exist_ok=True)
    cleanup_old_temp_dirs()
    ffmpeg_ok = _ffmpeg_available()
    app.state.ffmpeg_available = ffmpeg_ok
    if not ffmpeg_ok:
        print("WARNING: ffmpeg not found on PATH. Some features will not work.")
    print(f"MP3 Cutter API started. Temp dir: {settings.temp_dir}")
    yield
    # Shutdown — clean remaining temp files
    if os.path.isdir(settings.temp_dir):
        shutil.rmtree(settings.temp_dir, ignore_errors=True)


app = FastAPI(
    title="MP3 Cutter API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(audio_router, prefix="/api")


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "ffmpeg": getattr(app.state, "ffmpeg_available", False),
        "version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
