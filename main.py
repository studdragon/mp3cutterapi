"""MP3 Cutter API — FastAPI backend for audio processing."""

import logging
import os
import shutil
import subprocess
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from routers.audio import router as audio_router
from utils.file_helpers import cleanup_old_temp_dirs
from utils.rate_limit import SlidingWindowRateLimiter, client_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ffmpeg_available() -> bool:
    try:
        result = subprocess.run(
            [settings.ffmpeg_path, "-version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
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
        logger.warning(
            "ffmpeg not found at %r. Processing endpoints will return 503.",
            settings.ffmpeg_path,
        )
    logger.info("MP3 Cutter API started. Temp dir: %s", settings.temp_dir)
    yield
    # Shutdown — clean remaining temp files
    if os.path.isdir(settings.temp_dir):
        shutil.rmtree(settings.temp_dir, ignore_errors=True)


app = FastAPI(
    title="MP3 Cutter API",
    version="1.0.0",
    lifespan=lifespan,
)

rate_limiter = SlidingWindowRateLimiter(
    max_requests=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window_seconds,
)


@app.middleware("http")
async def guard_request(request: Request, call_next):
    """Reject oversized and over-frequent requests before any work is done.

    Content-Length is checked here so a huge upload is refused before
    python-multipart buffers it to disk; save_upload_streaming enforces the same
    cap while reading, which covers chunked requests that omit the header.
    """
    if request.method == "POST" and request.url.path.startswith("/api/"):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > settings.max_file_size_bytes:
                    return JSONResponse(
                        {
                            "detail": f"Request too large. Maximum allowed size is "
                            f"{settings.max_file_size_mb} MB."
                        },
                        status_code=413,
                    )
            except ValueError:
                return JSONResponse({"detail": "Invalid Content-Length"}, 400)

        allowed, retry_after = rate_limiter.check(
            client_key(request.scope.get("client"), request.headers.get("x-forwarded-for"))
        )
        if not allowed:
            return JSONResponse(
                {"detail": "Too many requests. Please wait and try again."},
                status_code=429,
                headers={"Retry-After": str(int(retry_after))},
            )

    return await call_next(request)


# CORS. expose_headers is required for the browser to read Content-Disposition
# cross-origin; without it the client cannot recover the real download filename
# and falls back to a guessed name and extension.
# rstrip("/") because CORS origins are compared exactly: a stray trailing slash in
# ALLOWED_ORIGINS silently fails to match and looks identical to the origin having
# been left out altogether.
origins = [o.strip().rstrip("/") for o in settings.allowed_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    expose_headers=["Content-Disposition"],
)

# Routes
app.include_router(audio_router, prefix="/api")


@app.get("/api/health")
async def health_check():
    ffmpeg_ok = getattr(app.state, "ffmpeg_available", False)
    return JSONResponse(
        {
            "status": "ok" if ffmpeg_ok else "degraded",
            "ffmpeg": ffmpeg_ok,
            "version": "1.0.0",
            "max_file_size_mb": settings.max_file_size_mb,
        },
        status_code=200 if ffmpeg_ok else 503,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
