import os
import uuid
import shutil
import time

from config import settings


def create_temp_dir() -> str:
    path = os.path.join(settings.temp_dir, str(uuid.uuid4()))
    os.makedirs(path, exist_ok=True)
    return path


def save_upload_sync(data: bytes, filename: str, dest_dir: str) -> str:
    ext = os.path.splitext(filename)[1] or ".tmp"
    path = os.path.join(dest_dir, f"input{ext}")
    with open(path, "wb") as f:
        f.write(data)
    return path


def cleanup_temp_dir(path: str) -> None:
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
    except Exception:
        pass


def cleanup_old_temp_dirs(max_age_seconds: int = 3600) -> None:
    """Remove temp directories older than max_age_seconds (crash recovery)."""
    if not os.path.isdir(settings.temp_dir):
        return
    now = time.time()
    for name in os.listdir(settings.temp_dir):
        dirpath = os.path.join(settings.temp_dir, name)
        if os.path.isdir(dirpath):
            try:
                age = now - os.path.getmtime(dirpath)
                if age > max_age_seconds:
                    shutil.rmtree(dirpath)
            except Exception:
                pass


def get_output_path(temp_dir: str, original_name: str, suffix: str, fmt: str) -> str:
    base = os.path.splitext(os.path.basename(original_name))[0]
    return os.path.join(temp_dir, f"{base}_{suffix}.{fmt}")


MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "aac": "audio/aac",
    "m4a": "audio/mp4",
    "m4r": "audio/x-m4r",
    "zip": "application/zip",
}


def get_media_type(fmt: str) -> str:
    return MEDIA_TYPES.get(fmt, "application/octet-stream")
