import os
import re
import shutil
import time
import uuid

from fastapi import HTTPException, UploadFile

from config import settings

# 1 MiB chunks: large enough to keep syscall overhead low, small enough that an
# oversized upload is rejected long before it can exhaust memory.
UPLOAD_CHUNK_SIZE = 1024 * 1024

_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def create_temp_dir() -> str:
    path = os.path.join(settings.temp_dir, str(uuid.uuid4()))
    os.makedirs(path, exist_ok=True)
    return path


def safe_base_name(original_name: str, fallback: str = "audio") -> str:
    """Reduce a client-supplied filename to a bare, path-free stem.

    `original_name` arrives from the multipart Content-Disposition header and is
    fully attacker-controlled: browsers strip directories, raw HTTP clients do
    not. Anything derived from it and joined onto a path must go through here or
    a caller can escape the temp directory (e.g. "../../etc/cron.d/x").
    """
    base = os.path.basename(original_name.replace("\\", "/"))
    base = os.path.splitext(base)[0]
    base = _UNSAFE_NAME_CHARS.sub("_", base).strip("._-")
    return base or fallback


def safe_extension(original_name: str, fallback: str = ".tmp") -> str:
    ext = os.path.splitext(os.path.basename(original_name.replace("\\", "/")))[1]
    ext = _UNSAFE_NAME_CHARS.sub("", ext)
    if not ext or ext == ".":
        return fallback
    return ext[:16]


async def save_upload_streaming(
    upload: UploadFile,
    dest_dir: str,
    fallback_name: str = "audio.mp3",
    filename_prefix: str = "input",
) -> str:
    """Stream an upload to disk, aborting as soon as it exceeds the size cap.

    Reading the whole body first and checking the length afterwards (the previous
    behaviour) meant a multi-gigabyte POST was fully materialised in memory
    before being rejected.
    """
    ext = safe_extension(upload.filename or fallback_name, ".tmp")
    path = os.path.join(dest_dir, f"{filename_prefix}{ext}")
    max_bytes = settings.max_file_size_bytes
    total = 0

    with open(path, "wb") as out:
        while True:
            chunk = await upload.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(
                    413,
                    f"File too large. Maximum allowed size is "
                    f"{settings.max_file_size_mb} MB.",
                )
            out.write(chunk)

    if total == 0:
        raise HTTPException(400, "Uploaded file is empty")
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
    return os.path.join(temp_dir, f"{safe_base_name(original_name)}_{suffix}.{fmt}")


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
