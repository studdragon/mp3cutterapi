"""All audio processing API endpoints.

Every handler follows the same shape, so the boilerplate lives in `_process`:
create a temp dir, stream the upload to disk under a size cap, run the blocking
processor in a worker thread, and hand back a FileResponse that deletes the temp
dir once the body has been flushed. The temp dir is removed on every exit path.
"""

import asyncio
import logging
import os
import subprocess
from typing import Any, Callable

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from services.audio_processor import (
    process_bass_boost,
    process_compress,
    process_convert,
    process_cut,
    process_equalize,
    process_extract,
    process_fade,
    process_join,
    process_karaoke,
    process_noise,
    process_pitch,
    process_reverse,
    process_ringtone,
    process_speed,
    process_split,
    process_volume,
)
from utils.file_helpers import (
    cleanup_temp_dir,
    create_temp_dir,
    get_media_type,
    save_upload_streaming,
)
from utils.validators import (
    validate_bitrate,
    validate_compression,
    validate_crossfade,
    validate_fade,
    validate_format,
    validate_gain_db,
    validate_noise_reduction,
    validate_sample_rate,
    validate_semitones,
    validate_speed,
    validate_split,
    validate_time_range,
    validate_volume,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Called with (input_path, original_name, temp_dir); returns the processor's args.
ArgBuilder = Callable[[str, str, str], tuple[Any, ...]]

# ffmpeg is invoked with a 300s timeout; pydub's own export has none, so a
# pathological file can still stall a worker thread.
TIMEOUT_ERRORS = (subprocess.TimeoutExpired, asyncio.TimeoutError)


def _require_ffmpeg(request: Request) -> None:
    """Fail fast and legibly when ffmpeg is missing.

    Startup already probes for it; without this check every endpoint surfaced the
    same opaque 500 instead of telling the caller the service is misconfigured.
    """
    if not getattr(request.app.state, "ffmpeg_available", False):
        raise HTTPException(
            503,
            "Audio processing is unavailable: ffmpeg is not installed on the server.",
        )


def _file_response(path: str, fmt: str, temp_dir: str) -> FileResponse:
    return FileResponse(
        path,
        media_type=get_media_type(fmt),
        filename=os.path.basename(path),
        background=BackgroundTask(cleanup_temp_dir, temp_dir),
    )


def _original_name(file: UploadFile, fallback: str) -> str:
    return os.path.basename((file.filename or fallback).replace("\\", "/"))


async def _process(
    request: Request,
    file: UploadFile,
    processor: Callable[..., str],
    build_args: ArgBuilder,
    response_fmt: str,
    fallback_name: str = "audio.mp3",
) -> FileResponse:
    """Run `processor` over a single uploaded file.

    `response_fmt` is the format used for the Content-Type, which is not always
    the requested audio format -- /split returns a zip.
    """
    _require_ffmpeg(request)
    temp_dir = create_temp_dir()
    try:
        original_name = _original_name(file, fallback_name)
        input_path = await save_upload_streaming(file, temp_dir, fallback_name)
        output = await asyncio.to_thread(
            processor, *build_args(input_path, original_name, temp_dir)
        )
        return _file_response(output, response_fmt, temp_dir)
    except HTTPException:
        cleanup_temp_dir(temp_dir)
        raise
    except ValueError as exc:
        # Processors raise ValueError for input the caller can correct (a mono
        # file for vocal removal, a malformed EQ payload, an impossible segment
        # count). These were previously reported as opaque 500s.
        cleanup_temp_dir(temp_dir)
        raise HTTPException(400, str(exc)) from exc
    except TIMEOUT_ERRORS as exc:
        cleanup_temp_dir(temp_dir)
        logger.warning("Audio processing timed out")
        raise HTTPException(504, "Audio processing timed out") from exc
    except Exception:
        cleanup_temp_dir(temp_dir)
        logger.exception("Audio processing failed")
        raise HTTPException(500, "Audio processing failed")


# ─── Cut / Trim ─────────────────────────────────────────────

@router.post("/cut")
async def cut_audio(
    request: Request,
    file: UploadFile = File(...),
    start: float = Form(0),
    end: float = Form(0),
    format: str = Form("mp3"),
    fade_in: float = Form(0),
    fade_out: float = Form(0),
    remove_selection: bool = Form(False),
):
    fmt = validate_format(format)
    validate_time_range(start, end)
    validate_fade(fade_in, fade_out)
    return await _process(
        request,
        file,
        process_cut,
        lambda path, name, tmp: (
            path, start, end, fmt, fade_in, fade_out, tmp, name, remove_selection,
        ),
        fmt,
    )


# ─── Join / Merge ───────────────────────────────────────────

@router.post("/join")
async def join_audio(
    request: Request,
    files: list[UploadFile] = File(...),
    format: str = Form("mp3"),
    crossfade_ms: int = Form(0),
):
    fmt = validate_format(format)
    if len(files) < 2:
        raise HTTPException(400, "At least 2 files required")
    if len(files) > 50:
        raise HTTPException(400, "At most 50 files can be merged at once")
    validate_crossfade(crossfade_ms)

    _require_ffmpeg(request)
    temp_dir = create_temp_dir()
    try:
        input_paths = [
            await save_upload_streaming(
                upload, temp_dir, "audio.mp3", filename_prefix=f"input_{i}"
            )
            for i, upload in enumerate(files)
        ]
        output = await asyncio.to_thread(
            process_join, input_paths, fmt, crossfade_ms, temp_dir
        )
        return _file_response(output, fmt, temp_dir)
    except HTTPException:
        cleanup_temp_dir(temp_dir)
        raise
    except ValueError as exc:
        cleanup_temp_dir(temp_dir)
        raise HTTPException(400, str(exc)) from exc
    except TIMEOUT_ERRORS as exc:
        cleanup_temp_dir(temp_dir)
        raise HTTPException(504, "Audio processing timed out") from exc
    except Exception:
        cleanup_temp_dir(temp_dir)
        logger.exception("Audio processing failed")
        raise HTTPException(500, "Audio processing failed")


# ─── Convert ────────────────────────────────────────────────

@router.post("/convert")
async def convert_audio(
    request: Request,
    file: UploadFile = File(...),
    format: str = Form("mp3"),
    sample_rate: int = Form(44100),
    bitrate: str = Form(""),
):
    fmt = validate_format(format)
    validate_sample_rate(sample_rate)
    checked_bitrate = validate_bitrate(bitrate)
    return await _process(
        request,
        file,
        process_convert,
        lambda path, name, tmp: (path, fmt, sample_rate, checked_bitrate, tmp, name),
        fmt,
    )


# ─── Volume ─────────────────────────────────────────────────

@router.post("/volume")
async def change_volume(
    request: Request,
    file: UploadFile = File(...),
    volume_percent: float = Form(100),
    format: str = Form("mp3"),
):
    fmt = validate_format(format)
    validate_volume(volume_percent)
    return await _process(
        request,
        file,
        process_volume,
        lambda path, name, tmp: (path, volume_percent, fmt, tmp, name),
        fmt,
    )


# ─── Speed ──────────────────────────────────────────────────

@router.post("/speed")
async def change_speed(
    request: Request,
    file: UploadFile = File(...),
    speed: float = Form(1.0),
    format: str = Form("mp3"),
):
    fmt = validate_format(format)
    validate_speed(speed)
    return await _process(
        request,
        file,
        process_speed,
        lambda path, name, tmp: (path, speed, fmt, tmp, name),
        fmt,
    )


# ─── Pitch ──────────────────────────────────────────────────

@router.post("/pitch")
async def change_pitch(
    request: Request,
    file: UploadFile = File(...),
    semitones: float = Form(0),
    format: str = Form("mp3"),
):
    fmt = validate_format(format)
    validate_semitones(semitones)
    return await _process(
        request,
        file,
        process_pitch,
        lambda path, name, tmp: (path, semitones, fmt, tmp, name),
        fmt,
    )


# ─── Fade ───────────────────────────────────────────────────

@router.post("/fade")
async def add_fade(
    request: Request,
    file: UploadFile = File(...),
    fade_in_sec: float = Form(0),
    fade_out_sec: float = Form(0),
    format: str = Form("mp3"),
):
    fmt = validate_format(format)
    validate_fade(fade_in_sec, fade_out_sec)
    return await _process(
        request,
        file,
        process_fade,
        lambda path, name, tmp: (path, fade_in_sec, fade_out_sec, fmt, tmp, name),
        fmt,
    )


# ─── Reverse ────────────────────────────────────────────────

@router.post("/reverse")
async def reverse_audio(
    request: Request,
    file: UploadFile = File(...),
    format: str = Form("mp3"),
):
    fmt = validate_format(format)
    return await _process(
        request,
        file,
        process_reverse,
        lambda path, name, tmp: (path, fmt, tmp, name),
        fmt,
    )


# ─── Ringtone ───────────────────────────────────────────────

@router.post("/ringtone")
async def make_ringtone(
    request: Request,
    file: UploadFile = File(...),
    start: float = Form(0),
    end: float = Form(0),
    fade_in: float = Form(0),
    fade_out: float = Form(0),
    format: str = Form("mp3"),
):
    fmt = validate_format(format)
    validate_time_range(start, end)
    validate_fade(fade_in, fade_out)
    return await _process(
        request,
        file,
        process_ringtone,
        lambda path, name, tmp: (
            path, start, end, fade_in, fade_out, fmt, tmp, name,
        ),
        fmt,
    )


# ─── Noise Reduction ────────────────────────────────────────

@router.post("/noise")
async def remove_noise(
    request: Request,
    file: UploadFile = File(...),
    reduction: float = Form(50),
    format: str = Form("mp3"),
):
    fmt = validate_format(format)
    validate_noise_reduction(reduction)
    return await _process(
        request,
        file,
        process_noise,
        lambda path, name, tmp: (path, reduction, fmt, tmp, name),
        fmt,
    )


# ─── Extract Audio from Video ───────────────────────────────

@router.post("/extract")
async def extract_audio(
    request: Request,
    file: UploadFile = File(...),
    format: str = Form("mp3"),
):
    fmt = validate_format(format)
    return await _process(
        request,
        file,
        process_extract,
        lambda path, name, tmp: (path, fmt, tmp, name),
        fmt,
        fallback_name="video.mp4",
    )


# ─── Dynamic Range Compression ──────────────────────────────

@router.post("/compress")
async def compress_audio(
    request: Request,
    file: UploadFile = File(...),
    threshold: float = Form(-24),
    ratio: float = Form(4),
    attack: float = Form(0.003),
    release: float = Form(0.25),
    format: str = Form("mp3"),
):
    fmt = validate_format(format)
    validate_compression(threshold, ratio, attack, release)
    return await _process(
        request,
        file,
        process_compress,
        lambda path, name, tmp: (
            path, threshold, ratio, attack, release, fmt, tmp, name,
        ),
        fmt,
    )


# ─── Karaoke (Vocal Removal) ────────────────────────────────

@router.post("/karaoke")
async def make_karaoke(
    request: Request,
    file: UploadFile = File(...),
    format: str = Form("mp3"),
):
    fmt = validate_format(format)
    return await _process(
        request,
        file,
        process_karaoke,
        lambda path, name, tmp: (path, fmt, tmp, name),
        fmt,
    )


# ─── Split ──────────────────────────────────────────────────

@router.post("/split")
async def split_audio(
    request: Request,
    file: UploadFile = File(...),
    segments: int = Form(2),
    mode: str = Form("count"),
    segment_duration: float = Form(0),
    format: str = Form("mp3"),
):
    fmt = validate_format(format)
    checked_mode = validate_split(mode, segments, segment_duration)
    # The response body is a zip of segments, not a single audio file.
    return await _process(
        request,
        file,
        process_split,
        lambda path, name, tmp: (
            path, segments, checked_mode, segment_duration, fmt, tmp, name,
        ),
        "zip",
    )


# ─── Equalizer ──────────────────────────────────────────────

@router.post("/equalize")
async def equalize_audio(
    request: Request,
    file: UploadFile = File(...),
    bands: str = Form(""),
    format: str = Form("mp3"),
):
    fmt = validate_format(format)
    if not bands:
        raise HTTPException(400, "EQ bands JSON is required")
    return await _process(
        request,
        file,
        process_equalize,
        lambda path, name, tmp: (path, bands, fmt, tmp, name),
        fmt,
    )


# ─── Bass Boost ──────────────────────────────────────────────

@router.post("/bass-boost")
async def bass_boost_audio(
    request: Request,
    file: UploadFile = File(...),
    sub_gain: float = Form(0),
    bass_gain: float = Form(0),
    upper_gain: float = Form(0),
    format: str = Form("mp3"),
):
    fmt = validate_format(format)
    validate_gain_db(sub_gain, "Sub bass gain")
    validate_gain_db(bass_gain, "Bass gain")
    validate_gain_db(upper_gain, "Upper bass gain")
    return await _process(
        request,
        file,
        process_bass_boost,
        lambda path, name, tmp: (
            path, sub_gain, bass_gain, upper_gain, fmt, tmp, name,
        ),
        fmt,
    )
