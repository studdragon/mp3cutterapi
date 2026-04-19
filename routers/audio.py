"""All audio processing API endpoints."""

import asyncio
import logging
import os

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from services.audio_processor import (
    process_8d,
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
    save_upload_sync,
)
from utils.validators import (
    validate_compression,
    validate_crossfade,
    validate_fade,
    validate_file_size,
    validate_format,
    validate_noise_reduction,
    validate_sample_rate,
    validate_semitones,
    validate_speed,
    validate_time_range,
    validate_volume,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _file_response(path: str, fmt: str, temp_dir: str) -> FileResponse:
    return FileResponse(
        path,
        media_type=get_media_type(fmt),
        filename=os.path.basename(path),
        background=BackgroundTask(cleanup_temp_dir, temp_dir),
    )


# ─── Cut / Trim ─────────────────────────────────────────────

@router.post("/cut")
async def cut_audio(
    file: UploadFile = File(...),
    start: float = Form(0),
    end: float = Form(0),
    format: str = Form("mp3"),
    fade_in: float = Form(0),
    fade_out: float = Form(0),
    remove_selection: int = Form(0),
):
    fmt = validate_format(format)
    validate_time_range(start, end)
    temp_dir = create_temp_dir()
    try:
        data = await file.read()
        validate_file_size(data)
        input_path = save_upload_sync(data, file.filename or "audio.mp3", temp_dir)
        output = await asyncio.to_thread(process_cut, input_path, start, end, fmt, fade_in, fade_out, temp_dir, file.filename or "audio", bool(remove_selection))
        return _file_response(output, fmt, temp_dir)
    except HTTPException:
        cleanup_temp_dir(temp_dir)
        raise
    except Exception:
        cleanup_temp_dir(temp_dir)
        logger.exception("Audio processing failed")
        raise HTTPException(500, "Audio processing failed")


# ─── Join / Merge ───────────────────────────────────────────

@router.post("/join")
async def join_audio(
    files: list[UploadFile] = File(...),
    format: str = Form("mp3"),
    crossfade_ms: int = Form(0),
):
    fmt = validate_format(format)
    if len(files) < 2:
        raise HTTPException(400, "At least 2 files required")
    validate_crossfade(crossfade_ms)
    temp_dir = create_temp_dir()
    try:
        input_paths = []
        for i, f in enumerate(files):
            data = await f.read()
            validate_file_size(data)
            ext = os.path.splitext(f.filename or "audio.mp3")[1] or ".mp3"
            path = os.path.join(temp_dir, f"input_{i}{ext}")
            with open(path, "wb") as out:
                out.write(data)
            input_paths.append(path)
        output = await asyncio.to_thread(process_join, input_paths, fmt, crossfade_ms, temp_dir)
        return _file_response(output, fmt, temp_dir)
    except HTTPException:
        cleanup_temp_dir(temp_dir)
        raise
    except Exception:
        cleanup_temp_dir(temp_dir)
        logger.exception("Audio processing failed")
        raise HTTPException(500, "Audio processing failed")


# ─── Convert ────────────────────────────────────────────────

@router.post("/convert")
async def convert_audio(
    file: UploadFile = File(...),
    format: str = Form("mp3"),
    sample_rate: int = Form(44100),
    bitrate: str = Form(""),
):
    fmt = validate_format(format)
    validate_sample_rate(sample_rate)
    temp_dir = create_temp_dir()
    try:
        data = await file.read()
        validate_file_size(data)
        input_path = save_upload_sync(data, file.filename or "audio.mp3", temp_dir)
        output = await asyncio.to_thread(process_convert, input_path, fmt, sample_rate, bitrate or None, temp_dir, file.filename or "audio")
        return _file_response(output, fmt, temp_dir)
    except HTTPException:
        cleanup_temp_dir(temp_dir)
        raise
    except Exception:
        cleanup_temp_dir(temp_dir)
        logger.exception("Audio processing failed")
        raise HTTPException(500, "Audio processing failed")


# ─── Volume ─────────────────────────────────────────────────

@router.post("/volume")
async def change_volume(
    file: UploadFile = File(...),
    volume_percent: float = Form(100),
    format: str = Form("mp3"),
):
    fmt = validate_format(format)
    validate_volume(volume_percent)
    temp_dir = create_temp_dir()
    try:
        data = await file.read()
        validate_file_size(data)
        input_path = save_upload_sync(data, file.filename or "audio.mp3", temp_dir)
        output = await asyncio.to_thread(process_volume, input_path, volume_percent, fmt, temp_dir, file.filename or "audio")
        return _file_response(output, fmt, temp_dir)
    except HTTPException:
        cleanup_temp_dir(temp_dir)
        raise
    except Exception:
        cleanup_temp_dir(temp_dir)
        logger.exception("Audio processing failed")
        raise HTTPException(500, "Audio processing failed")


# ─── Speed ──────────────────────────────────────────────────

@router.post("/speed")
async def change_speed(
    file: UploadFile = File(...),
    speed: float = Form(1.0),
    format: str = Form("mp3"),
):
    fmt = validate_format(format)
    validate_speed(speed)
    temp_dir = create_temp_dir()
    try:
        data = await file.read()
        validate_file_size(data)
        input_path = save_upload_sync(data, file.filename or "audio.mp3", temp_dir)
        output = await asyncio.to_thread(process_speed, input_path, speed, fmt, temp_dir, file.filename or "audio")
        return _file_response(output, fmt, temp_dir)
    except HTTPException:
        cleanup_temp_dir(temp_dir)
        raise
    except Exception:
        cleanup_temp_dir(temp_dir)
        logger.exception("Audio processing failed")
        raise HTTPException(500, "Audio processing failed")


# ─── Pitch ──────────────────────────────────────────────────

@router.post("/pitch")
async def change_pitch(
    file: UploadFile = File(...),
    semitones: float = Form(0),
    format: str = Form("mp3"),
):
    fmt = validate_format(format)
    validate_semitones(semitones)
    temp_dir = create_temp_dir()
    try:
        data = await file.read()
        validate_file_size(data)
        input_path = save_upload_sync(data, file.filename or "audio.mp3", temp_dir)
        output = await asyncio.to_thread(process_pitch, input_path, semitones, fmt, temp_dir, file.filename or "audio")
        return _file_response(output, fmt, temp_dir)
    except HTTPException:
        cleanup_temp_dir(temp_dir)
        raise
    except Exception:
        cleanup_temp_dir(temp_dir)
        logger.exception("Audio processing failed")
        raise HTTPException(500, "Audio processing failed")


# ─── Fade ───────────────────────────────────────────────────

@router.post("/fade")
async def add_fade(
    file: UploadFile = File(...),
    fade_in_sec: float = Form(0),
    fade_out_sec: float = Form(0),
    format: str = Form("mp3"),
):
    fmt = validate_format(format)
    validate_fade(fade_in_sec, fade_out_sec)
    temp_dir = create_temp_dir()
    try:
        data = await file.read()
        validate_file_size(data)
        input_path = save_upload_sync(data, file.filename or "audio.mp3", temp_dir)
        output = await asyncio.to_thread(process_fade, input_path, fade_in_sec, fade_out_sec, fmt, temp_dir, file.filename or "audio")
        return _file_response(output, fmt, temp_dir)
    except HTTPException:
        cleanup_temp_dir(temp_dir)
        raise
    except Exception:
        cleanup_temp_dir(temp_dir)
        logger.exception("Audio processing failed")
        raise HTTPException(500, "Audio processing failed")


# ─── Reverse ────────────────────────────────────────────────

@router.post("/reverse")
async def reverse_audio(
    file: UploadFile = File(...),
    format: str = Form("mp3"),
):
    fmt = validate_format(format)
    temp_dir = create_temp_dir()
    try:
        data = await file.read()
        validate_file_size(data)
        input_path = save_upload_sync(data, file.filename or "audio.mp3", temp_dir)
        output = await asyncio.to_thread(process_reverse, input_path, fmt, temp_dir, file.filename or "audio")
        return _file_response(output, fmt, temp_dir)
    except HTTPException:
        cleanup_temp_dir(temp_dir)
        raise
    except Exception:
        cleanup_temp_dir(temp_dir)
        logger.exception("Audio processing failed")
        raise HTTPException(500, "Audio processing failed")


# ─── Ringtone ───────────────────────────────────────────────

@router.post("/ringtone")
async def make_ringtone(
    file: UploadFile = File(...),
    start: float = Form(0),
    end: float = Form(0),
    fade_in: float = Form(0),
    fade_out: float = Form(0),
    format: str = Form("mp3"),
):
    fmt = validate_format(format)
    validate_time_range(start, end)
    temp_dir = create_temp_dir()
    try:
        data = await file.read()
        validate_file_size(data)
        input_path = save_upload_sync(data, file.filename or "audio.mp3", temp_dir)
        output = await asyncio.to_thread(process_ringtone, input_path, start, end, fade_in, fade_out, fmt, temp_dir, file.filename or "audio")
        return _file_response(output, fmt, temp_dir)
    except HTTPException:
        cleanup_temp_dir(temp_dir)
        raise
    except Exception:
        cleanup_temp_dir(temp_dir)
        logger.exception("Audio processing failed")
        raise HTTPException(500, "Audio processing failed")


# ─── Noise Reduction ────────────────────────────────────────

@router.post("/noise")
async def remove_noise(
    file: UploadFile = File(...),
    reduction: float = Form(50),
    format: str = Form("mp3"),
):
    fmt = validate_format(format)
    validate_noise_reduction(reduction)
    temp_dir = create_temp_dir()
    try:
        data = await file.read()
        validate_file_size(data)
        input_path = save_upload_sync(data, file.filename or "audio.mp3", temp_dir)
        output = await asyncio.to_thread(process_noise, input_path, reduction, fmt, temp_dir, file.filename or "audio")
        return _file_response(output, fmt, temp_dir)
    except HTTPException:
        cleanup_temp_dir(temp_dir)
        raise
    except Exception:
        cleanup_temp_dir(temp_dir)
        logger.exception("Audio processing failed")
        raise HTTPException(500, "Audio processing failed")


# ─── Extract Audio from Video ───────────────────────────────

@router.post("/extract")
async def extract_audio(
    file: UploadFile = File(...),
    format: str = Form("mp3"),
):
    fmt = validate_format(format)
    temp_dir = create_temp_dir()
    try:
        data = await file.read()
        validate_file_size(data)
        input_path = save_upload_sync(data, file.filename or "video.mp4", temp_dir)
        output = await asyncio.to_thread(process_extract, input_path, fmt, temp_dir, file.filename or "video")
        return _file_response(output, fmt, temp_dir)
    except HTTPException:
        cleanup_temp_dir(temp_dir)
        raise
    except Exception:
        cleanup_temp_dir(temp_dir)
        logger.exception("Audio processing failed")
        raise HTTPException(500, "Audio processing failed")


# ─── Dynamic Range Compression ──────────────────────────────

@router.post("/compress")
async def compress_audio(
    file: UploadFile = File(...),
    threshold: float = Form(-24),
    ratio: float = Form(4),
    attack: float = Form(3),
    release: float = Form(250),
    format: str = Form("mp3"),
):
    fmt = validate_format(format)
    validate_compression(threshold, ratio, attack, release)
    temp_dir = create_temp_dir()
    try:
        data = await file.read()
        validate_file_size(data)
        input_path = save_upload_sync(data, file.filename or "audio.mp3", temp_dir)
        output = await asyncio.to_thread(process_compress, input_path, threshold, ratio, attack, release, fmt, temp_dir, file.filename or "audio")
        return _file_response(output, fmt, temp_dir)
    except HTTPException:
        cleanup_temp_dir(temp_dir)
        raise
    except Exception:
        cleanup_temp_dir(temp_dir)
        logger.exception("Audio processing failed")
        raise HTTPException(500, "Audio processing failed")


# ─── Karaoke (Vocal Removal) ────────────────────────────────

@router.post("/karaoke")
async def make_karaoke(
    file: UploadFile = File(...),
    format: str = Form("mp3"),
):
    fmt = validate_format(format)
    temp_dir = create_temp_dir()
    try:
        data = await file.read()
        validate_file_size(data)
        input_path = save_upload_sync(data, file.filename or "audio.mp3", temp_dir)
        output = await asyncio.to_thread(process_karaoke, input_path, fmt, temp_dir, file.filename or "audio")
        return _file_response(output, fmt, temp_dir)
    except HTTPException:
        cleanup_temp_dir(temp_dir)
        raise
    except ValueError as e:
        cleanup_temp_dir(temp_dir)
        raise HTTPException(400, str(e))
    except Exception:
        cleanup_temp_dir(temp_dir)
        logger.exception("Audio processing failed")
        raise HTTPException(500, "Audio processing failed")


# ─── Split ──────────────────────────────────────────────────

@router.post("/split")
async def split_audio(
    file: UploadFile = File(...),
    segments: int = Form(2),
    mode: str = Form("count"),
    segment_duration: float = Form(0),
    format: str = Form("mp3"),
):
    fmt = validate_format(format)
    temp_dir = create_temp_dir()
    try:
        data = await file.read()
        validate_file_size(data)
        input_path = save_upload_sync(data, file.filename or "audio.mp3", temp_dir)
        output = await asyncio.to_thread(process_split, input_path, segments, mode, segment_duration, fmt, temp_dir, file.filename or "audio")
        return _file_response(output, "zip", temp_dir)
    except HTTPException:
        cleanup_temp_dir(temp_dir)
        raise
    except Exception:
        cleanup_temp_dir(temp_dir)
        logger.exception("Audio processing failed")
        raise HTTPException(500, "Audio processing failed")


# ─── Equalizer ──────────────────────────────────────────────

@router.post("/equalize")
async def equalize_audio(
    file: UploadFile = File(...),
    bands: str = Form(""),
    format: str = Form("mp3"),
):
    fmt = validate_format(format)
    if not bands:
        raise HTTPException(400, "EQ bands JSON is required")
    temp_dir = create_temp_dir()
    try:
        data = await file.read()
        validate_file_size(data)
        input_path = save_upload_sync(data, file.filename or "audio.mp3", temp_dir)
        output = await asyncio.to_thread(process_equalize, input_path, bands, fmt, temp_dir, file.filename or "audio")
        return _file_response(output, fmt, temp_dir)
    except HTTPException:
        cleanup_temp_dir(temp_dir)
        raise
    except Exception:
        cleanup_temp_dir(temp_dir)
        logger.exception("Audio processing failed")
        raise HTTPException(500, "Audio processing failed")


# ─── 8D Audio ───────────────────────────────────────────────

@router.post("/8d")
async def create_8d_audio(
    file: UploadFile = File(...),
    speed: float = Form(0.1),
    depth: float = Form(0.8),
    reverb: int = Form(1),
    format: str = Form("mp3"),
):
    fmt = validate_format(format)
    if not (0.02 <= speed <= 0.5):
        raise HTTPException(400, "Speed must be between 0.02 and 0.5 Hz")
    if not (0.1 <= depth <= 1.0):
        raise HTTPException(400, "Depth must be between 0.1 and 1.0")
    temp_dir = create_temp_dir()
    try:
        data = await file.read()
        validate_file_size(data)
        input_path = save_upload_sync(data, file.filename or "audio.mp3", temp_dir)
        output = await asyncio.to_thread(
            process_8d, input_path, speed, depth, bool(reverb), fmt, temp_dir, file.filename or "audio"
        )
        return _file_response(output, fmt, temp_dir)
    except HTTPException:
        cleanup_temp_dir(temp_dir)
        raise
    except Exception:
        cleanup_temp_dir(temp_dir)
        logger.exception("Audio processing failed")
        raise HTTPException(500, "Audio processing failed")


# ─── Bass Boost ──────────────────────────────────────────────

@router.post("/bass-boost")
async def bass_boost_audio(
    file: UploadFile = File(...),
    sub_gain: float = Form(0),
    bass_gain: float = Form(0),
    upper_gain: float = Form(0),
    format: str = Form("mp3"),
):
    fmt = validate_format(format)
    for gain in (sub_gain, bass_gain, upper_gain):
        if not (-24 <= gain <= 24):
            raise HTTPException(400, f"Gain must be between -24 and 24 dB, got {gain}")
    temp_dir = create_temp_dir()
    try:
        data = await file.read()
        validate_file_size(data)
        input_path = save_upload_sync(data, file.filename or "audio.mp3", temp_dir)
        output = await asyncio.to_thread(process_bass_boost, input_path, sub_gain, bass_gain, upper_gain, fmt, temp_dir, file.filename or "audio")
        return _file_response(output, fmt, temp_dir)
    except HTTPException:
        cleanup_temp_dir(temp_dir)
        raise
    except Exception:
        cleanup_temp_dir(temp_dir)
        logger.exception("Audio processing failed")
        raise HTTPException(500, "Audio processing failed")
