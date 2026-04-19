from fastapi import HTTPException

from config import settings

ALLOWED_FORMATS = {"mp3", "wav", "ogg", "flac", "aac", "m4a", "m4r", "opus"}


def validate_file_size(data: bytes) -> None:
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            413,
            f"File too large. Maximum allowed size is {settings.max_file_size_mb} MB.",
        )


def validate_format(fmt: str) -> str:
    fmt = fmt.lower().strip()
    if fmt not in ALLOWED_FORMATS:
        raise HTTPException(400, f"Unsupported format: {fmt}. Allowed: {', '.join(sorted(ALLOWED_FORMATS))}")
    return fmt


def validate_time_range(start: float, end: float) -> None:
    if start < 0:
        raise HTTPException(400, "Start time cannot be negative")
    if end <= start:
        raise HTTPException(400, "End time must be greater than start time")


def validate_speed(speed: float) -> None:
    if not (0.25 <= speed <= 4.0):
        raise HTTPException(400, "Speed must be between 0.25 and 4.0")


def validate_semitones(semitones: float) -> None:
    if not (-12 <= semitones <= 12):
        raise HTTPException(400, "Semitones must be between -12 and 12")


def validate_sample_rate(sample_rate: int) -> None:
    if not (8000 <= sample_rate <= 192000):
        raise HTTPException(400, "Sample rate must be between 8000 and 192000 Hz")


def validate_crossfade(crossfade_ms: int) -> None:
    if crossfade_ms < 0 or crossfade_ms > 30000:
        raise HTTPException(400, "Crossfade must be between 0 and 30000 ms")


def validate_fade(fade_in: float, fade_out: float) -> None:
    if fade_in < 0 or fade_out < 0:
        raise HTTPException(400, "Fade durations cannot be negative")


def validate_noise_reduction(reduction: float) -> None:
    if not (0 <= reduction <= 100):
        raise HTTPException(400, "Noise reduction must be between 0 and 100")


def validate_compression(
    threshold: float, ratio: float, attack: float, release: float,
) -> None:
    if not (-100 <= threshold <= 0):
        raise HTTPException(400, "Threshold must be between -100 and 0 dB")
    if not (1 <= ratio <= 100):
        raise HTTPException(400, "Ratio must be between 1 and 100")
    if not (0 <= attack <= 2000):
        raise HTTPException(400, "Attack must be between 0 and 2000 ms")
    if not (0 <= release <= 9000):
        raise HTTPException(400, "Release must be between 0 and 9000 ms")


def validate_volume(volume_percent: float) -> None:
    if not (0 < volume_percent <= 1000):
        raise HTTPException(400, "Volume must be between 0 and 1000 percent")
