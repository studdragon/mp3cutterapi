import re

from fastapi import HTTPException

from config import settings

ALLOWED_FORMATS = {"mp3", "wav", "ogg", "flac", "aac", "m4a", "m4r"}
ALLOWED_SPLIT_MODES = {"count", "duration"}

# ffmpeg accepts forms like "192k", "320K", "128000". Anything else is rejected
# rather than forwarded, so a caller cannot smuggle extra ffmpeg arguments in.
_BITRATE_RE = re.compile(r"^\d{1,7}[kK]?$")


def validate_format(fmt: str) -> str:
    fmt = fmt.lower().strip()
    if fmt not in ALLOWED_FORMATS:
        raise HTTPException(
            400,
            f"Unsupported format: {fmt}. Allowed: {', '.join(sorted(ALLOWED_FORMATS))}",
        )
    return fmt


def validate_bitrate(bitrate: str) -> str | None:
    bitrate = (bitrate or "").strip()
    if not bitrate:
        return None
    if not _BITRATE_RE.match(bitrate):
        raise HTTPException(400, "Bitrate must look like '192k' or '192000'")
    return bitrate


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
    if not (0 <= attack <= 1):
        raise HTTPException(400, "Attack must be between 0 and 1 seconds")
    if not (0 <= release <= 5):
        raise HTTPException(400, "Release must be between 0 and 5 seconds")


def validate_volume(volume_percent: float) -> None:
    if not (0 < volume_percent <= 1000):
        raise HTTPException(400, "Volume must be between 0 and 1000 percent")


def validate_gain_db(gain: float, label: str = "Gain") -> None:
    if not (-24 <= gain <= 24):
        raise HTTPException(400, f"{label} must be between -24 and 24 dB, got {gain}")


def validate_split(mode: str, segments: int, segment_duration: float) -> str:
    """Guard /split, which previously accepted any segment count.

    segments=0 raised ZeroDivisionError (a 500), and a large count or a tiny
    duration turned a modest upload into hundreds of thousands of files.
    """
    mode = (mode or "count").strip().lower()
    if mode not in ALLOWED_SPLIT_MODES:
        raise HTTPException(
            400,
            f"Unsupported split mode: {mode}. Allowed: {', '.join(sorted(ALLOWED_SPLIT_MODES))}",
        )

    if mode == "count":
        if segments < 2:
            raise HTTPException(400, "Segment count must be at least 2")
        if segments > settings.max_split_segments:
            raise HTTPException(
                400,
                f"Segment count must be at most {settings.max_split_segments}",
            )
    else:
        if segment_duration < settings.min_segment_duration_seconds:
            raise HTTPException(
                400,
                f"Segment duration must be at least "
                f"{settings.min_segment_duration_seconds} seconds",
            )

    return mode


def validate_segment_yield(total_ms: int, segment_duration: float) -> None:
    """Reject duration-mode splits that would emit too many files."""
    if segment_duration <= 0:
        raise HTTPException(400, "Segment duration must be greater than zero")
    projected = int(total_ms // int(segment_duration * 1000)) + 1
    if projected > settings.max_split_segments:
        raise HTTPException(
            400,
            f"That segment duration would produce {projected} files; the maximum "
            f"is {settings.max_split_segments}. Use a longer duration.",
        )
