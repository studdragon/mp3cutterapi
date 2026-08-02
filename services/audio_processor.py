"""All audio processing functions. Each takes input path + params, returns output path."""

import json
import math
import os
import subprocess
import zipfile

import numpy as np
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from config import settings
from utils.file_helpers import get_output_path, safe_base_name


class UnsupportedAudioError(ValueError):
    """Raised when the upload itself is the problem, so callers get a 400.

    A corrupt or non-audio upload previously bubbled up as a generic exception
    and was reported as "Audio processing failed" with a 500.
    """


# Markers ffmpeg emits when the input, not the operation, is at fault.
_DECODE_ERROR_MARKERS = (
    "invalid data found",
    "does not contain any stream",
    "no such file or directory",
    "unknown format",
    "moov atom not found",
    "end of file",
)


def _load_audio(input_path: str) -> AudioSegment:
    try:
        return AudioSegment.from_file(input_path)
    except CouldntDecodeError as exc:
        raise UnsupportedAudioError(
            "Could not read this file. It may be corrupt or in an unsupported format."
        ) from exc
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise UnsupportedAudioError(
            "Could not read this file. It may be corrupt or in an unsupported format."
        ) from exc


def _segment_from_samples(
    samples: np.ndarray, frame_rate: int, sample_width: int, channels: int
) -> AudioSegment:
    """Rebuild an AudioSegment from float samples at the source's bit depth.

    Hardcoding int16 here silently clipped 24- and 32-bit sources to +/-32767,
    which destroyed the audio instead of just resampling it.
    """
    width = sample_width if sample_width in (1, 2, 4) else 2
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[width]
    peak = float(np.iinfo(dtype).max)
    clipped = np.clip(samples, -peak, peak).astype(dtype)
    return AudioSegment(
        clipped.tobytes(),
        frame_rate=frame_rate,
        sample_width=width,
        channels=channels,
    )


def _export(audio: AudioSegment, path: str, fmt: str, bitrate: str | None = None) -> str:
    params: list[str] = []
    if fmt == "m4r":
        # M4R is an AAC/MP4 container with a .m4r extension; export as mp4 then
        # rename. Strip only the trailing extension -- str.replace would also
        # rewrite a ".m4r" occurring inside the stem.
        mp4_path = f"{os.path.splitext(path)[0]}.mp4"
        if bitrate:
            params += ["-b:a", bitrate]
        audio.export(mp4_path, format="mp4", parameters=params or None)
        os.rename(mp4_path, path)
        return path
    if bitrate and fmt in ("mp3", "aac", "ogg"):
        params += ["-b:a", bitrate]
    audio.export(path, format=fmt, parameters=params or None)
    return path


def _run_ffmpeg(args: list[str]) -> None:
    # -nostdin stops ffmpeg blocking forever on a prompt when stdin is closed,
    # and -y makes overwrites explicit rather than relying on a fresh temp dir.
    result = subprocess.run(
        [settings.ffmpeg_path, "-nostdin", "-hide_banner", "-y"] + args,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        stderr = result.stderr or ""
        lowered = stderr.lower()
        if any(marker in lowered for marker in _DECODE_ERROR_MARKERS):
            raise UnsupportedAudioError(
                "Could not read this file. It may be corrupt or in an "
                "unsupported format."
            )
        raise RuntimeError(f"ffmpeg error: {stderr[:500]}")


# â”€â”€â”€ 1. Cut / Trim â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def process_cut(
    input_path: str,
    start: float,
    end: float,
    fmt: str,
    fade_in: float,
    fade_out: float,
    temp_dir: str,
    original_name: str,
    remove_selection: bool = False,
) -> str:
    audio = _load_audio(input_path)
    start_ms = int(start * 1000)
    end_ms = int(end * 1000)
    if remove_selection:
        # Delete the selected span and stitch the surrounding audio together.
        trimmed = audio[:start_ms] + audio[end_ms:]
        if len(trimmed) == 0:
            raise ValueError(
                "Removing that selection would leave an empty file. "
                "Select a smaller range."
            )
    else:
        trimmed = audio[start_ms:end_ms]
    if fade_in > 0:
        trimmed = trimmed.fade_in(int(fade_in * 1000))
    if fade_out > 0:
        trimmed = trimmed.fade_out(int(fade_out * 1000))
    out = get_output_path(temp_dir, original_name, "trimmed", fmt)
    return _export(trimmed, out, fmt)


# â”€â”€â”€ 2. Join / Merge â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def process_join(
    input_paths: list[str],
    fmt: str,
    crossfade_ms: int,
    temp_dir: str,
) -> str:
    segments = [_load_audio(p) for p in input_paths]
    if not segments:
        raise ValueError("No audio files provided")
    combined = segments[0]
    for seg in segments[1:]:
        if crossfade_ms > 0:
            combined = combined.append(seg, crossfade=min(crossfade_ms, len(combined), len(seg)))
        else:
            combined = combined + seg
    out = os.path.join(temp_dir, f"merged.{fmt}")
    return _export(combined, out, fmt)


# â”€â”€â”€ 3. Convert â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def process_convert(
    input_path: str,
    fmt: str,
    sample_rate: int,
    bitrate: str | None,
    temp_dir: str,
    original_name: str,
) -> str:
    audio = _load_audio(input_path)
    if sample_rate:
        audio = audio.set_frame_rate(sample_rate)
    out = get_output_path(temp_dir, original_name, "converted", fmt)
    return _export(audio, out, fmt, bitrate)


# â”€â”€â”€ 4. Volume â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def process_volume(
    input_path: str,
    volume_percent: float,
    fmt: str,
    temp_dir: str,
    original_name: str,
) -> str:
    audio = _load_audio(input_path)
    if volume_percent <= 0:
        volume_percent = 0.01
    db_change = 20 * math.log10(volume_percent / 100)
    adjusted = audio + db_change
    out = get_output_path(temp_dir, original_name, "volume", fmt)
    return _export(adjusted, out, fmt)


# â”€â”€â”€ 5. Speed (without pitch change) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _build_atempo_chain(speed: float) -> str:
    """atempo accepts 0.5â€“100.0; chain for extreme values."""
    filters = []
    remaining = speed
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    filters.append(f"atempo={remaining:.6f}")
    return ",".join(filters)


def process_speed(
    input_path: str,
    speed: float,
    fmt: str,
    temp_dir: str,
    original_name: str,
) -> str:
    out = get_output_path(temp_dir, original_name, "speed", fmt)
    filter_str = _build_atempo_chain(speed)
    _run_ffmpeg(["-i", input_path, "-filter:a", filter_str, "-vn", out])
    return out


# â”€â”€â”€ 6. Pitch (without speed change) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def process_pitch(
    input_path: str,
    semitones: float,
    fmt: str,
    temp_dir: str,
    original_name: str,
) -> str:
    out = get_output_path(temp_dir, original_name, "pitch", fmt)
    audio = _load_audio(input_path)
    sr = audio.frame_rate
    ratio = 2 ** (semitones / 12)
    new_sr = int(sr * ratio)
    # asetrate shifts pitch but also changes playback speed; aresample only
    # returns the stream to the original sample rate, it does not restore the
    # duration. atempo undoes the speed change so only the pitch moves.
    filter_str = (
        f"asetrate={new_sr},aresample={sr},{_build_atempo_chain(1 / ratio)}"
    )
    _run_ffmpeg(["-i", input_path, "-af", filter_str, "-vn", out])
    return out


# â”€â”€â”€ 7. Fade â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def process_fade(
    input_path: str,
    fade_in_sec: float,
    fade_out_sec: float,
    fmt: str,
    temp_dir: str,
    original_name: str,
) -> str:
    audio = _load_audio(input_path)
    if fade_in_sec > 0:
        audio = audio.fade_in(int(fade_in_sec * 1000))
    if fade_out_sec > 0:
        audio = audio.fade_out(int(fade_out_sec * 1000))
    out = get_output_path(temp_dir, original_name, "faded", fmt)
    return _export(audio, out, fmt)


# â”€â”€â”€ 8. Reverse â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def process_reverse(
    input_path: str,
    fmt: str,
    temp_dir: str,
    original_name: str,
) -> str:
    audio = _load_audio(input_path).reverse()
    out = get_output_path(temp_dir, original_name, "reversed", fmt)
    return _export(audio, out, fmt)


# â”€â”€â”€ 9. Ringtone â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def process_ringtone(
    input_path: str,
    start: float,
    end: float,
    fade_in: float,
    fade_out: float,
    fmt: str,
    temp_dir: str,
    original_name: str,
) -> str:
    audio = _load_audio(input_path)
    start_ms = int(start * 1000)
    end_ms = int(end * 1000)
    max_ms = int(settings.max_ringtone_seconds * 1000)
    duration_ms = min(end_ms - start_ms, max_ms)
    trimmed = audio[start_ms : start_ms + duration_ms]
    if len(trimmed) == 0:
        raise ValueError("The selected range falls outside the audio")
    if fade_in > 0:
        trimmed = trimmed.fade_in(int(fade_in * 1000))
    if fade_out > 0:
        trimmed = trimmed.fade_out(int(fade_out * 1000))
    out = get_output_path(temp_dir, original_name, "ringtone", fmt)
    return _export(trimmed, out, fmt)


# â”€â”€â”€ 10. Noise Reduction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def process_noise(
    input_path: str,
    reduction: float,
    fmt: str,
    temp_dir: str,
    original_name: str,
) -> str:
    import noisereduce as nr

    audio = _load_audio(input_path)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    channels = audio.channels

    if channels > 1:
        samples = samples.reshape((-1, channels))
        reduced_channels = []
        for ch in range(channels):
            reduced = nr.reduce_noise(
                y=samples[:, ch],
                sr=audio.frame_rate,
                prop_decrease=reduction / 100.0,
                stationary=True,
            )
            reduced_channels.append(reduced)
        reduced_all = np.column_stack(reduced_channels).flatten()
    else:
        reduced_all = nr.reduce_noise(
            y=samples,
            sr=audio.frame_rate,
            prop_decrease=reduction / 100.0,
            stationary=True,
        )

    result = _segment_from_samples(
        reduced_all, audio.frame_rate, audio.sample_width, channels
    )
    out = get_output_path(temp_dir, original_name, "denoised", fmt)
    return _export(result, out, fmt)


# â”€â”€â”€ 11. Extract Audio from Video â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def process_extract(
    input_path: str,
    fmt: str,
    temp_dir: str,
    original_name: str,
) -> str:
    out = get_output_path(temp_dir, original_name, "audio", fmt)
    _run_ffmpeg(["-i", input_path, "-vn", out])
    return out


# â”€â”€â”€ 12. Dynamic Range Compression â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def process_compress(
    input_path: str,
    threshold: float,
    ratio: float,
    attack: float,
    release: float,
    fmt: str,
    temp_dir: str,
    original_name: str,
) -> str:
    out = get_output_path(temp_dir, original_name, "compressed", fmt)
    filter_str = (
        f"acompressor=threshold={threshold}dB"
        f":ratio={ratio}"
        f":attack={attack}"
        f":release={release}"
    )
    _run_ffmpeg(["-i", input_path, "-af", filter_str, "-vn", out])
    return out


# â”€â”€â”€ 13. Karaoke (Vocal Removal) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def process_karaoke(
    input_path: str,
    fmt: str,
    temp_dir: str,
    original_name: str,
) -> str:
    audio = _load_audio(input_path)
    if audio.channels < 2:
        raise ValueError("Vocal removal requires stereo audio")

    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples = samples.reshape((-1, audio.channels))
    # Vocals usually sit centred in the stereo image, so subtracting the channels
    # cancels them and leaves the instrumental. Only the first two channels carry
    # that relationship.
    instrumental = (samples[:, 0] - samples[:, 1]) / 2

    result = _segment_from_samples(
        instrumental, audio.frame_rate, audio.sample_width, 1
    )
    out = get_output_path(temp_dir, original_name, "instrumental", fmt)
    return _export(result, out, fmt)


# â”€â”€â”€ 14. Split â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def process_split(
    input_path: str,
    segments: int,
    mode: str,
    segment_duration: float,
    fmt: str,
    temp_dir: str,
    original_name: str,
) -> str:
    audio = _load_audio(input_path)
    total_ms = len(audio)
    if total_ms <= 0:
        raise ValueError("Audio file contains no playable audio")
    # safe_base_name strips any directory component. Joining the raw client
    # filename here allowed "../../.." to write outside temp_dir.
    base = safe_base_name(original_name)

    split_points: list[tuple[int, int]] = []
    if mode == "duration":
        dur_ms = int(segment_duration * 1000)
        if dur_ms <= 0:
            raise ValueError("Segment duration must be greater than zero")
        pos = 0
        while pos < total_ms:
            split_points.append((pos, min(pos + dur_ms, total_ms)))
            pos += dur_ms
    else:
        if segments < 2:
            raise ValueError("Segment count must be at least 2")
        seg_len = total_ms // segments
        if seg_len <= 0:
            raise ValueError(
                "This file is too short to split into that many segments"
            )
        for i in range(segments):
            start = i * seg_len
            end = total_ms if i == segments - 1 else (i + 1) * seg_len
            split_points.append((start, end))

    if len(split_points) > settings.max_split_segments:
        raise ValueError(
            f"That would produce {len(split_points)} files; the maximum is "
            f"{settings.max_split_segments}."
        )

    segment_paths = []
    for i, (s, e) in enumerate(split_points):
        seg = audio[s:e]
        seg_path = os.path.join(temp_dir, f"{base}_segment_{i + 1}.{fmt}")
        _export(seg, seg_path, fmt)
        segment_paths.append(seg_path)

    zip_path = os.path.join(temp_dir, f"{base}_segments.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in segment_paths:
            zf.write(p, os.path.basename(p))

    return zip_path


# â”€â”€â”€ 15. Equalizer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

BAND_FREQS = [32, 64, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]


def process_equalize(
    input_path: str,
    bands_json: str,
    fmt: str,
    temp_dir: str,
    original_name: str,
) -> str:
    try:
        raw = json.loads(bands_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"EQ bands must be valid JSON: {exc.msg}") from exc

    if not isinstance(raw, list):
        raise ValueError("EQ bands must be a JSON array")
    if len(raw) != len(BAND_FREQS):
        raise ValueError(
            f"Expected exactly {len(BAND_FREQS)} EQ band gains, got {len(raw)}"
        )

    # Accept both the bare gain array and the self-describing
    # [{"freq": .., "gain": ..}] form so a client cannot silently 500 here.
    gains: list[float] = []
    for entry in raw:
        value = entry.get("gain") if isinstance(entry, dict) else entry
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"EQ gain must be a number, got {type(value).__name__}"
            )
        if not (-24 <= value <= 24):
            raise ValueError(f"EQ gain must be between -24 and 24 dB, got {value}")
        gains.append(float(value))

    eq_filters = []
    for freq, gain in zip(BAND_FREQS, gains):
        if gain != 0:
            eq_filters.append(f"equalizer=f={freq}:t=q:w=1.4:g={gain}")

    filter_str = ",".join(eq_filters) if eq_filters else "anull"
    out = get_output_path(temp_dir, original_name, "eq", fmt)
    _run_ffmpeg(["-i", input_path, "-af", filter_str, "-vn", out])
    return out


# â”€â”€â”€ 16. Bass Boost â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def process_bass_boost(
    input_path: str,
    sub_gain: float,    # dB gain at 80 Hz  (lowshelf)
    bass_gain: float,   # dB gain at 250 Hz (peaking)
    upper_gain: float,  # dB gain at 500 Hz (peaking)
    fmt: str,
    temp_dir: str,
    original_name: str,
) -> str:
    out = get_output_path(temp_dir, original_name, "bass-boosted", fmt)
    filters: list[str] = []
    if sub_gain != 0:
        filters.append(f"equalizer=f=80:width_type=o:width=1:gain={sub_gain:.2f}")
    if bass_gain != 0:
        filters.append(f"equalizer=f=250:width_type=o:width=1:gain={bass_gain:.2f}")
    if upper_gain != 0:
        filters.append(f"equalizer=f=500:width_type=o:width=1:gain={upper_gain:.2f}")
    if filters:
        _run_ffmpeg(["-i", input_path, "-af", ",".join(filters), "-vn", out])
    else:
        audio = _load_audio(input_path)
        _export(audio, out, fmt)
    return out
