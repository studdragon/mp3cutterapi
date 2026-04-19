"""All audio processing functions. Each takes input path + params, returns output path."""

import json
import math
import os
import subprocess
import zipfile

import numpy as np
from pydub import AudioSegment

from config import settings
from utils.file_helpers import get_output_path


def _export(audio: AudioSegment, path: str, fmt: str, bitrate: str | None = None) -> str:
    params: list[str] = []
    if fmt == "m4r":
        # M4R is an AAC/MP4 container with .m4r extension; export as mp4 then rename
        mp4_path = path.replace(".m4r", ".mp4")
        if bitrate:
            params += ["-b:a", bitrate]
        audio.export(mp4_path, format="mp4", parameters=params or None)
        os.rename(mp4_path, path)
        return path
    if bitrate and fmt in ("mp3", "aac", "ogg", "opus"):
        params += ["-b:a", bitrate]
    audio.export(path, format=fmt, parameters=params or None)
    return path


def _run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(
        [settings.ffmpeg_path] + args,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr[:500]}")


# ─── 1. Cut / Trim ──────────────────────────────────────────

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
    audio = AudioSegment.from_file(input_path)
    start_ms = int(start * 1000)
    end_ms = int(end * 1000)
    if remove_selection:
        trimmed = audio[:start_ms] + audio[end_ms:]
    else:
        trimmed = audio[start_ms:end_ms]
    if fade_in > 0:
        trimmed = trimmed.fade_in(int(fade_in * 1000))
    if fade_out > 0:
        trimmed = trimmed.fade_out(int(fade_out * 1000))
    out = get_output_path(temp_dir, original_name, "trimmed", fmt)
    return _export(trimmed, out, fmt)


# ─── 2. Join / Merge ────────────────────────────────────────

def process_join(
    input_paths: list[str],
    fmt: str,
    crossfade_ms: int,
    temp_dir: str,
) -> str:
    segments = [AudioSegment.from_file(p) for p in input_paths]
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


# ─── 3. Convert ─────────────────────────────────────────────

def process_convert(
    input_path: str,
    fmt: str,
    sample_rate: int,
    bitrate: str | None,
    temp_dir: str,
    original_name: str,
) -> str:
    audio = AudioSegment.from_file(input_path)
    if sample_rate:
        audio = audio.set_frame_rate(sample_rate)
    out = get_output_path(temp_dir, original_name, "converted", fmt)
    return _export(audio, out, fmt, bitrate)


# ─── 4. Volume ──────────────────────────────────────────────

def process_volume(
    input_path: str,
    volume_percent: float,
    fmt: str,
    temp_dir: str,
    original_name: str,
) -> str:
    audio = AudioSegment.from_file(input_path)
    if volume_percent <= 0:
        volume_percent = 0.01
    db_change = 20 * math.log10(volume_percent / 100)
    adjusted = audio + db_change
    out = get_output_path(temp_dir, original_name, "volume", fmt)
    return _export(adjusted, out, fmt)


# ─── 5. Speed (without pitch change) ────────────────────────

def _build_atempo_chain(speed: float) -> str:
    """atempo accepts 0.5–100.0; chain for extreme values."""
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


# ─── 6. Pitch (without speed change) ────────────────────────

def process_pitch(
    input_path: str,
    semitones: float,
    fmt: str,
    temp_dir: str,
    original_name: str,
) -> str:
    out = get_output_path(temp_dir, original_name, "pitch", fmt)
    # Get original sample rate
    audio = AudioSegment.from_file(input_path)
    sr = audio.frame_rate
    new_sr = int(sr * (2 ** (semitones / 12)))
    # asetrate changes pitch, aresample restores original duration
    filter_str = f"asetrate={new_sr},aresample={sr}"
    _run_ffmpeg(["-i", input_path, "-af", filter_str, "-vn", out])
    return out


# ─── 7. Fade ────────────────────────────────────────────────

def process_fade(
    input_path: str,
    fade_in_sec: float,
    fade_out_sec: float,
    fmt: str,
    temp_dir: str,
    original_name: str,
) -> str:
    audio = AudioSegment.from_file(input_path)
    if fade_in_sec > 0:
        audio = audio.fade_in(int(fade_in_sec * 1000))
    if fade_out_sec > 0:
        audio = audio.fade_out(int(fade_out_sec * 1000))
    out = get_output_path(temp_dir, original_name, "faded", fmt)
    return _export(audio, out, fmt)


# ─── 8. Reverse ─────────────────────────────────────────────

def process_reverse(
    input_path: str,
    fmt: str,
    temp_dir: str,
    original_name: str,
) -> str:
    audio = AudioSegment.from_file(input_path).reverse()
    out = get_output_path(temp_dir, original_name, "reversed", fmt)
    return _export(audio, out, fmt)


# ─── 9. Ringtone ────────────────────────────────────────────

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
    audio = AudioSegment.from_file(input_path)
    start_ms = int(start * 1000)
    end_ms = int(end * 1000)
    duration_ms = min(end_ms - start_ms, 40000)
    trimmed = audio[start_ms : start_ms + duration_ms]
    if fade_in > 0:
        trimmed = trimmed.fade_in(int(fade_in * 1000))
    if fade_out > 0:
        trimmed = trimmed.fade_out(int(fade_out * 1000))
    out = get_output_path(temp_dir, original_name, "ringtone", fmt)
    return _export(trimmed, out, fmt)


# ─── 10. Noise Reduction ────────────────────────────────────

def process_noise(
    input_path: str,
    reduction: float,
    fmt: str,
    temp_dir: str,
    original_name: str,
) -> str:
    import noisereduce as nr

    audio = AudioSegment.from_file(input_path)
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

    reduced_int = np.clip(reduced_all, -32768, 32767).astype(np.int16)
    result = AudioSegment(
        reduced_int.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=2,
        channels=channels,
    )
    out = get_output_path(temp_dir, original_name, "denoised", fmt)
    return _export(result, out, fmt)


# ─── 11. Extract Audio from Video ───────────────────────────

def process_extract(
    input_path: str,
    fmt: str,
    temp_dir: str,
    original_name: str,
) -> str:
    out = get_output_path(temp_dir, original_name, "audio", fmt)
    _run_ffmpeg(["-i", input_path, "-vn", "-y", out])
    return out


# ─── 12. Dynamic Range Compression ──────────────────────────

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


# ─── 13. Karaoke (Vocal Removal) ────────────────────────────

def process_karaoke(
    input_path: str,
    fmt: str,
    temp_dir: str,
    original_name: str,
) -> str:
    audio = AudioSegment.from_file(input_path)
    if audio.channels < 2:
        raise ValueError("Vocal removal requires stereo audio")

    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples = samples.reshape((-1, 2))
    left = samples[:, 0]
    right = samples[:, 1]
    instrumental = ((left - right) / 2).astype(np.int16)

    result = AudioSegment(
        instrumental.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=2,
        channels=1,
    )
    out = get_output_path(temp_dir, original_name, "instrumental", fmt)
    return _export(result, out, fmt)


# ─── 14. Split ──────────────────────────────────────────────

def process_split(
    input_path: str,
    segments: int,
    mode: str,
    segment_duration: float,
    fmt: str,
    temp_dir: str,
    original_name: str,
) -> str:
    audio = AudioSegment.from_file(input_path)
    total_ms = len(audio)
    base = os.path.splitext(original_name)[0]

    split_points: list[tuple[int, int]] = []
    if mode == "duration" and segment_duration > 0:
        dur_ms = int(segment_duration * 1000)
        pos = 0
        while pos < total_ms:
            split_points.append((pos, min(pos + dur_ms, total_ms)))
            pos += dur_ms
    else:
        seg_len = total_ms // segments
        for i in range(segments):
            start = i * seg_len
            end = total_ms if i == segments - 1 else (i + 1) * seg_len
            split_points.append((start, end))

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


# ─── 15. Equalizer ──────────────────────────────────────────

BAND_FREQS = [32, 64, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]


def process_equalize(
    input_path: str,
    bands_json: str,
    fmt: str,
    temp_dir: str,
    original_name: str,
) -> str:
    gains = json.loads(bands_json)
    if len(gains) != 10:
        raise ValueError("Expected exactly 10 EQ band gains")
    for g in gains:
        if not isinstance(g, (int, float)):
            raise ValueError(f"EQ gain must be a number, got {type(g).__name__}")
        if not (-24 <= g <= 24):
            raise ValueError(f"EQ gain must be between -24 and 24 dB, got {g}")

    eq_filters = []
    for freq, gain in zip(BAND_FREQS, gains):
        if gain != 0:
            eq_filters.append(f"equalizer=f={freq}:t=q:w=1.4:g={gain}")

    filter_str = ",".join(eq_filters) if eq_filters else "anull"
    out = get_output_path(temp_dir, original_name, "eq", fmt)
    _run_ffmpeg(["-i", input_path, "-af", filter_str, "-vn", out])
    return out


# ─── 17. 8D Audio ───────────────────────────────────────

def process_8d(
    input_path: str,
    speed: float,     # LFO panning rate in Hz (0.02 – 0.5)
    depth: float,     # stereo panning width  0.0 – 1.0
    reverb: bool,     # add spatial echo/reverb
    fmt: str,
    temp_dir: str,
    original_name: str,
) -> str:
    out = get_output_path(temp_dir, original_name, "8d", fmt)
    filters: list[str] = [
        "aformat=channel_layouts=stereo",
    ]
    if reverb:
        filters.append("aecho=0.6:0.3:60|80:0.3|0.2")
    filters.append(
        f"apulsator=hz={speed:.4f}:mode=sine:width={depth:.2f}"
        f":level_in=0.9:level_out=0.9"
    )
    _run_ffmpeg(["-i", input_path, "-af", ",".join(filters), "-vn", out])
    return out


# ─── 16. Bass Boost ─────────────────────────────────────────

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
        audio = AudioSegment.from_file(input_path)
        _export(audio, out, fmt)
    return out
