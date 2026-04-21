"""All audio processing functions. Each takes input path + params, returns output path."""

import json
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


def _ffprobe_path() -> str:
    base = os.path.basename(settings.ffmpeg_path)
    probe = base.replace("ffmpeg", "ffprobe")
    directory = os.path.dirname(settings.ffmpeg_path)
    return os.path.join(directory, probe) if directory else probe


def _get_audio_info(path: str) -> dict:
    """Return {'duration': float (s), 'sample_rate': int (Hz)} via ffprobe."""
    result = subprocess.run(
        [_ffprobe_path(), "-v", "quiet", "-print_format", "json",
         "-show_streams", "-select_streams", "a:0", path],
        capture_output=True, text=True, timeout=30,
    )
    streams = json.loads(result.stdout).get("streams", [{}])
    s = streams[0] if streams else {}
    duration = float(s.get("duration") or 0)
    if not duration:
        dur_str = (s.get("tags") or {}).get("DURATION", "")
        if ":" in dur_str:
            h, m, sec = dur_str.split(":")
            duration = int(h) * 3600 + int(m) * 60 + float(sec)
    sample_rate = int(s.get("sample_rate") or 44100)
    return {"duration": duration, "sample_rate": sample_rate}


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
    out = get_output_path(temp_dir, original_name, "trimmed", fmt)

    if remove_selection:
        # Must concat two segments — pydub is simplest here
        audio = AudioSegment.from_file(input_path)
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        trimmed = audio[:start_ms] + audio[end_ms:]
        if fade_in > 0:
            trimmed = trimmed.fade_in(int(fade_in * 1000))
        if fade_out > 0:
            trimmed = trimmed.fade_out(int(fade_out * 1000))
        return _export(trimmed, out, fmt)

    # Fast path: direct ffmpeg seek — no full-file decode
    duration = end - start
    filters: list[str] = []
    if fade_in > 0:
        filters.append(f"afade=t=in:st=0:d={fade_in:.3f}")
    if fade_out > 0:
        fade_start = max(0.0, duration - fade_out)
        filters.append(f"afade=t=out:st={fade_start:.3f}:d={fade_out:.3f}")

    codec_args: list[str] = []
    if fmt == "mp3":
        codec_args = ["-c:a", "libmp3lame", "-b:a", "192k"]
    elif fmt == "wav":
        codec_args = ["-c:a", "pcm_s16le"]
    elif fmt == "ogg":
        codec_args = ["-c:a", "libvorbis"]
    elif fmt == "flac":
        codec_args = ["-c:a", "flac"]
    elif fmt in ("aac", "m4a", "m4r", "m4b"):
        codec_args = ["-c:a", "aac", "-b:a", "192k"]
    elif fmt == "opus":
        codec_args = ["-c:a", "libopus"]
    else:
        codec_args = ["-c:a", "copy"]

    args = [
        "-ss", str(start),
        "-i", input_path,
        "-t", str(duration),
        "-vn",
        *codec_args,
    ]
    if filters:
        args += ["-af", ",".join(filters)]
    args += ["-y", out]
    _run_ffmpeg(args)
    return out


# ─── 2. Join / Merge ────────────────────────────────────────

def process_join(
    input_paths: list[str],
    fmt: str,
    crossfade_ms: int,
    temp_dir: str,
) -> str:
    if not input_paths:
        raise ValueError("No audio files provided")
    out = os.path.join(temp_dir, f"merged.{fmt}")
    n = len(input_paths)
    if crossfade_ms > 0:
        crossfade_sec = crossfade_ms / 1000.0
        fc_parts = []
        for i in range(n - 1):
            a = f"[cf{i}]" if i > 0 else "[0:a]"
            b = f"[{i + 1}:a]"
            label = f"[cf{i + 1}]" if i < n - 2 else "[aout]"
            fc_parts.append(f"{a}{b}acrossfade=d={crossfade_sec:.3f}:c1=tri:c2=tri{label}")
        filter_complex = ";".join(fc_parts)
    else:
        labels = "".join(f"[{i}:a]" for i in range(n))
        filter_complex = f"{labels}concat=n={n}:v=0:a=1[aout]"
    args: list[str] = []
    for p in input_paths:
        args += ["-i", p]
    args += ["-filter_complex", filter_complex, "-map", "[aout]", "-vn", "-y", out]
    _run_ffmpeg(args)
    return out


# ─── 3. Convert ─────────────────────────────────────────────

def process_convert(
    input_path: str,
    fmt: str,
    sample_rate: int,
    bitrate: str | None,
    temp_dir: str,
    original_name: str,
) -> str:
    out = get_output_path(temp_dir, original_name, "converted", fmt)
    args = ["-i", input_path, "-vn"]
    if sample_rate:
        args += ["-ar", str(sample_rate)]
    if bitrate and fmt in ("mp3", "aac", "ogg", "opus"):
        args += ["-b:a", bitrate]
    if fmt == "m4r":
        mp4_out = out.replace(".m4r", ".mp4")
        args += ["-y", mp4_out]
        _run_ffmpeg(args)
        os.rename(mp4_out, out)
        return out
    args += ["-y", out]
    _run_ffmpeg(args)
    return out


# ─── 4. Volume ──────────────────────────────────────────────

def process_volume(
    input_path: str,
    volume_percent: float,
    fmt: str,
    temp_dir: str,
    original_name: str,
) -> str:
    if volume_percent <= 0:
        volume_percent = 0.01
    ratio = volume_percent / 100.0
    out = get_output_path(temp_dir, original_name, "volume", fmt)
    _run_ffmpeg(["-i", input_path, "-af", f"volume={ratio:.6f}", "-vn", "-y", out])
    return out


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
    sr = _get_audio_info(input_path)["sample_rate"]
    new_sr = int(sr * (2 ** (semitones / 12)))
    filter_str = f"asetrate={new_sr},aresample={sr}"
    _run_ffmpeg(["-i", input_path, "-af", filter_str, "-vn", "-y", out])
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
    out = get_output_path(temp_dir, original_name, "faded", fmt)
    filters: list[str] = []
    if fade_in_sec > 0:
        filters.append(f"afade=t=in:st=0:d={fade_in_sec:.3f}")
    if fade_out_sec > 0:
        duration = _get_audio_info(input_path)["duration"]
        fade_start = max(0.0, duration - fade_out_sec)
        filters.append(f"afade=t=out:st={fade_start:.3f}:d={fade_out_sec:.3f}")
    filter_str = ",".join(filters) if filters else "anull"
    _run_ffmpeg(["-i", input_path, "-af", filter_str, "-vn", "-y", out])
    return out


# ─── 8. Reverse ─────────────────────────────────────────────

def process_reverse(
    input_path: str,
    fmt: str,
    temp_dir: str,
    original_name: str,
) -> str:
    out = get_output_path(temp_dir, original_name, "reversed", fmt)
    _run_ffmpeg(["-i", input_path, "-af", "areverse", "-vn", "-y", out])
    return out


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
    out = get_output_path(temp_dir, original_name, "ringtone", fmt)
    duration = min(end - start, 40.0)
    filters: list[str] = []
    if fade_in > 0:
        filters.append(f"afade=t=in:st=0:d={fade_in:.3f}")
    if fade_out > 0:
        fade_start = max(0.0, duration - fade_out)
        filters.append(f"afade=t=out:st={fade_start:.3f}:d={fade_out:.3f}")
    actual_out = out.replace(".m4r", ".mp4") if fmt == "m4r" else out
    codec_args: list[str] = ["-c:a", "aac", "-b:a", "192k"] if fmt == "m4r" else []
    args = ["-ss", str(start), "-i", input_path, "-t", str(duration), "-vn", *codec_args]
    if filters:
        args += ["-af", ",".join(filters)]
    args += ["-y", actual_out]
    _run_ffmpeg(args)
    if actual_out != out:
        os.rename(actual_out, out)
    return out


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
    total_sec = _get_audio_info(input_path)["duration"]
    base = os.path.splitext(original_name)[0]

    split_points: list[tuple[float, float]] = []
    if mode == "duration" and segment_duration > 0:
        pos = 0.0
        while pos < total_sec:
            split_points.append((pos, min(pos + segment_duration, total_sec)))
            pos += segment_duration
    else:
        seg_len = total_sec / segments
        for i in range(segments):
            s = i * seg_len
            e = total_sec if i == segments - 1 else (i + 1) * seg_len
            split_points.append((s, e))

    segment_paths = []
    for i, (s, e) in enumerate(split_points):
        seg_path = os.path.join(temp_dir, f"{base}_segment_{i + 1}.{fmt}")
        _run_ffmpeg(["-ss", str(s), "-i", input_path, "-t", str(e - s), "-vn", "-y", seg_path])
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
        _run_ffmpeg(["-i", input_path, "-af", ",".join(filters), "-vn", "-y", out])
    else:
        _run_ffmpeg(["-i", input_path, "-c:a", "copy", "-vn", "-y", out])
    return out
