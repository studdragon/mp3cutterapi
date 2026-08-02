import os
import tempfile
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    allowed_origins: str = "http://localhost:3000"
    temp_dir: str = os.path.join(tempfile.gettempdir(), "mp3cutter")
    max_file_size_mb: int = 500
    ffmpeg_path: str = "ffmpeg"

    # Abuse limits. Every endpoint spawns ffmpeg and holds a decoded copy of the
    # audio in memory, so an unauthenticated caller can trivially exhaust the
    # host without these.
    rate_limit_requests: int = 20
    rate_limit_window_seconds: int = 60

    # Upper bound on the number of files /split may emit. Without this a request
    # for a million segments turns a small upload into a zip bomb.
    max_split_segments: int = 500
    min_segment_duration_seconds: float = 0.5

    # Longest ringtone /ringtone will emit, in seconds.
    max_ringtone_seconds: float = 40.0

    class Config:
        env_file = ".env"
        extra = "ignore"

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


settings = Settings()
