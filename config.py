import os
import tempfile
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Browser origins allowed to call this API, comma separated. Becomes
    # CORSMiddleware's allow_origins in main.py.
    #
    # An origin missing from this list does not produce a helpful error: Starlette
    # simply omits the Access-Control-Allow-Origin header, the request succeeds
    # and logs a 200, and the browser discards the response. It reads as a network
    # fault on the front end while the server looks perfectly healthy.
    #
    # Matched exactly on scheme, host and port, so http and https differ, www and
    # non-www differ, and a trailing slash breaks the match. main.py strips
    # trailing slashes defensively.
    #
    # 4321 is the Astro dev server, 3000 the older Next.js one. Override with
    # ALLOWED_ORIGINS to add a staging or preview domain without editing code.
    allowed_origins: str = (
        "https://mp3cutteronline.com,"
        "https://www.mp3cutteronline.com,"
        "http://localhost:4321,"
        "http://localhost:3000"
    )
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
