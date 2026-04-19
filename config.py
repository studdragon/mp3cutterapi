import os
import tempfile
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    allowed_origins: str = "http://localhost:3000,https://mp3cutteronline.com,https://www.mp3cutteronline.com"
    temp_dir: str = os.path.join(tempfile.gettempdir(), "mp3cutter")
    max_file_size_mb: int = 500
    ffmpeg_path: str = "ffmpeg"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
