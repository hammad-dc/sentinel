from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Sentinel"
    # Adjust credentials to your local PostgreSQL credentials
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/sentinel_db"

    # Resolve project root dynamically (two levels up from backend/app/core)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    STORAGE_DIR: Path = BASE_DIR / "storage"
    ORIGINALS_DIR: Path = STORAGE_DIR / "originals"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

# Ensure target directories exist when the app boots
settings.ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)