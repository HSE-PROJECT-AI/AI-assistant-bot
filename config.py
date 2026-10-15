from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    telegram_bot_token: SecretStr
    groq_api_key: SecretStr
    database_url: SecretStr
    groq_model: str = "openai/gpt-oss-20b"
    timezone: str = "Europe/Moscow"
    reminder_check_minutes: int = Field(default=1, ge=1, le=15)

    @field_validator("telegram_bot_token", "groq_api_key", "database_url")
    @classmethod
    def require_secret(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("Environment variable must not be empty")
        return value

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: SecretStr) -> SecretStr:
        if make_url(value.get_secret_value()).drivername != "postgresql+asyncpg":
            raise ValueError("DATABASE_URL must use postgresql+asyncpg")
        return value

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as error:
            raise ValueError("Unknown IANA timezone") from error
        return value
