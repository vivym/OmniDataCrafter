from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    log_level: str = "INFO"

    mongo_uri: str = "mongodb://mongo:27017"
    mongo_db: str = "OmniDataCrafter"

    worker_token: str = "worker-token"


settings = Settings(
    _env_file=".env",
    _env_file_encoding="utf-8",
)
