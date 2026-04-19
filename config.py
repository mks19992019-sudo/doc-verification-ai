from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    ANTHROPIC_API_KEY: str = "your_anthropic_api_key_here"
    GROQ_API_KEY: str = "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"
    CNN_WEIGHTS_PATH: str = "weights/resnet50_forgery.pt"
    MAX_FILE_SIZE_MB: int = 20
    # Comma-separated list of allowed frontend origins for CORS
    # Example: http://localhost:3000,https://your-frontend.vercel.app
    ALLOWED_ORIGINS: str = "*"


settings = Settings()