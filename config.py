from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    ANTHROPIC_API_KEY: str = "your_anthropic_api_key_here"
    GROQ_API_KEY: str = ""
    CNN_WEIGHTS_PATH: str = "weights/resnet50_forgery.pt"
    MAX_FILE_SIZE_MB: int = 20


settings = Settings()