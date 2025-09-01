"""Configuration settings for Interactive Adventure Generator."""

import os
from typing import Optional


class Settings:
    """Application configuration settings."""

    def __init__(self):
        """Initialize settings with environment variables and defaults."""
        self._load_settings()
        
    def _load_settings(self):
        """Load settings from environment variables."""
        self.google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
        self.default_language: str = os.getenv("DEFAULT_LANGUAGE", "English")
        self.default_temperature: float = float(
            os.getenv("DEFAULT_TEMPERATURE", "1.0")
        )
        self.gradio_port: int = int(os.getenv("GRADIO_PORT", "7860"))
        self.gradio_host: str = os.getenv("GRADIO_HOST", "0.0.0.0")
        
        # Local model configuration
        self.local_model_name: str = os.getenv(
            "LOCAL_MODEL_NAME", 
            "HuggingFaceTB/SmolLM2-135M-Instruct"
        )
        
        # Model paths
        self.data_dir: str = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data"
        )
        self.voices_dir: str = os.path.join(self.data_dir, "voices")
        self.models_dir: str = os.path.join(self.data_dir, "models")
        
        # Supported languages
        self.supported_languages = ["English", "Español"]
        
        # Piper voice names
        self.piper_voice_names = {
            "Español": "es_ES-carlfm-x_low",
            "English": "en_US-joe-medium"
        }
        
        # Whisper language codes
        self.language_codes = {
            "Español": "es",
            "English": "en"
        }

    def reload(self):
        """Reload settings from environment variables."""
        self._load_settings()

    @property
    def use_local_model(self) -> bool:
        """Determine if local model should be used."""
        return not self.google_api_key or self.google_api_key.strip() == ""

    def validate(self) -> None:
        """Validate configuration settings."""
        if self.default_language not in self.supported_languages:
            raise ValueError(
                f"Unsupported language: {self.default_language}. "
                f"Supported: {self.supported_languages}"
            )
        
        if not (0 <= self.default_temperature <= 2):
            raise ValueError(
                f"Temperature must be between 0 and 2, got: "
                f"{self.default_temperature}"
            )


# Global settings instance
settings = Settings()