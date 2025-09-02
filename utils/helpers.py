"""Utility functions and helpers."""

import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Union

from config.settings import settings
from models.base_model import BaseModel
from models.gemini_model import GeminiModel
from models.local_model import LocalModel


def create_model_instance(temperature: float = None) -> BaseModel:
    """Create appropriate model instance based on configuration.
    
    Args:
        temperature: Model temperature, uses default if None
        
    Returns:
        Initialized model instance (Gemini or Local)
    """
    if temperature is None:
        temperature = settings.default_temperature
        
    if settings.use_local_model:
        print("Using local model (no Google API key found)")
        return LocalModel(temperature=temperature)
    else:
        print("Using Google Gemini model")
        return GeminiModel(
            api_key=settings.google_api_key,
            temperature=temperature
        )


def process_multimodal_input(user_input: Union[str, Dict[str, Any]]) -> tuple:
    """Process multimodal input from Gradio.
    
    Args:
        user_input: Either string or dict with text and files
        
    Returns:
        Tuple of (text_input, audio_path)
    """
    input_text = ""
    recorded_audio = None
    
    if isinstance(user_input, dict):
        input_text = user_input.get("text", "").strip()
        files = user_input.get("files", [])
        if files:
            file_item = files[0]
            if isinstance(file_item, dict):
                recorded_audio = file_item.get("path", None)
            else:
                recorded_audio = file_item
    else:
        input_text = str(user_input).strip()
    
    return input_text, recorded_audio


def cleanup_temp_files() -> None:
    """Clean up temporary audio files."""
    try:
        temp_dir = Path(tempfile.gettempdir())
        
        # Clean up temporary audio files
        patterns = ["temp_*.wav", "temp_*.mp3", "gradio_*.wav"]
        for pattern in patterns:
            for temp_file in temp_dir.glob(pattern):
                try:
                    temp_file.unlink()
                except Exception:
                    pass  # Ignore individual file errors
                    
    except Exception:
        pass  # Ignore cleanup errors


def validate_language(language: str) -> str:
    """Validate and normalize language setting.
    
    Args:
        language: Language string to validate
        
    Returns:
        Validated language string
        
    Raises:
        ValueError: If language is not supported
    """
    if language not in settings.supported_languages:
        raise ValueError(
            f"Unsupported language: {language}. "
            f"Supported: {settings.supported_languages}"
        )
    return language


def validate_temperature(temperature: float) -> float:
    """Validate temperature setting.
    
    Args:
        temperature: Temperature value to validate
        
    Returns:
        Validated temperature value
        
    Raises:
        ValueError: If temperature is out of range
    """
    if not (0 <= temperature <= 2):
        raise ValueError(
            f"Temperature must be between 0 and 2, got: {temperature}"
        )
    return temperature


def setup_environment() -> None:
    """Set up application environment and directories."""
    try:
        # Create data directories if they don't exist
        os.makedirs(settings.voices_dir, exist_ok=True)
        os.makedirs(settings.models_dir, exist_ok=True)
        
        # Clean up any existing temp files on startup
        cleanup_temp_files()
        
    except Exception as e:
        print(f"Warning: Environment setup failed: {e}")


def get_model_info() -> Dict[str, str]:
    """Get information about the current model configuration.
    
    Returns:
        Dictionary with model information
    """
    if settings.use_local_model:
        # Extract model name from full path for display
        model_display_name = settings.local_model_name.split('/')[-1]
        return {
            "type": "local",
            "name": model_display_name,
            "provider": "Hugging Face",
            "requires_api_key": False
        }
    else:
        return {
            "type": "remote",
            "name": "Gemini 2.0 Flash",
            "provider": "Google",
            "requires_api_key": True
        }


def format_error_message(error: Exception, context: str = "") -> str:
    """Format error message for user display.
    
    Args:
        error: Exception that occurred
        context: Additional context for the error
        
    Returns:
        Formatted error message
    """
    base_message = f"An error occurred"
    if context:
        base_message += f" while {context}"
    
    error_details = str(error)
    if error_details:
        return f"{base_message}: {error_details}"
    else:
        return f"{base_message}. Please try again."