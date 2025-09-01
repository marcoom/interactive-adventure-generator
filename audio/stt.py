"""Speech-to-text using OpenAI Whisper."""

from typing import Optional
import whisper

from config.settings import settings


class STTEngine:
    """Speech-to-text engine using OpenAI Whisper."""

    def __init__(self, model_size: str = "small"):
        """Initialize STT engine with Whisper model.
        
        Args:
            model_size: Whisper model size ("tiny", "small", "medium", etc.)
        """
        self.model_size = model_size
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the Whisper model."""
        try:
            self.model = whisper.load_model(self.model_size)
        except Exception as e:
            print(f"Warning: Failed to load Whisper model: {e}")
            self.model = None

    def is_available(self) -> bool:
        """Check if STT engine is available.
        
        Returns:
            True if STT engine is ready
        """
        return self.model is not None

    def transcribe(self, audio_path: str, language: str) -> Optional[str]:
        """Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            language: Target language for transcription
            
        Returns:
            Transcribed text, or None if transcription fails
        """
        if not self.is_available():
            print("Warning: STT engine not available")
            return None
            
        if not audio_path:
            return None
            
        try:
            # Get language code for Whisper
            language_code = settings.language_codes.get(language, "en")
            
            # Transcribe audio
            result = self.model.transcribe(
                audio_path, 
                language=language_code
            )
            
            # Extract and clean text
            text = result.get("text", "").strip()
            return text if text else None
            
        except Exception as e:
            print(f"Warning: STT transcription failed: {e}")
            return None

    def get_supported_languages(self) -> list:
        """Get list of supported language codes.
        
        Returns:
            List of supported language codes
        """
        return list(settings.language_codes.values())

    def reload_model(self, model_size: str = None) -> None:
        """Reload the Whisper model with optional size change.
        
        Args:
            model_size: New model size, or None to keep current size
        """
        if model_size:
            self.model_size = model_size
            
        self._load_model()