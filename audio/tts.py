"""Text-to-speech using Piper TTS."""

import tempfile
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import soundfile as sf

from piper.voice import PiperVoice
from piper.download import get_voices, ensure_voice_exists, find_voice

from config.settings import settings


class TTSEngine:
    """Text-to-speech engine using Piper TTS."""

    def __init__(self):
        """Initialize TTS engine with Piper voices."""
        self.voices: Dict[str, PiperVoice] = {}
        self.download_dir = Path.home() / ".local/share/piper-tts/piper-voices"
        self._setup_voices()

    def _setup_voices(self) -> None:
        """Set up and download required Piper voices."""
        try:
            # Ensure download directory exists
            self.download_dir.mkdir(parents=True, exist_ok=True)
            
            # Get available voices info
            voices_info = get_voices(self.download_dir)
            
            # Force redownload of corrupted configurations
            for voice_name in settings.piper_voice_names.values():
                config_file = self.download_dir / f"{voice_name}.onnx.json"
                model_file = self.download_dir / f"{voice_name}.onnx"
                if config_file.exists() or model_file.exists():
                    try:
                        # Try to load to check if corrupted
                        model_path, config_path = find_voice(
                            voice_name, [self.download_dir]
                        )
                        PiperVoice.load(model_path, config_path=config_path)
                    except Exception:
                        # Remove corrupted files to force redownload
                        if config_file.exists():
                            config_file.unlink()
                        if model_file.exists():
                            model_file.unlink()

            # Download and load voices
            for lang, voice_name in settings.piper_voice_names.items():
                try:
                    ensure_voice_exists(
                        voice_name, [self.download_dir], 
                        self.download_dir, voices_info
                    )
                    
                    model_path, config_path = find_voice(
                        voice_name, [self.download_dir]
                    )
                    
                    self.voices[lang] = PiperVoice.load(
                        model_path, config_path=config_path
                    )
                    
                except Exception as e:
                    print(f"Warning: Failed to load voice for {lang}: {e}")
                    
        except Exception as e:
            print(f"Warning: TTS setup failed: {e}")

    def is_available(self, language: str) -> bool:
        """Check if TTS is available for the given language.
        
        Args:
            language: Target language
            
        Returns:
            True if TTS is available for the language
        """
        return language in self.voices

    def synthesize(self, text: str, language: str) -> Optional[str]:
        """Convert text to speech and return audio file path.
        
        Args:
            text: Text to synthesize
            language: Target language
            
        Returns:
            Path to temporary audio file, or None if synthesis fails
        """
        if not self.is_available(language):
            print(f"Warning: TTS not available for language: {language}")
            return None
            
        if not text.strip():
            return None
            
        try:
            voice = self.voices[language]
            
            # Generate audio chunks
            raw_chunks = voice.synthesize_stream_raw(text)
            audio_bytes = b"".join(raw_chunks)
            
            # Convert to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Create temporary file
            temp_audio = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            )
            
            # Write audio to file
            sf.write(
                temp_audio.name, 
                audio_array, 
                voice.config.sample_rate,
                subtype="PCM_16"
            )
            
            return temp_audio.name
            
        except Exception as e:
            print(f"Warning: TTS synthesis failed: {e}")
            return None

    def get_available_languages(self) -> list:
        """Get list of available TTS languages.
        
        Returns:
            List of available language names
        """
        return list(self.voices.keys())

    def cleanup_temp_files(self) -> None:
        """Clean up temporary audio files."""
        try:
            temp_dir = Path(tempfile.gettempdir())
            
            # Find and remove temporary audio files
            for temp_file in temp_dir.glob("temp_*.wav"):
                try:
                    temp_file.unlink()
                except Exception:
                    pass  # Ignore errors during cleanup
                    
        except Exception:
            pass  # Ignore errors during cleanup