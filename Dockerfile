# Interactive Adventure Generator Dockerfile
# Multi-stage build for optimized container size

# Build stage for downloading models
FROM python:3.11-slim as model-downloader

# Install system dependencies for downloading models
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages needed for model downloads
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    transformers \
    openai-whisper \
    piper-tts

# Create directories for models
RUN mkdir -p /tmp/models /tmp/voices

# Pre-download local model
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    model_name='HuggingFaceTB/SmolLM2-135M-Instruct'; \
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='/tmp/models'); \
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='/tmp/models')"

# Pre-download Whisper model
RUN python -c "import whisper; whisper.load_model('small', download_root='/tmp/models')"

# Pre-download Piper TTS voices
RUN python -c "
from piper.voice import PiperVoice
from piper.download import get_voices, ensure_voice_exists, find_voice
from pathlib import Path

download_dir = Path('/tmp/voices')
download_dir.mkdir(exist_ok=True)

voices_info = get_voices(download_dir)
voice_names = ['es_ES-carlfm-x_low', 'en_US-joe-medium']

for voice_name in voice_names:
    try:
        ensure_voice_exists(voice_name, [download_dir], download_dir, voices_info)
        print(f'Downloaded voice: {voice_name}')
    except Exception as e:
        print(f'Warning: Failed to download {voice_name}: {e}')
"

# Production stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Audio processing
    libsndfile1 \
    ffmpeg \
    # System utilities
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pre-downloaded models from build stage
COPY --from=model-downloader /tmp/models /home/appuser/.cache
COPY --from=model-downloader /tmp/voices /home/appuser/.local/share/piper-tts/piper-voices

# Copy application code
COPY --chown=appuser:appuser . .

# Create data directories
RUN mkdir -p data/voices data/models && \
    chown -R appuser:appuser data/

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run application
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]