# Interactive Adventure Generator Dockerfile
# Multi-stage build for optimized container size

# Build stage for downloading models
FROM python:3.11-slim AS model-downloader

# Install system dependencies for downloading models
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages needed for model downloads
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers openai-whisper piper-tts

# Create directories for models
RUN mkdir -p /tmp/models /tmp/voices


# Pre-download Whisper model
RUN python -c "import whisper; whisper.load_model('small', download_root='/tmp/models')"

# Copy local TTS voices instead of downloading
COPY data/voices/* /tmp/voices/

# Production stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Install system dependencies including build tools for PyTorch compilation
RUN apt-get update && apt-get install -y \
    # Audio processing
    libsndfile1 \
    ffmpeg \
    # System utilities
    curl \
    # Build tools for PyTorch JIT compilation (fixes quantization errors)
    build-essential \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pre-downloaded Whisper models from build stage with correct ownership
COPY --from=model-downloader --chown=appuser:appuser /tmp/models /home/appuser/.cache
COPY --from=model-downloader --chown=appuser:appuser /tmp/voices /home/appuser/.local/share/piper-tts/piper-voices

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