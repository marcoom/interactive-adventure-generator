# Interactive Adventure Generator

**An AI-powered storytelling application that creates dynamic, interactive narratives where your choices directly influence story progression and outcomes.**

Create your own story and make decisions that can lead you to success or failure. Choose wisely!

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-green.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)

## Features

- **Dual AI Support**: Google Gemini 2.0 Flash (with API key) or local model (CPU-based, no API key required)
- **Voice Interaction**: Speak your choices using Whisper speech-to-text
- **Narrated Stories**: Listen to AI-generated narration with Piper TTS
- **Multi-language**: English and Spanish support for both text and voice
- **Customizable**: Adjustable AI creativity and custom story preferences
- **User-Friendly**: Clean Gradio web interface
- **Docker Ready**: Fully containerized with pre-downloaded models

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Build and run the container
docker build -t interactive-adventure .
docker run -p 7860:7860 interactive-adventure

# With Google API key (optional)
docker run -p 7860:7860 -e GOOGLE_API_KEY=your_key_here interactive-adventure
```

### Option 2: Local Installation

#### Prerequisites

**System Dependencies:**
- Python 3.11+
- ffmpeg (required for voice interaction)

```bash
# Install ffmpeg
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg

# Windows:
# Download from https://ffmpeg.org/download.html
```

#### Installation

```bash
# Clone the repository
git clone <repository-url>
cd interactive-adventure-generator

# Copy environment template and configure
cp .env.example .env
# Edit .env file with your API keys and settings (optional)
nano .env  # or use your preferred editor

# Install Python dependencies
pip install -r requirements.txt

# Run the application
python app.py

# With custom port
python app.py --port 8080 --share
```

Visit `http://localhost:7860` to start your adventure!

## How to Play

1. **Read the Story**: The AI narrator sets the scene and presents you with situations
2. **Make Choices**: Type or speak your decisions in the input field
3. **Experience Consequences**: Watch how your choices shape the narrative
4. **Reach Endings**: Stories conclude naturally or when your character meets their fate
5. **Customize Experience**: Adjust language, AI creativity, and story preferences

## Architecture

```
interactive-adventure-generator/
├── app.py                      # Main application entry point
├── config/
│   └── settings.py            # Configuration management
├── core/
│   ├── narrator.py            # AI narrator logic and prompts
│   └── story_manager.py       # Story state and flow management
├── models/
│   ├── base_model.py          # Abstract model interface
│   ├── gemini_model.py        # Google Gemini implementation
│   └── local_model.py         # Local model implementation
├── audio/
│   ├── tts.py                 # Text-to-speech (Piper)
│   └── stt.py                 # Speech-to-text (Whisper)
├── ui/
│   └── interface.py           # Gradio web interface
├── utils/
│   └── helpers.py             # Utility functions
└── data/                      # Pre-downloaded models and voices
```

## Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# Optional - uses local model if not provided
GOOGLE_API_KEY=your_google_api_key

# Application settings
DEFAULT_LANGUAGE=English          # English or Español
DEFAULT_TEMPERATURE=1.0           # AI creativity (0.0-2.0)

# Local model configuration (used when no API key)
LOCAL_MODEL_NAME=HuggingFaceTB/SmolLM2-135M-Instruct

# Server settings
GRADIO_PORT=7860                  # Server port
GRADIO_HOST=0.0.0.0              # Server host
```

### Available Local Models

When running without a Google API key, you can choose from these models:

- `HuggingFaceTB/SmolLM2-135M-Instruct` - Fastest, smallest (default)
- `HuggingFaceTB/SmolLM2-360M-Instruct` - Balanced performance
- `HuggingFaceTB/SmolLM2-1.7B` - Better quality, slower
- `HuggingFaceTB/SmolLM3-3B` - Best quality, slowest

### Model Selection Logic

- **With GOOGLE_API_KEY**: Uses Google Gemini 2.0 Flash (fast, cloud-based)
- **Without API key**: Uses local SmolLM2-135M (slower, but completely offline)

## AI Capabilities Used

| Capability | Implementation |
|------------|----------------|
| **Few-shot Prompting** | System prompts with example scenarios guide narrative generation |
| **Long Context** | Full conversation history maintained for coherent multi-turn stories |
| **Audio Understanding** | Voice input transcribed locally via Whisper |
| **Agent Architecture** | Story flow managed through narrator and player nodes |
| **Context Caching** | Session state preserved for consistent storytelling |
| **Streaming** | Real-time response generation for immediate engagement |

## Docker Details

The Docker image includes:
- Pre-downloaded AI models (SmolLM2, Whisper small)
- Pre-downloaded TTS voices (English & Spanish)
- Optimized multi-stage build
- Non-root user for security
- Health checks

## Development

### Local Development

```bash
# Install development dependencies
pip install -r requirements.txt

# Run with debug mode
python app.py --debug

# Run tests (when available)
pytest
```

### Project Structure Principles

- **PEP 8 Compliant**: 79-character lines, proper naming conventions
- **Clean Architecture**: Single responsibility, dependency injection
- **Modular Design**: Each component has a specific purpose
- **Error Handling**: Graceful fallbacks and user-friendly messages

## Educational Background

This project was developed for the [5-day Gen AI Intensive Course with Google](https://rsvp.withgoogle.com/events/google-generative-ai-intensive_2025q1) capstone project and demonstrates practical application of:

- Large Language Models (LLMs)
- Multimodal AI (text + voice)
- Agent-based architectures
- Real-time streaming
- Cloud and local model deployment

## Contributing

Contributions welcome! Please feel free to submit pull requests or open issues.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.