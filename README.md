# Interactive Adventure Generator

An AI-powered storytelling application that creates dynamic, interactive narratives where your choices directly influence story progression and outcomes. Create your own adventure and make decisions that can lead you to success or failure.

Supports both cloud-based AI (Google Gemini) and fully offline local models, with voice interaction and multilingual narration.

## Features

- **Dual AI Support**: Google Gemini 2.0 Flash or local CPU-based models
- **Voice Interaction**: Speak your choices with Whisper speech-to-text
- **Audio Narration**: Listen to AI-generated stories with Piper TTS
- **Multilingual**: English and Spanish text and voice support
- **Customizable**: Adjust AI creativity and story preferences
- **Web Interface**: Clean Gradio-based UI
- **Docker Ready**: Containerized with pre-downloaded models

## Quick Start

### Docker (Recommended)

```bash
# Build and run
docker build -t interactive-adventure .
docker run -p 7860:7860 interactive-adventure

# With Google API key (optional, for better performance)
docker run -p 7860:7860 -e GOOGLE_API_KEY=your_key_here interactive-adventure
```

### Local Installation

**Requirements:** Python 3.11+, ffmpeg

```bash
# Install system dependency
sudo apt-get install ffmpeg  # Ubuntu/Debian
brew install ffmpeg         # macOS

# Clone and setup
git clone <repository-url>
cd interactive-adventure-generator
cp .env.example .env  # Optional: add GOOGLE_API_KEY
pip install -r requirements.txt

# Run
python app.py
```

Visit `http://localhost:7860` to start your adventure. Works offline with local models or with Google API key for enhanced performance.

## Usage

### How It Works

1. **Story Introduction**: AI generates an initial scenario and presents you with a situation
2. **Make Your Choice**: Type your action or use the microphone to speak your decision
3. **Story Continues**: AI responds with consequences based on your choices
4. **Branching Paths**: Your decisions directly influence story direction and outcomes
5. **Natural Endings**: Stories conclude when the narrative reaches a natural end

### Input Methods
- **Text Input**: Type actions and decisions in the input field
- **Voice Input**: Click microphone icon and speak (transcribed via Whisper)
- **Mixed Mode**: Combine text and voice input as preferred

### Story Controls
**End Story Commands** (case insensitive): `quit`, `exit`, `q`, `goodbye`

The AI responds with "The End." and disables input until you start a new story.

### Customization
Access "Additional Settings" panel for:
- **Language**: English or Spanish (text and voice)
- **Temperature**: AI creativity level (0.0 conservative, 2.0 very creative) 
- **Autoplay Narration**: Automatic audio narration of responses
- **User Preferences**: Custom guidelines for story themes and settings

## Project Structure

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
├── notebooks/                 # Jupyter demos and prototypes
└── data/                      # Pre-downloaded models and voices
```

The application follows clean architecture principles with modular components. The `models/` directory implements an abstract base class pattern for swapping between cloud and local AI providers. The `notebooks/` directory contains the original Jupyter prototype and individual component demonstrations.

## Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# Google API Key (optional - provides superior performance)
GOOGLE_API_KEY=your_google_api_key

# Application settings
DEFAULT_LANGUAGE=English          # English or Español
DEFAULT_TEMPERATURE=1.0           # AI creativity (0.0-2.0)

# Local model configuration
LOCAL_MODEL_NAME=HuggingFaceTB/SmolLM2-135M-Instruct

# Server settings
GRADIO_PORT=7860
GRADIO_HOST=0.0.0.0
```

### API Key Setup

For enhanced performance, get a free Google API key:
1. Visit [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key)
2. Create your API key following the instructions
3. Add to `.env`: `GOOGLE_API_KEY=your_api_key_here`

Without an API key, the system automatically uses local models (slower but fully offline).

### Model Selection

**Automatic Selection:**
- With `GOOGLE_API_KEY`: Uses Google Gemini 2.0 Flash (recommended)
- Without API key: Automatically uses local SmolLM2-135M (offline)

**Available Local Models:**
- `HuggingFaceTB/SmolLM2-135M-Instruct` - Fastest, smallest (default)
- `HuggingFaceTB/SmolLM2-360M-Instruct` - Balanced performance
- `HuggingFaceTB/SmolLM2-1.7B` - Better quality, slower
- `HuggingFaceTB/SmolLM3-3B` - Best quality, slowest

**Performance Comparison:**

| Feature | Google Gemini | Local Models |
|---------|---------------|-------------|
| Setup | Requires API key | No setup |
| Speed | Very fast | Slower (CPU) |
| Quality | Superior | Good |
| Privacy | Cloud | Fully offline |
| Cost | Free tier | Completely free |
| Resources | Minimal | High CPU usage |

## Development

### Architecture Principles
- **Clean Architecture**: Single responsibility, dependency injection
- **Modular Design**: Each component has a specific purpose  
- **PEP 8 Compliant**: 79-character lines, proper naming conventions
- **Error Handling**: Graceful fallbacks and user-friendly messages

### Key Implementation Details
- **Model Abstraction**: BaseModel interface allows swapping AI providers
- **Multimodal Pipeline**: Integrated text and voice processing
- **State Management**: Conversation history preserved in Gradio state
- **Resource Optimization**: 8-bit quantization for local models

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.