# Product Requirement Document
## Interactive Adventure Generator

### 1. Executive Summary
The Interactive Adventure Generator is an AI-powered storytelling application that creates dynamic, interactive narratives where user choices directly influence story progression and outcomes. The project will be migrated from a Jupyter notebook implementation to a modular Python application with support for both cloud-based (Google Gemini) and local LLM models.

### 2. Project Objectives
- Port existing functional notebook code to a structured Python application
- Maintain all current functionality without modifications
- Add fallback support for local LLM when Google API key is unavailable
- Create a containerized application with Docker support

### 3. Technical Architecture

#### 3.1 Project Structure
```
interactive-adventure-generator/
├── app.py                      # Main application entry point
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
├── .dockerignore              # Docker ignore patterns
├── .gitignore                 # Git ignore patterns
├── .env.example               # Environment variables template
├── LICENSE                    # MIT License
├── README.md                  # Project documentation
├── data/
│   ├── voices/                # Pre-downloaded Piper TTS voices
│   └── models/                # Pre-downloaded Whisper model
├── config/
│   ├── __init__.py
│   └── settings.py            # Configuration management
├── core/
│   ├── __init__.py
│   ├── narrator.py            # LLM narrator logic and prompts
│   └── story_manager.py       # Story state and flow management
├── models/
│   ├── __init__.py
│   ├── base_model.py          # Abstract base class for LLM providers
│   ├── gemini_model.py        # Google Gemini implementation
│   └── local_model.py         # Local SmolLM2 implementation
├── audio/
│   ├── __init__.py
│   ├── tts.py                 # Text-to-speech (Piper)
│   └── stt.py                 # Speech-to-text (Whisper)
├── ui/
│   ├── __init__.py
│   └── interface.py           # Gradio UI components
├── utils/
│   ├── __init__.py
│   └── helpers.py             # Utility functions and temp file cleanup
└── Notebooks/
    ├── individual-components-demos/
    │   ├── LangChain Chatbot demo.ipynb
    │   ├── Piper TTS demo.ipynb
    │   └── Whisper STT demo.ipynb
    └── interactive-adventure-generator-submission.ipynb
```

### 4. Technical Requirements

#### 4.1 Core Functionality
- **Interactive Storytelling**: Dynamic narrative generation with decision points
- **Multimodal Input**: Support for text and voice input
- **Text-to-Speech**: Narrator voice synthesis using Piper TTS
- **Speech-to-Text**: Voice command recognition using Whisper
- **Language Support**: English and Spanish languages
- **Temperature Control**: Adjustable LLM creativity parameter
- **Custom Instructions**: User-defined story preferences

#### 4.2 LLM Model Support
- **Primary**: Google Gemini 2.0 Flash (requires API key)
- **Fallback**: SmolLM2-135M local model (CPU-based, no API key required)
- **Model Selection Logic**: Presence of GOOGLE_API_KEY environment variable determines model:
  - If GOOGLE_API_KEY is set: Use Google Gemini
  - If GOOGLE_API_KEY is not set or empty: Use local SmolLM2 model

#### 4.3 Dependencies
Core dependencies from the notebook:
- `langgraph==0.3.21`
- `langchain-google-genai==2.1.2`
- `openai-whisper`
- `piper-tts`
- `gradio`
- `soundfile`
- `numpy`

Additional dependencies for local model support:
- `transformers`
- `torch` (CPU version)
- `python-dotenv` (for environment variable management)

### 5. Implementation Specifications

#### 5.1 Code Standards
**PEP 8 Compliance:**
- Maximum line length: 79 characters
- Indentation: 4 spaces
- Class names: CapWords convention
- Function/variable names: lowercase_with_underscores
- Constants: UPPER_CASE_WITH_UNDERSCORES
- Two blank lines between top-level definitions
- One blank line between method definitions

**PEP 257 Docstring Conventions:**
- All modules, classes, and public functions must have docstrings
- Use triple double quotes for docstrings
- One-line docstrings for simple functions
- Multi-line docstrings with summary line, blank line, and detailed description
- Document parameters, return values, and exceptions

**Clean Code Principles:**
- Single Responsibility Principle for all classes
- Dependency injection for configurability
- Abstract base classes for extensibility
- Minimal comments (self-documenting code)
- No hardcoded values (use configuration)
- Error handling with specific exceptions

#### 5.2 Class Architecture

**BaseModel (Abstract)**
```python
class BaseModel(ABC):
    """Abstract base class for language model providers."""
    
    @abstractmethod
    def generate(self, messages: List[Message]) -> str:
        """Generate narrative response from message history."""
        pass
    
    @abstractmethod
    def stream(self, messages: List[Message]) -> Iterator[str]:
        """Stream narrative response token by token."""
        pass
```

**StoryManager**
- Manages conversation history using Gradio's gr.State (minimal changes from notebook)
- Handles story state transitions
- Detects ending conditions
- Maintains context between turns in memory

**NarratorAgent**
- Contains system prompts and examples
- Formats messages for LLM
- Processes user preferences
- Manages language-specific prompts

**AudioProcessor**
- TTSEngine: Manages Piper voices from local data/voices directory
- STTEngine: Handles Whisper transcription using pre-downloaded small model
- Language-aware processing
- Implements temporary file cleanup for audio files

**GradioInterface**
- Builds UI components (maintaining notebook structure)
- Manages event handlers
- Handles multimodal input processing
- Controls audio playback

### 6. Configuration Management

#### 6.1 Environment Variables
```
GOOGLE_API_KEY=<optional - if set, uses Gemini; if not, uses local model>
DEFAULT_LANGUAGE=English|Spanish
DEFAULT_TEMPERATURE=1.0
```

#### 6.2 Configuration Loading Priority
1. Environment variables (.env file)
2. System environment variables
3. Default values in settings.py

### 7. Docker Configuration

#### 7.1 Dockerfile Requirements
- Base image: Python 3.11-slim
- Install system dependencies for audio processing
- Pre-download Piper TTS voices during build
- Pre-download Whisper small model during build
- Copy and install Python requirements
- Set appropriate working directory
- Expose Gradio port (7860)
- Non-root user for security

#### 7.2 Build Optimization
- Multi-stage build for smaller image
- Cache dependency layers
- Exclude unnecessary files via .dockerignore

### 8. Deployment Readiness

#### 8.1 Docker Container
The application must be fully functional when run as a Docker container:
```bash
docker build -t interactive-adventure .
docker run -p 7860:7860 interactive-adventure
```

#### 8.2 Environment Configuration
- Support for environment variable injection
- Configurable ports via environment variables
- Volume mounting for persistent data if needed

### 9. License

**License: MIT**

The project will be released under the MIT License, providing maximum flexibility for users while maintaining attribution requirements. All dependencies are compatible with MIT licensing.

### 10. Migration Strategy

#### 10.1 Code Extraction Process
1. Extract configuration constants to `config/settings.py`
2. Move LLM interaction logic to `models/` directory (keeping structure simple)
3. Separate audio processing into `audio/` modules
4. Extract UI components to `ui/interface.py`
5. Create story management logic in `core/`
6. Implement model abstraction layer with minimal wrapper complexity

#### 10.2 Preservation Requirements
- Maintain exact prompt structure and examples from notebook
- Preserve streaming functionality
- Keep all UI elements and layout unchanged
- Retain session state management using gr.State
- Maintain audio processing pipeline
- Minimize code modifications from original notebook

#### 10.3 Resource Management
- Store Piper TTS voices in `data/voices/` directory
- Store Whisper model in `data/models/` directory
- Implement automatic cleanup for temporary audio files
- All resources contained within project directory structure

### 11. Quality Assurance

#### 11.1 Functional Verification
- Text input generates appropriate responses
- Voice input correctly transcribed and processed
- Story endings detected properly
- Language switching works correctly
- Temperature adjustment affects output
- Local model fallback activates without API key

#### 11.2 Performance Criteria
- Response generation begins within 2 seconds
- Audio synthesis completes within 1 second per paragraph
- Local model inference remains responsive on CPU
- Memory usage stays below 2GB for local model

### 12. Additional Considerations

#### 12.1 Error Handling
- Graceful fallback when API key invalid
- Clear error messages for audio device issues
- Network timeout handling for API calls
- Model loading failure recovery

#### 12.2 Logging
- Structured logging with appropriate levels
- Separate logs for model selection, audio processing, and UI events
- No sensitive information in logs

#### 12.3 Implementation Notes
- LangChain integration approach for local model determined by implementer
- Implementation should prioritize simplicity and clarity over complexity
- Single-user assumption (no concurrency handling required)
- SmolLM2-135M model performance to be validated post-implementation
- Model upgrade path available if needed (larger SmolLM2 variants)

---

**Document Status**: Ready for Implementation