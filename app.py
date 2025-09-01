#!/usr/bin/env python3
"""
Interactive Adventure Generator

An AI-powered storytelling application that creates dynamic, interactive 
narratives where user choices directly influence story progression and outcomes.

Supports both Google Gemini (with API key) and local models.
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def load_environment():
    """Load environment variables from .env file if present."""
    try:
        from dotenv import load_dotenv
        env_path = PROJECT_ROOT / '.env'
        if env_path.exists():
            load_dotenv(env_path)
            print(f"Loaded environment from: {env_path}")
    except ImportError:
        # python-dotenv not available, skip loading
        pass


def validate_environment():
    """Validate environment and display configuration info."""
    # Import settings after environment is loaded
    from config.settings import settings
    from utils.helpers import get_model_info
    
    try:
        settings.validate()
        print("✓ Configuration validated successfully")
    except Exception as e:
        print(f"⚠ Configuration warning: {e}")
    
    # Display model information
    model_info = get_model_info()
    print(f"Model: {model_info['name']} ({model_info['provider']})")
    print(f"Type: {'Local' if model_info['type'] == 'local' else 'Remote'}")
    
    if model_info['requires_api_key'] and not settings.google_api_key:
        print("⚠ No Google API key found - will use local model")


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive Adventure Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  GOOGLE_API_KEY    Google API key (optional - uses local model if not set)
  DEFAULT_LANGUAGE  Default language (English or Español)
  DEFAULT_TEMPERATURE  Default model temperature (0.0-2.0)
  LOCAL_MODEL_NAME  Local model to use (default: SmolLM2-135M-Instruct)
  GRADIO_PORT      Server port (default: 7860)
  GRADIO_HOST      Server host (default: 0.0.0.0)

Examples:
  python app.py                    # Start with default settings
  python app.py --share           # Start with public sharing enabled
  python app.py --port 8080       # Start on custom port
  
For Docker:
  docker run -p 7860:7860 interactive-adventure
        """
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=None,
        help='Server port (overrides GRADIO_PORT)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default=None,
        help='Server host (overrides GRADIO_HOST)'
    )
    
    parser.add_argument(
        '--share', '-s',
        action='store_true',
        help='Create public sharing link'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )

    args = parser.parse_args()
    
    # Override settings with command line arguments
    if args.port:
        settings.gradio_port = args.port
    if args.host:
        settings.gradio_host = args.host
    
    if args.debug:
        warnings.filterwarnings('default')
        print("Debug mode enabled")

    print("🎭 Interactive Adventure Generator")
    print("=" * 50)
    
    # Load environment variables
    load_environment()
    
    # Import settings after environment is loaded and reload them
    from config.settings import settings
    from ui.interface import GradioInterface
    from utils.helpers import setup_environment
    
    # Reload settings to pick up environment variables
    settings.reload()
    
    # Set up environment
    setup_environment()
    
    # Validate configuration
    validate_environment()
    
    print("=" * 50)
    print(f"Starting server on {settings.gradio_host}:{settings.gradio_port}")
    
    if settings.use_local_model:
        print("⚠ Note: Local model may take longer to load on first run")
    
    try:
        # Create and launch interface
        interface = GradioInterface()
        interface.launch(share=args.share)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
        
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        if args.debug:
            raise
        sys.exit(1)
    
    finally:
        # Cleanup
        try:
            from utils.helpers import cleanup_temp_files
            cleanup_temp_files()
        except Exception:
            pass


if __name__ == "__main__":
    main()