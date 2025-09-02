"""Gradio user interface for Interactive Adventure Generator."""

import gradio as gr
from typing import Iterator, Tuple, List, Optional

from core.story_manager import StoryManager
from core.narrator import NarratorAgent
from audio.tts import TTSEngine
from audio.stt import STTEngine
from utils.helpers import (
    create_model_instance, process_multimodal_input,
    validate_language, validate_temperature, format_error_message
)
from config.settings import settings


class GradioInterface:
    """Gradio interface for the Interactive Adventure Generator."""

    def __init__(self):
        """Initialize the Gradio interface."""
        self.model = create_model_instance()
        self.narrator = NarratorAgent(self.model, settings.default_language)
        self.story_manager = StoryManager(self.model)
        self.tts_engine = TTSEngine()
        self.stt_engine = STTEngine()
        
        # Initialize with first story
        self.initial_story = self.narrator.generate_initial_story()
        self.initial_chat_history = [("", self.initial_story)]
        
        # Generate initial audio
        self.initial_audio = None
        if self.tts_engine.is_available(settings.default_language):
            self.initial_audio = self.tts_engine.synthesize(
                self.initial_story, settings.default_language
            )

    def integrated_generate(
        self,
        user_input,
        selected_language: str,
        autoplay_enabled: bool,
        temperature_value: float,
        chat_history: List[Tuple[str, str]]
    ) -> Iterator[Tuple[List[Tuple[str, str]], List[Tuple[str, str]], 
                       gr.update, Optional[gr.update]]]:
        """Process user input and stream LLM response with audio generation."""
        try:
            # Update model temperature
            self.model.set_temperature(temperature_value)
            self.narrator.set_language(selected_language)
            
            # Update story manager chat history
            self.story_manager.chat_history = chat_history.copy()
            
            # Process multimodal input
            input_text, recorded_audio = process_multimodal_input(user_input)
            
            # Handle audio transcription
            final_input = input_text
            if not final_input and recorded_audio:
                if self.stt_engine.is_available():
                    final_input = self.stt_engine.transcribe(
                        recorded_audio, selected_language
                    )
                    if not final_input:
                        yield (
                            chat_history, chat_history,
                            gr.update(interactive=True, value=""),
                            None
                        )
                        return

            # Process user input through story manager
            should_continue, is_exit = self.story_manager.process_user_input(
                final_input, self.narrator.get_system_message(), final_input
            )
            
            if not should_continue:
                if is_exit:
                    yield (
                        self.story_manager.get_chat_history(),
                        self.story_manager.get_chat_history(),
                        gr.update(interactive=False, value=""),
                        None
                    )
                else:
                    yield (
                        chat_history, chat_history,
                        gr.update(interactive=True, value=""),
                        None
                    )
                return

            # Stream response generation
            system_msg = self.narrator.get_system_message()
            for token, updated_history in self.story_manager.generate_response_stream(
                system_msg, final_input
            ):
                yield (
                    updated_history, updated_history,
                    gr.update(interactive=True, value=""),
                    None
                )

            # Check if story ended and generate audio
            current_history = self.story_manager.get_chat_history()
            if current_history:
                last_response = current_history[-1][1]
                story_ended = self.story_manager.is_story_ended(last_response)
                
                # Generate narration audio
                narration_audio = None
                if self.tts_engine.is_available(selected_language):
                    narration_audio = self.tts_engine.synthesize(
                        last_response, selected_language
                    )

                yield (
                    current_history, current_history,
                    gr.update(interactive=not story_ended, value=""),
                    gr.update(
                        autoplay=autoplay_enabled,
                        value=narration_audio
                    ) if narration_audio else None
                )

        except Exception as e:
            error_msg = format_error_message(e, "processing input")
            print(f"Error in integrated_generate: {error_msg}")
            yield (
                chat_history, chat_history,
                gr.update(interactive=True, value=""),
                None
            )

    def reset_chat(
        self,
        selected_language: str,
        autoplay_enabled: bool,
        temperature_value: float,
        user_preferences: str
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], 
               gr.update, Optional[gr.update]]:
        """Reset conversation with updated settings."""
        try:
            # Validate inputs
            validate_language(selected_language)
            validate_temperature(temperature_value)
            
            # Update model and narrator settings
            self.model.set_temperature(temperature_value)
            self.narrator.set_language(selected_language)
            self.narrator.set_user_preferences(user_preferences)
            
            # Generate new story
            new_message = self.narrator.generate_initial_story()
            new_history = [("", new_message)]
            
            # Reset story manager
            self.story_manager.reset_story(new_message)
            
            # Generate initial audio
            initial_audio = None
            if self.tts_engine.is_available(selected_language):
                initial_audio = self.tts_engine.synthesize(
                    new_message, selected_language
                )

            return (
                new_history, new_history,
                gr.update(interactive=True, value=""),
                gr.update(
                    autoplay=autoplay_enabled,
                    value=initial_audio
                ) if initial_audio else gr.update(value=None)
            )

        except Exception as e:
            error_msg = format_error_message(e, "resetting story")
            print(f"Error in reset_chat: {error_msg}")
            
            # Return current state on error
            return (
                self.initial_chat_history, self.initial_chat_history,
                gr.update(interactive=True, value=""),
                gr.update(value=self.initial_audio)
            )

    def create_interface(self) -> gr.Blocks:
        """Create and configure the Gradio interface."""
        with gr.Blocks(title="Interactive Adventure Generator") as demo:
            # Header
            gr.Markdown("## Interactive Adventure")
            
            # Chat display
            chat_box = gr.Chatbot(
                value=self.initial_chat_history,
                type="tuples",
                height=400
            )
            chat_state = gr.State(self.initial_chat_history)
            
            # Input and audio section
            with gr.Row():
                with gr.Column(scale=3):
                    user_input = gr.MultimodalTextbox(
                        label="Your action",
                        placeholder="Type your action or press the microphone icon",
                        sources=["microphone"]
                    )
                with gr.Column(scale=1):
                    narration_audio = gr.Audio(
                        value=self.initial_audio,
                        label="Narration",
                        type="filepath",
                        autoplay=False
                    )
            
            # Settings and reset section
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Accordion("Additional Settings", open=False):
                        setting_language = gr.Dropdown(
                            choices=settings.supported_languages,
                            value=settings.default_language,
                            label="Language"
                        )
                        setting_autoplay = gr.Checkbox(
                            value=False,
                            label="Autoplay Narration"
                        )
                        setting_temperature = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            step=0.1,
                            value=settings.default_temperature,
                            label="Temperature"
                        )
                        setting_preferences = gr.Textbox(
                            label="User Preferences",
                            placeholder="Write your own guidelines to the story",
                            lines=2
                        )
                with gr.Column(scale=1):
                    reset_button = gr.Button("Start Again", variant="primary")
            
            # Event handlers
            user_input.submit(
                fn=self.integrated_generate,
                inputs=[
                    user_input, setting_language, setting_autoplay,
                    setting_temperature, chat_state
                ],
                outputs=[chat_box, chat_state, user_input, narration_audio]
            )
            
            reset_button.click(
                fn=self.reset_chat,
                inputs=[
                    setting_language, setting_autoplay,
                    setting_temperature, setting_preferences
                ],
                outputs=[chat_box, chat_state, user_input, narration_audio],
                queue=False
            )

        return demo

    def launch(self, share: bool = False) -> None:
        """Launch the Gradio interface."""
        import signal
        import sys
        
        demo = self.create_interface()
        
        def signal_handler(sig, frame):
            print("\n👋 Shutting down gracefully...")
            self.cleanup()
            sys.exit(0)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            demo.launch(
                server_name=settings.gradio_host,
                server_port=settings.gradio_port,
                share=share,
                show_error=True,
                prevent_thread_lock=False
            )
        except KeyboardInterrupt:
            signal_handler(signal.SIGINT, None)

    def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            # Cleanup TTS temporary files
            if hasattr(self, 'tts_engine'):
                self.tts_engine.cleanup_temp_files()
        except Exception:
            pass