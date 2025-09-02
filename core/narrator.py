"""Narrator agent logic and prompts."""

from typing import Optional
from langchain.schema import SystemMessage, HumanMessage

from models.base_model import BaseModel


class NarratorAgent:
    """Narrator agent for story generation with system prompts and examples.
    
    This class manages the AI narrator that generates interactive stories
    based on user choices and preferences.
    """

    # System prompt for the narrator
    NARRATOR_SYSTEM_MESSAGE = (
        "You are the narrator of an interactive story where the player's "
        "choices directly influence the progression and outcome of the "
        "narrative. Begin with an engaging introduction: set the stage with "
        "vivid, sensory details; describe the setting, introduce key "
        "characters, and hint at the main conflict. Speak directly to the "
        "player using 'you' to draw them into the story. As the story "
        "unfolds, organically introduce decision points where you ask the "
        "player what will he do. Reflect the consequences of the player's "
        "choices, leading to multiple possible endings. If the narrative "
        "reaches a point where the character dies, end with 'The End.' If "
        "the story concludes naturally, finish with 'The End.'"
    )

    # Story generation guidelines
    INTRO_STORY_PROMPT = (
        "Generate a welcome message for the interactive story with a new "
        "setting, introducing the player's role, environment, key characters, "
        "and hinting at the main conflict. End with a question asking what "
        "the player will do. Do not present possible alternatives, let the "
        "player create his own."
    )

    # Example scenarios
    EXAMPLE1 = (
        "Example 1: You are a warrior in a medieval town, your sister "
        "recently died at the hands of an evil sorcerer. You are currently "
        "heading to the market to complete an errand for a friend, where you "
        "find a stranger sitting on a table and mysteriously looking at you. "
        "What will you do?"
    )

    EXAMPLE2 = (
        "Example 2: You are a pirate in Blackbeard's ship. The morning was "
        "going as usual, with the salty odor and calm waters, until you hear "
        "a stomp and see a giant tentacle going into the water. You ask "
        "another tripulant but he didn't hear or see anything. What will you "
        "do?"
    )

    EXAMPLE3 = (
        "Example 3: You are a spaceship captain in the hunt for the infamous "
        "thief Lauren DeHugh, your crew follows you with pride and loyalty, "
        "but recently the moods have been weird, you suspect that the new "
        "passenger may have something to do, but it could also be nothing. "
        "Currently, you need to check the map and then you have a few "
        "minutes of spare time. What do you want to do?"
    )

    def __init__(self, model: BaseModel, language: str = "English"):
        """Initialize narrator agent.
        
        Args:
            model: Language model for generation
            language: Target language for responses
        """
        self.model = model
        self.language = language
        self.user_preferences: Optional[str] = None

    def set_language(self, language: str) -> None:
        """Set the target language for responses.
        
        Args:
            language: Target language ("English" or "Español")
        """
        self.language = language

    def set_user_preferences(self, preferences: str) -> None:
        """Set custom user preferences for story generation.
        
        Args:
            preferences: User's custom story preferences
        """
        self.user_preferences = preferences.strip() if preferences else None

    def get_language_prompt(self) -> str:
        """Get language-specific prompt.
        
        Returns:
            Language instruction prompt
        """
        return f"Answer always using the language {self.language}"

    def get_preferences_prompt(self) -> str:
        """Get user preferences prompt.
        
        Returns:
            User preferences instruction or empty string
        """
        if self.user_preferences:
            return (
                f"The story should take into consideration the following: "
                f"{self.user_preferences}"
            )
        return ""

    def build_initial_story_prompt(self) -> str:
        """Build complete initial story generation prompt.
        
        Returns:
            Complete prompt for initial story generation
        """
        prompt_parts = [
            self.INTRO_STORY_PROMPT,
            self.EXAMPLE1,
            self.EXAMPLE2,
            self.EXAMPLE3,
            self.get_language_prompt()
        ]
        
        preferences_prompt = self.get_preferences_prompt()
        if preferences_prompt:
            prompt_parts.append(preferences_prompt)
        
        return "".join(prompt_parts)

    def generate_initial_story(self) -> str:
        """Generate initial story message.
        
        Returns:
            Generated initial story text
        """
        messages = [
            SystemMessage(content=self.NARRATOR_SYSTEM_MESSAGE),
            HumanMessage(content=self.build_initial_story_prompt())
        ]
        
        try:
            return self.model.generate(messages)
        except Exception as e:
            # Fallback message if generation fails
            return (
                "Welcome to your adventure! You find yourself standing at "
                "the edge of a mysterious forest, with an ancient path "
                "leading into the shadows. The air is thick with magic and "
                "possibility. What will you do?"
            )

    def get_system_message(self) -> str:
        """Get the narrator system message.
        
        Returns:
            System message for the narrator
        """
        return self.NARRATOR_SYSTEM_MESSAGE

    def update_model_temperature(self, temperature: float) -> None:
        """Update the model temperature setting.
        
        Args:
            temperature: New temperature value (0-2)
        """
        self.model.set_temperature(temperature)