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
        "You are the narrator of an interactive story. Write ONLY the story "
        "content itself - no explanations, no meta-commentary, no multiple "
        "options. Stay in character at all times. Begin with an engaging "
        "introduction: set the stage with vivid, sensory details; describe "
        "the setting, introduce key characters, and hint at the main conflict. "
        "Speak directly to the player using 'you' to draw them into the story. "
        "As the story unfolds, organically introduce decision points where you "
        "ask the player what will he do. Reflect the consequences of the "
        "player's choices, leading to multiple possible endings. If the "
        "narrative reaches a point where the character dies, end with exactly "
        "'The End.' If the story concludes naturally, finish with exactly "
        "'The End.' Always use the exact phrase 'The End.' in English, "
        "regardless of the language you are responding in. Never break "
        "character or explain what you are doing."
    )

    # Story generation guidelines
    INTRO_STORY_PROMPT = (
        "Start the interactive story immediately. Create a new setting, "
        "introduce the player's role, describe the environment with sensory "
        "details, introduce key characters, and hint at the main conflict. "
        "End with a single question asking what the player will do. Write "
        "only the story content - no explanations or options."
    )

    # Example scenarios - clean, direct story openings
    EXAMPLE1 = (
        "You are a warrior in the medieval town of Ravenshollow. The cobblestone "
        "streets echo with your heavy footsteps as grief weighs on your heart - "
        "your sister fell to an evil sorcerer's dark magic just days ago. The "
        "morning market bustles around you as you complete an errand for your "
        "friend Henrik, but a hooded stranger at a wooden table catches your eye. "
        "He stares directly at you with piercing blue eyes, his weathered hands "
        "drumming a strange rhythm on the table's surface. What will you do?"
    )

    EXAMPLE2 = (
        "The salty spray of the Caribbean sea hits your face as you stand on "
        "the deck of Blackbeard's ship, the Queen Anne's Revenge. Dawn breaks "
        "peacefully over calm waters when suddenly - THUD! - something massive "
        "strikes the hull. You glimpse a enormous tentacle, thick as a tree trunk, "
        "sliding back into the depths. When you turn to alert your crewmate "
        "beside you, he just shrugs. 'Heard nothin', saw nothin',' he mutters, "
        "returning to his rope work. The ocean surface shows no trace of the "
        "creature. What will you do?"
    )

    EXAMPLE3 = (
        "Captain's quarters aboard the starship Nebula's Edge feel unusually "
        "tense as you hunt the galaxy's most wanted thief, Lauren DeHugh. Your "
        "loyal crew has served you for years, but lately strange whispers echo "
        "through the corridors. The new passenger you picked up on Kepler Station "
        "keeps to herself, but something about her feels... familiar. You need to "
        "review the star charts before your next jump, but the nagging suspicion "
        "won't leave you alone. What will you do?"
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
            SystemMessage(content=self.get_system_message()),
            HumanMessage(content=self.build_initial_story_prompt())
        ]
        
        try:
            return self.model.generate(messages)
        except Exception:
            # Fallback message if generation fails
            return (
                "Welcome to your adventure! You find yourself standing at "
                "the edge of a mysterious forest, with an ancient path "
                "leading into the shadows. The air is thick with magic and "
                "possibility. What will you do?"
            )

    def get_system_message(self) -> str:
        """Get the narrator system message with language instructions.
        
        Returns:
            Complete system message for the narrator including language
        """
        system_parts = [self.NARRATOR_SYSTEM_MESSAGE]
        
        # Add language instruction
        language_prompt = self.get_language_prompt()
        if language_prompt:
            system_parts.append(f" {language_prompt}")
        
        return "".join(system_parts)

    def update_model_temperature(self, temperature: float) -> None:
        """Update the model temperature setting.
        
        Args:
            temperature: New temperature value (0-2)
        """
        self.model.set_temperature(temperature)