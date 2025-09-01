"""Story state and flow management."""

from typing import List, Tuple, Optional
from langchain.schema import BaseMessage, SystemMessage, AIMessage, HumanMessage

from models.base_model import BaseModel


class StoryManager:
    """Manages story state and conversation flow."""

    def __init__(self, model: BaseModel):
        """Initialize story manager with model.
        
        Args:
            model: Language model for generating responses
        """
        self.model = model
        self.chat_history: List[Tuple[str, str]] = []
        
    def reset_story(self, initial_message: str) -> List[Tuple[str, str]]:
        """Reset conversation with new initial message.
        
        Args:
            initial_message: New opening story message
            
        Returns:
            New chat history with initial message
        """
        self.chat_history = [("", initial_message)]
        return self.chat_history

    def is_story_ended(self, message: str) -> bool:
        """Check if story has reached an ending.
        
        Args:
            message: Message to check for ending condition
            
        Returns:
            True if story has ended
        """
        return "The End." in message

    def is_exit_command(self, user_input: str) -> bool:
        """Check if user wants to exit the story.
        
        Args:
            user_input: User's input text
            
        Returns:
            True if user wants to exit
        """
        return user_input.lower().strip() in {"q", "quit", "exit", "goodbye"}

    def build_message_history(
        self, 
        narrator_system_message: str
    ) -> List[BaseMessage]:
        """Build message history for model input.
        
        Args:
            narrator_system_message: System prompt for the narrator
            
        Returns:
            List of formatted messages for the model
        """
        messages = [SystemMessage(content=narrator_system_message)]
        
        # Handle initial message (empty human input)
        if self.chat_history and self.chat_history[0][0] == "":
            messages.append(AIMessage(content=self.chat_history[0][1]))
        else:
            # Add conversation history
            for human_msg, ai_msg in self.chat_history:
                if human_msg:  # Skip empty human messages
                    messages.append(HumanMessage(content=human_msg))
                if ai_msg:  # Skip empty AI messages
                    messages.append(AIMessage(content=ai_msg))
        
        return messages

    def add_user_message(self, user_input: str) -> None:
        """Add user message to chat history.
        
        Args:
            user_input: User's input text
        """
        self.chat_history.append((user_input, ""))

    def update_ai_response(self, response: str) -> None:
        """Update the latest AI response in chat history.
        
        Args:
            response: AI response text
        """
        if self.chat_history:
            user_input = self.chat_history[-1][0]
            self.chat_history[-1] = (user_input, response)

    def get_chat_history(self) -> List[Tuple[str, str]]:
        """Get current chat history.
        
        Returns:
            Current chat history
        """
        return self.chat_history.copy()

    def process_user_input(
        self,
        user_input: str,
        narrator_system_message: str,
        user_message: str = None
    ) -> Tuple[bool, bool]:
        """Process user input and determine next actions.
        
        Args:
            user_input: Raw user input
            narrator_system_message: System prompt for narrator
            user_message: Optional override message to add to history
            
        Returns:
            Tuple of (should_continue, is_exit_command)
        """
        # Check for exit commands
        if self.is_exit_command(user_input):
            self.chat_history.append((user_input, "The End."))
            return False, True
        
        # Check for empty input
        if not user_input.strip():
            return False, False
        
        # Add user message to history
        message_to_add = user_message if user_message else user_input
        self.add_user_message(message_to_add)
        
        return True, False

    def generate_response_stream(
        self,
        narrator_system_message: str,
        final_user_input: str
    ):
        """Generate streaming response from the model.
        
        Args:
            narrator_system_message: System prompt for narrator
            final_user_input: Final processed user input
            
        Yields:
            Response tokens and updated chat history
        """
        # Build message history for model
        messages = self.build_message_history(narrator_system_message)
        messages.append(HumanMessage(content=final_user_input))
        
        # Stream response from model
        generated_text = ""
        for token in self.model.stream(messages):
            generated_text += token
            # Update chat history with current generation
            self.update_ai_response(generated_text)
            yield token, self.get_chat_history()