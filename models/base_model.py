"""Abstract base class for language model providers."""

from abc import ABC, abstractmethod
from typing import Iterator, List
from langchain.schema import BaseMessage


class BaseModel(ABC):
    """Abstract base class for language model providers."""

    def __init__(self, temperature: float = 1.0):
        """Initialize the model with temperature setting."""
        self.temperature = temperature

    @abstractmethod
    def generate(self, messages: List[BaseMessage]) -> str:
        """Generate narrative response from message history.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Generated response text
        """
        pass

    @abstractmethod
    def stream(self, messages: List[BaseMessage]) -> Iterator[str]:
        """Stream narrative response token by token.
        
        Args:
            messages: List of conversation messages
            
        Yields:
            Response tokens as they are generated
        """
        pass

    def set_temperature(self, temperature: float) -> None:
        """Update the model temperature setting.
        
        Args:
            temperature: New temperature value (0-2)
        """
        if not (0 <= temperature <= 2):
            raise ValueError(f"Temperature must be between 0 and 2, got: {temperature}")
        self.temperature = temperature