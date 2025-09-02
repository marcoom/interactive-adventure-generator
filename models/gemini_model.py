"""Google Gemini model implementation."""

from typing import Iterator, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import BaseMessage

from .base_model import BaseModel


class GeminiModel(BaseModel):
    """Google Gemini 2.0 Flash model implementation."""

    def __init__(self, api_key: str, temperature: float = 1.0):
        """Initialize Gemini model with API key.
        
        Args:
            api_key: Google API key for authentication
            temperature: Model temperature setting
        """
        super().__init__(temperature)
        self.api_key = api_key
        self._llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=temperature
        )
        self._streaming_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            disable_streaming=False,
            temperature=temperature
        )

    def generate(self, messages: List[BaseMessage]) -> str:
        """Generate narrative response from message history.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Generated response text
        """
        try:
            response = self._llm.invoke(messages)
            return response.content
        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {e}")

    def stream(self, messages: List[BaseMessage]) -> Iterator[str]:
        """Stream narrative response token by token.
        
        Args:
            messages: List of conversation messages
            
        Yields:
            Response tokens as they are generated
        """
        try:
            for chunk in self._streaming_llm.stream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            raise RuntimeError(f"Gemini streaming failed: {e}")

    def set_temperature(self, temperature: float) -> None:
        """Update the model temperature setting.
        
        Args:
            temperature: New temperature value (0-2)
        """
        super().set_temperature(temperature)
        self._llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=self.api_key,
            temperature=temperature
        )
        self._streaming_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=self.api_key,
            disable_streaming=False,
            temperature=temperature
        )