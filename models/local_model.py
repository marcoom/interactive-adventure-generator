"""Local model implementation."""

from typing import Iterator, List
from langchain.schema import BaseMessage
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from quanto import quantize, freeze, qint8

from .base_model import BaseModel

# Constants
MAX_INPUT_LENGTH = 2048
MAX_NEW_TOKENS = 512


class LocalModel(BaseModel):
    """Local model implementation - mirrors GeminiModel architecture."""

    def __init__(self, temperature: float = 1.0):
        """Initialize local model directly in main process.
        
        Args:
            temperature: Model temperature setting
        """
        super().__init__(temperature)
        from config.settings import settings
        self.model_name = settings.local_model_name
        
        # Load model and tokenizer directly (like Gemini loads its LLM instances)
        self._load_model_instances()
    
    def _load_model_instances(self):
        """Load model and tokenizer instances."""
        try:
            print(f"Loading local model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.float32,
                device_map="auto",
                trust_remote_code=False
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Apply quantization for memory efficiency
            try:
                quantize(model=self.model, weights=qint8, activations=None)
                freeze(self.model)
                print("Model quantization applied successfully")
            except Exception as e:
                print(f"Warning: Quantization failed, continuing without: {e}")
                
            print("Local model loaded successfully")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load local model: {e}")
    
    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert LangChain messages to model prompt format."""
        chat_messages = []
        for message in messages:
            if hasattr(message, 'content'):
                if "SystemMessage" in str(type(message)):
                    chat_messages.append({"role": "system", "content": message.content})
                elif "HumanMessage" in str(type(message)):
                    chat_messages.append({"role": "user", "content": message.content})
                elif "AIMessage" in str(type(message)):
                    chat_messages.append({"role": "assistant", "content": message.content})
        
        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback format
            prompt_parts = []
            for msg in chat_messages:
                if msg["role"] == "system":
                    prompt_parts.append(f"System: {msg['content']}\n")
                elif msg["role"] == "user":
                    prompt_parts.append(f"User: {msg['content']}\n")
                elif msg["role"] == "assistant":
                    prompt_parts.append(f"Assistant: {msg['content']}\n")
            prompt_parts.append("Assistant: ")
            return "".join(prompt_parts)
    
    def generate(self, messages: List[BaseMessage]) -> str:
        """Generate narrative response from message history.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Generated response text
        """
        try:
            prompt = self._messages_to_prompt(messages)
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_INPUT_LENGTH
            )
            
            # Move inputs to the same device as the model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            raise RuntimeError(f"Local model generation failed: {e}")
    
    def stream(self, messages: List[BaseMessage]) -> Iterator[str]:
        """Stream narrative response token by token.
        
        Args:
            messages: List of conversation messages
            
        Yields:
            Response tokens as they are generated
        """
        try:
            # For simplicity, use generate() and yield word by word like Gemini
            response = self.generate(messages)
            words = response.split()
            for word in words:
                yield word + " "
        except Exception as e:
            raise RuntimeError(f"Local model streaming failed: {e}")
    
    def set_temperature(self, temperature: float) -> None:
        """Update the model temperature setting.
        
        Args:
            temperature: New temperature value (0-2)
        """
        super().set_temperature(temperature)
        # Temperature is applied in generate() method, no model recreation needed