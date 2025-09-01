"""Local model implementation."""

from typing import Iterator, List
from langchain.schema import BaseMessage, SystemMessage, HumanMessage, AIMessage
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from quanto import quantize, freeze

from .base_model import BaseModel


class LocalModel(BaseModel):
    """Local model implementation."""

    def __init__(self, temperature: float = 1.0):
        """Initialize local model.
        
        Args:
            temperature: Model temperature setting
        """
        super().__init__(temperature)
        from config.settings import settings
        self.model_name = settings.local_model_name
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the local model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.float32,
                device_map="auto",
                trust_remote_code=False
            )
            
            # Set pad token if not available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Apply int8 quantization to model weights
            try:
                quantize(model=self.model, weights="int8", activations=None)
                freeze(self.model)
            except Exception as quant_error:
                print(f"Warning: Quantization failed, continuing without quantization: {quant_error}")

        except Exception as e:
            raise RuntimeError(f"Failed to load local model: {e}")

    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert LangChain messages to model prompt format.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Formatted prompt string
        """
        # Convert LangChain messages to chat format
        chat_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                chat_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                chat_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                chat_messages.append({"role": "assistant", "content": message.content})
        
        # Use tokenizer's built-in chat template
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
            return self.tokenizer.apply_chat_template(
                chat_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback to simple format if no template available
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

    def _clean_response(self, text: str) -> str:
        """Clean response text by removing stray template tokens.
        
        Args:
            text: Raw response text
            
        Returns:
            Cleaned response text
        """
        if not text:
            return text
            
        # Remove common template tokens that might leak through
        tokens_to_remove = [
            "<|system|>", "<|user|>", "<|assistant|>", 
            "<|im_start|>", "<|im_end|>",
            "System:", "User:", "Assistant:",
            "[INST]", "[/INST]", "<s>", "</s>"
        ]
        
        cleaned = text
        for token in tokens_to_remove:
            cleaned = cleaned.replace(token, "")
        
        return cleaned

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
                max_length=2048
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=512,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only the new tokens
            new_tokens = outputs[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Clean up any stray template tokens
            response = self._clean_response(response)
            
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
            prompt = self._messages_to_prompt(messages)
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True,
                max_length=2048
            )
            
            input_length = inputs.input_ids.shape[1]
            
            # Generate with streaming
            with torch.no_grad():
                for _ in range(512):  # max_new_tokens
                    outputs = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=1,
                        temperature=self.temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # Check if we hit EOS token
                    if outputs[0][-1] == self.tokenizer.eos_token_id:
                        break
                    
                    # Decode the new token
                    new_token = outputs[0][input_length:]
                    if len(new_token) > 0:
                        token_text = self.tokenizer.decode(
                            new_token[-1:], 
                            skip_special_tokens=True
                        )
                        # Clean and yield token if not empty
                        cleaned_token = self._clean_response(token_text)
                        if cleaned_token.strip():
                            yield cleaned_token
                    
                    # Update inputs for next iteration
                    inputs.input_ids = outputs
                    # Update attention mask
                    attention_extension = torch.ones(
                        (1, 1), dtype=inputs.attention_mask.dtype
                    )
                    inputs.attention_mask = torch.cat(
                        [inputs.attention_mask, attention_extension], dim=1
                    )
                    input_length += 1
                    
        except Exception:
            # Fallback to non-streaming generation
            response = self.generate(messages)
            for token in response.split():
                yield token + " "

    def set_temperature(self, temperature: float) -> None:
        """Update the model temperature setting.
        
        Args:
            temperature: New temperature value (0-2)
        """
        super().set_temperature(temperature)