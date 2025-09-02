"""Local model implementation."""

import multiprocessing as mp
import queue
import atexit
import random
from typing import Iterator, List
from langchain.schema import (
    BaseMessage, SystemMessage, HumanMessage, AIMessage
)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from quanto import quantize, freeze

from .base_model import BaseModel

# Constants for timeouts and limits
MODEL_LOADING_TIMEOUT = 300  # 5 minutes
GENERATION_TIMEOUT = 60  # 1 minute
CLEANUP_TIMEOUT = 5  # seconds
MAX_INPUT_LENGTH = 2048
MAX_NEW_TOKENS = 512


class LocalModel(BaseModel):
    """Local model implementation with multiprocessing."""

    def __init__(self, temperature: float = 1.0):
        """Initialize local model with separate process.
        
        Args:
            temperature: Model temperature setting
        """
        super().__init__(temperature)
        from config.settings import settings
        self.model_name = settings.local_model_name
        
        # Create queues for communication with worker process
        self.request_queue = mp.Queue()
        self.response_queue = mp.Queue()
        
        # Start the LLM worker process
        self.llm_process = mp.Process(
            target=self._llm_worker,
            args=(self.request_queue, self.response_queue, self.model_name)
        )
        self.llm_process.start()
        
        # Register cleanup function
        atexit.register(self._cleanup)
        
        # Wait for model to be ready
        self._wait_for_ready()

    def _wait_for_ready(self):
        """Wait for the LLM process to be ready."""
        try:
            # Send ready check and wait for response
            self.request_queue.put({"type": "ready_check"})
            response = self.response_queue.get(timeout=MODEL_LOADING_TIMEOUT)
            if response.get("status") != "ready":
                error_msg = response.get('error', 'Unknown error')
                raise RuntimeError(f"LLM process failed to start: {error_msg}")
        except queue.Empty:
            raise RuntimeError(
                "LLM process failed to start within timeout period"
            )
    
    def _cleanup(self):
        """Clean up the LLM process."""
        if hasattr(self, 'llm_process') and self.llm_process.is_alive():
            self.request_queue.put({"type": "shutdown"})
            self.llm_process.join(timeout=CLEANUP_TIMEOUT)
            if self.llm_process.is_alive():
                self.llm_process.terminate()
                self.llm_process.join()
    
    @staticmethod
    def _llm_worker(request_queue, response_queue, model_name):
        """Worker process that handles the actual LLM operations."""
        tokenizer = None
        model = None
        
        def load_model():
            """Load the model and tokenizer in the worker process."""
            nonlocal tokenizer, model
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float32,
                    device_map="auto",
                    trust_remote_code=False
                )
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                try:
                    quantize(model=model, weights="int8", activations=None)
                    freeze(model)
                except Exception as quant_error:
                    print(f"Warning: Quantization failed, continuing without quantization: {quant_error}")
                
                return True
            except Exception as e:
                print(f"Failed to load local model: {e}")
                return False
        
        def messages_to_prompt(messages):
            """Convert LangChain messages to model prompt format."""
            chat_messages = []
            for message in messages:
                if isinstance(message, dict):
                    # Handle serialized message format
                    if message.get("type") == "system":
                        chat_messages.append({"role": "system", "content": message["content"]})
                    elif message.get("type") == "human":
                        chat_messages.append({"role": "user", "content": message["content"]})
                    elif message.get("type") == "ai":
                        chat_messages.append({"role": "assistant", "content": message["content"]})
                else:
                    # Handle direct BaseMessage objects (fallback)
                    if hasattr(message, 'content'):
                        if "SystemMessage" in str(type(message)):
                            chat_messages.append({"role": "system", "content": message.content})
                        elif "HumanMessage" in str(type(message)):
                            chat_messages.append({"role": "user", "content": message.content})
                        elif "AIMessage" in str(type(message)):
                            chat_messages.append({"role": "assistant", "content": message.content})
            
            # Use tokenizer's built-in chat template
            has_template = (hasattr(tokenizer, 'apply_chat_template') and 
                           tokenizer.chat_template is not None)
            if has_template:
                return tokenizer.apply_chat_template(
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
        
        def clean_response(text):
            """Clean response text by removing stray template tokens."""
            if not text:
                return text
                
            tokens_to_remove = [
                "<|system|>", "<|user|>", "<|assistant|>", 
                "<|im_start|>", "<|im_end|>", "System:", "User:", 
                "Assistant:", "[INST]", "[/INST]", "<s>", "</s>"
            ]
            
            cleaned = text
            for token in tokens_to_remove:
                cleaned = cleaned.replace(token, "")
            
            return cleaned
        
        def handle_generate_request(request):
            """Handle text generation request."""
            try:
                # Set random seed for different results each time
                seed = random.randint(0, 2**32 - 1)
                torch.manual_seed(seed)
                
                prompt = messages_to_prompt(request["messages"])
                temperature = request.get("temperature", 1.0)
                
                inputs = tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True,
                    max_length=MAX_INPUT_LENGTH
                )
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                new_tokens = outputs[0][inputs.input_ids.shape[1]:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                response = clean_response(response)
                
                response_queue.put({"status": "success", "response": response.strip()})
            except Exception as e:
                response_queue.put({"status": "error", "error": str(e)})

        def handle_stream_request(request):
            """Handle streaming generation request."""
            try:
                # Set random seed for different results each time
                seed = random.randint(0, 2**32 - 1)
                torch.manual_seed(seed)
                
                prompt = messages_to_prompt(request["messages"])
                temperature = request.get("temperature", 1.0)
                
                inputs = tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True,
                    max_length=MAX_INPUT_LENGTH
                )
                
                input_length = inputs.input_ids.shape[1]
                response_queue.put({"status": "stream_start"})
                
                with torch.no_grad():
                    for _ in range(MAX_NEW_TOKENS):
                        outputs = model.generate(
                            inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            max_new_tokens=1,
                            temperature=temperature,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id
                        )
                        
                        if outputs[0][-1] == tokenizer.eos_token_id:
                            break
                        
                        new_token = outputs[0][input_length:]
                        if len(new_token) > 0:
                            token_text = tokenizer.decode(
                                new_token[-1:], 
                                skip_special_tokens=True
                            )
                            cleaned_token = clean_response(token_text)
                            if cleaned_token.strip():
                                response_queue.put({"status": "token", "token": cleaned_token})
                        
                        inputs.input_ids = outputs
                        attention_extension = torch.ones((1, 1), dtype=inputs.attention_mask.dtype)
                        inputs.attention_mask = torch.cat([inputs.attention_mask, attention_extension], dim=1)
                        input_length += 1
                
                response_queue.put({"status": "stream_end"})
            except Exception as e:
                response_queue.put({"status": "error", "error": str(e)})

        # Load model on startup
        model_loaded = load_model()
        
        if model_loaded:
            response_queue.put({"status": "ready"})
        else:
            response_queue.put({"status": "error", "error": "Failed to load model"})
            return
        
        # Main worker loop
        while True:
            try:
                request = request_queue.get(timeout=1.0)
                
                if request["type"] == "shutdown":
                    break
                elif request["type"] == "ready_check":
                    response_queue.put({"status": "ready"})
                elif request["type"] == "generate":
                    handle_generate_request(request)
                elif request["type"] == "stream":
                    handle_stream_request(request)
            
            except queue.Empty:
                continue
            except Exception as e:
                response_queue.put({"status": "error", "error": str(e)})

    def _serialize_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain messages to serializable format.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            List of serialized message dictionaries
        """
        serialized = []
        for message in messages:
            if isinstance(message, SystemMessage):
                serialized.append({"type": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                serialized.append({"type": "human", "content": message.content})
            elif isinstance(message, AIMessage):
                serialized.append({"type": "ai", "content": message.content})
        return serialized


    def generate(self, messages: List[BaseMessage]) -> str:
        """Generate narrative response from message history.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Generated response text
        """
        try:
            # Serialize messages for IPC
            serialized_messages = self._serialize_messages(messages)
            
            # Send generation request to worker process
            self.request_queue.put({
                "type": "generate",
                "messages": serialized_messages,
                "temperature": self.temperature
            })
            
            # Wait for response
            response = self.response_queue.get(timeout=GENERATION_TIMEOUT)
            
            if response["status"] == "success":
                return response["response"]
            else:
                raise RuntimeError(f"LLM generation failed: {response.get('error', 'Unknown error')}")
            
        except queue.Empty:
            raise RuntimeError("Local model generation timed out")
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
            # Serialize messages for IPC
            serialized_messages = self._serialize_messages(messages)
            
            # Send streaming request to worker process
            self.request_queue.put({
                "type": "stream",
                "messages": serialized_messages,
                "temperature": self.temperature
            })
            
            # Stream tokens from worker process
            while True:
                try:
                    response = self.response_queue.get(timeout=GENERATION_TIMEOUT)
                    
                    if response["status"] == "stream_start":
                        continue  # Start of stream, keep waiting for tokens
                    elif response["status"] == "token":
                        yield response["token"]
                    elif response["status"] == "stream_end":
                        break  # End of stream
                    elif response["status"] == "error":
                        raise RuntimeError(f"Streaming failed: {response.get('error', 'Unknown error')}")
                
                except queue.Empty:
                    raise RuntimeError("Streaming timed out")
            
        except Exception as e:
            # Fallback to non-streaming generation
            try:
                response = self.generate(messages)
                for token in response.split():
                    yield token + " "
            except Exception:
                raise RuntimeError(f"Local model streaming failed: {e}")

    def set_temperature(self, temperature: float) -> None:
        """Update the model temperature setting.
        
        Args:
            temperature: New temperature value (0-2)
        """
        super().set_temperature(temperature)
    
    def __del__(self):
        """Destructor to clean up resources."""
        self._cleanup()