"""
Proper Ollama model wrapper for smolagents framework.
Implements the smolagents.models.Model interface for local Ollama models.
"""
import requests
from typing import Optional
from smolagents.models import Model, ChatMessage
from smolagents.monitoring import TokenUsage


class OllamaModel(Model):
    """
    Ollama model wrapper compatible with smolagents CodeAgent.

    Supports easy model switching for upgrades:
    - llama3, llama3.1, llama3.2, llama3.3
    - qwen2.5, qwen2.5-coder (excellent for agentic tasks)
    - mistral, mixtral
    - And any other Ollama-supported model

    Args:
        model_id: Ollama model identifier (default: "llama3")
        base_url: Ollama API base URL (default: "http://localhost:11434")
        temperature: Sampling temperature (default: 0.7)
        **kwargs: Additional arguments passed to Model base class
    """

    def __init__(
        self,
        model_id: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        **kwargs
    ):
        # Initialize parent Model class
        super().__init__(model_id=model_id, **kwargs)
        self.base_url = base_url
        self.temperature = temperature

    def generate(
        self,
        messages: list[dict[str, str] | ChatMessage],
        stop_sequences: Optional[list[str]] = None,
        response_format: Optional[dict[str, str]] = None,
        tools_to_call_from: Optional[list] = None,
        **kwargs
    ) -> ChatMessage:
        """
        Generate a response using Ollama's chat API.

        Args:
            messages: List of message dicts or ChatMessage objects
            stop_sequences: Optional stop sequences
            response_format: Optional response format specification
            tools_to_call_from: Optional list of tools (unused by Ollama but required by interface)
            **kwargs: Additional generation parameters

        Returns:
            ChatMessage with the model's response
        """
        # Convert messages to Ollama format
        ollama_messages = self._convert_messages(messages)

        # Prepare request payload
        payload = {
            "model": self.model_id,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
            }
        }

        # Add stop sequences if provided
        if stop_sequences:
            payload["options"]["stop"] = stop_sequences

        # Make request to Ollama chat API
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120  # 2 minute timeout for long generations
            )
            response.raise_for_status()
            result = response.json()

            # Extract response content
            content = result.get("message", {}).get("content", "").strip()

            # Extract token usage if available
            prompt_tokens = result.get("prompt_eval_count", 0)
            completion_tokens = result.get("eval_count", 0)

            # Return ChatMessage with proper token usage
            return ChatMessage(
                role="assistant",
                content=content,
                token_usage=TokenUsage(
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens
                )
            )

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API request failed: {e}")

    def _convert_messages(
        self,
        messages: list[dict[str, str] | ChatMessage]
    ) -> list[dict[str, str]]:
        """
        Convert smolagents messages to Ollama chat format.

        Args:
            messages: List of message dicts or ChatMessage objects

        Returns:
            List of message dicts in Ollama format
        """
        ollama_messages = []

        for msg in messages:
            if isinstance(msg, ChatMessage):
                # Convert ChatMessage object
                ollama_messages.append({
                    "role": msg.role,
                    "content": msg.content or ""
                })
            elif isinstance(msg, dict):
                # Already in dict format, pass through
                # Handle both 'content' (string) and 'content' (list) formats
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Extract text from list format (for multimodal messages)
                    text_parts = [item.get("text", "") for item in content if isinstance(item, dict) and "text" in item]
                    content = " ".join(text_parts)

                ollama_messages.append({
                    "role": msg.get("role", "user"),
                    "content": content
                })

        return ollama_messages


def get_agentic_system_prompt() -> str:
    """
    Returns an optimized system prompt for agentic behavior with CodeAgent.
    This guides the model to properly use tools and reach final_answer.
    """
    return """You are Alfred, a capable AI assistant with access to tools.

Your task is to help users by breaking down problems and using available tools effectively.

IMPORTANT GUIDELINES:
1. Think step by step about what information you need
2. Use tools to gather information or perform actions
3. After gathering sufficient information, call final_answer with your conclusion
4. Be concise and focused - avoid unnecessary tool calls
5. Always end your work by calling final_answer(answer="your conclusion")

Remember: You must call final_answer to complete the task successfully."""
