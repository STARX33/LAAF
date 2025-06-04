import requests

class TokenUsage:
    def __init__(self, prompt_tokens=0, completion_tokens=0, total_tokens=0):
        self.input_tokens = prompt_tokens
        self.output_tokens = completion_tokens
        self.total_tokens = total_tokens

class ChatMessage:
    def __init__(self, role: str, content: str, token_usage=None):
        self.role = role
        self.content = content
        self.token_usage = token_usage or TokenUsage()

class OllamaModel:
    def __init__(self, model_id="llama3", base_url="http://localhost:11434"):
        self.model_id = model_id
        self.base_url = base_url

    def generate(self, prompt: str, **kwargs) -> ChatMessage:
        formatted_prompt = f"""### System:
You are a helpful assistant named Alfred. Stay concise.

### User:
{prompt}

### Assistant:"""

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_id,
                "prompt": formatted_prompt,
                "stream": False,
                "temperature": kwargs.get("temperature", 0.7)
            }
        )

        response.raise_for_status()
        result = response.json().get("response", "").strip()

        return ChatMessage(
            role="assistant",
            content=result,
            token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)  # Placeholder
        )
