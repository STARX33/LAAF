"""
Test to see if ToolCallingAgent can work with OllamaModel
"""
from smolagents import ToolCallingAgent, CodeAgent, tool
from ollama_model import OllamaModel

@tool
def simple_tool(text: str) -> str:
    """A simple test tool.

    Args:
        text: Input text

    Returns:
        Modified text
    """
    return f"Processed: {text}"

# Test 1: Check what agent types require
print("=" * 60)
print("Testing agent compatibility with OllamaModel")
print("=" * 60)

try:
    print("\n1. Testing CodeAgent (current approach)...")
    model = OllamaModel(model_id="llama3")
    agent = CodeAgent(
        tools=[simple_tool],
        model=model,
        max_steps=2
    )
    print("   [OK] CodeAgent initialization: SUCCESS")
except Exception as e:
    print(f"   [FAIL] CodeAgent initialization: FAILED - {e}")

try:
    print("\n2. Testing ToolCallingAgent...")
    model = OllamaModel(model_id="llama3")
    # ToolCallingAgent expects: Callable[[list[dict[str, str]]], ChatMessage]
    # Our model has generate(prompt: str) -> ChatMessage
    # This might need adaptation
    agent = ToolCallingAgent(
        tools=[simple_tool],
        model=model.generate,  # Pass the generate method
        max_steps=2
    )
    print("   [OK] ToolCallingAgent initialization: SUCCESS")
except Exception as e:
    print(f"   [FAIL] ToolCallingAgent initialization: FAILED - {e}")

print("\n" + "=" * 60)
print("Analysis:")
print("=" * 60)
print("""
Key findings:
- CodeAgent expects: smolagents.models.Model (with generate() method)
- ToolCallingAgent expects: Callable[[list[dict]], ChatMessage]

ToolCallingAgent is designed for models with native tool calling support
(like OpenAI's function calling, Anthropic's tool use, etc.)

For local Ollama models without native tool calling:
  -> CodeAgent is the RIGHT choice
  -> ToolCallingAgent would require significant model wrapper modifications
""")
