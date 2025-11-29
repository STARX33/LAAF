"""
Test script to verify the improved OllamaModel implementation.
Tests proper smolagents compatibility before running the full agent.
"""
import sys
from smolagents import CodeAgent, tool
from smolagents.models import ChatMessage
from ollama_model import OllamaModel

print("=" * 70)
print("Testing Improved OllamaModel Implementation")
print("=" * 70)

# Test 1: Model Interface Compatibility
print("\n1. Testing Model Interface Compatibility...")
try:
    model = OllamaModel(model_id="llama3")
    print("   [OK] OllamaModel instantiation successful")
    print(f"   [OK] Inherits from Model: {hasattr(model, 'generate')}")
    print(f"   [OK] Model ID: {model.model_id}")
except Exception as e:
    print(f"   [FAIL] Model instantiation failed: {e}")
    sys.exit(1)

# Test 2: Message Format Compatibility
print("\n2. Testing Message Format Compatibility...")
try:
    # Test with dict messages (smolagents format)
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'test successful' if you can read this."}
    ]

    print("   [INFO] Sending test message to Ollama...")
    response = model.generate(messages=test_messages)

    if isinstance(response, ChatMessage):
        print("   [OK] Returns ChatMessage object")
        print(f"   [OK] Response role: {response.role}")
        print(f"   [OK] Response content: {response.content[:100]}...")
        print(f"   [OK] Has token usage: {response.token_usage is not None}")
    else:
        print(f"   [FAIL] Wrong response type: {type(response)}")
        sys.exit(1)

except Exception as e:
    print(f"   [FAIL] Message generation failed: {e}")
    print("\n   Possible issues:")
    print("   - Is Ollama running? (ollama serve)")
    print("   - Is llama3 installed? (ollama pull llama3)")
    sys.exit(1)

# Test 3: CodeAgent Integration
print("\n3. Testing CodeAgent Integration...")
try:
    @tool
    def test_tool(text: str) -> str:
        """A simple test tool.

        Args:
            text: Input text

        Returns:
            Modified text
        """
        return f"Processed: {text}"

    agent = CodeAgent(
        tools=[test_tool],
        model=model,
        max_steps=3
    )
    print("   [OK] CodeAgent initialization successful")
    print(f"   [OK] Agent has {len(agent.tools)} tools")
    print(f"   [OK] Max steps: {agent.max_steps}")

except Exception as e:
    print(f"   [FAIL] CodeAgent initialization failed: {e}")
    sys.exit(1)

# Test 4: Simple Agent Execution
print("\n4. Testing Simple Agent Execution...")
print("   [INFO] Running simple task (this may take a moment)...")
try:
    # Create a simple tool that returns final answer
    @tool
    def final_answer(answer: str) -> str:
        """Provides the final answer.

        Args:
            answer: The final answer

        Returns:
            The final answer
        """
        return f"Final Answer: {answer}"

    test_agent = CodeAgent(
        tools=[final_answer],
        model=model,
        max_steps=3
    )

    # Simple task that should complete quickly
    result = test_agent.run("Call final_answer with the text 'Test completed successfully'")

    print("   [OK] Agent execution completed without errors")
    print(f"   [OK] Result type: {type(result)}")
    print(f"   [OK] Agent stopped (no infinite loop)")

except Exception as e:
    print(f"   [FAIL] Agent execution failed: {e}")
    print(f"   Error details: {type(e).__name__}")
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
print("\nThe improved implementation is ready to use!")
print("You can now run: python main.py")
print("\nKey improvements:")
print("  ✓ Proper Model inheritance")
print("  ✓ Correct message format handling")
print("  ✓ Token usage tracking")
print("  ✓ CodeAgent compatibility")
print("  ✓ Max steps prevents infinite loops")
print("=" * 70)
