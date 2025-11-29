"""
LAAF (Local Agentic AI Framework) - Main Runner
Privacy-first agentic AI with RAG, running fully offline.
"""
from dotenv import load_dotenv
from smolagents import CodeAgent
from tools import final_answer
from vision_tools import (
    detect_objects_from_latest_image,
    describe_image_with_blip
)
from ollama_model import OllamaModel, get_agentic_system_prompt
from rag_loader import _load_text_rag_context_impl, _load_image_rag_captions_impl

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION - Easy model upgrades here!
# =============================================================================

# Model Configuration
MODEL_ID = "qwen2:7b"  # Options: llama3, llama3.1, llama3.2, qwen2.5, qwen2.5-coder
TEMPERATURE = 0.3    # Lower = more focused, Higher = more creative (optimized for tool calling)
MAX_STEPS = 4        # Maximum agent steps before stopping (prevents infinite loops)

# RAG Configuration
ENABLE_TEXT_RAG = True   # Load text documents from rag/text/
ENABLE_IMAGE_RAG = True  # Load and caption images from rag/images/

# =============================================================================
# INITIALIZE COMPONENTS
# =============================================================================

print("=" * 70)
print("LAAF - Local Agentic AI Framework")
print("=" * 70)
print(f"Model: {MODEL_ID}")
print(f"Max Steps: {MAX_STEPS}")
print(f"Temperature: {TEMPERATURE}")
print("-" * 70)

# Load RAG context
rag_text = _load_text_rag_context_impl() if ENABLE_TEXT_RAG else ""
rag_images = _load_image_rag_captions_impl() if ENABLE_IMAGE_RAG else ""
rag_context = rag_text + rag_images

if not rag_context.strip():
    print("[RAG] WARNING: No RAG context loaded. Running with image-only analysis.")
else:
    print("[RAG] Context loaded successfully")
    if rag_text:
        text_count = len([line for line in rag_text.split('\n') if line.strip()])
        print(f"      - Text documents: {text_count} lines")
    if rag_images:
        img_count = rag_images.count("From image:")
        print(f"      - Image captions: {img_count} images")

print("-" * 70)

# Initialize Ollama model with proper smolagents compatibility
model = OllamaModel(
    model_id=MODEL_ID,
    temperature=TEMPERATURE
)

# Create Alfred agent with proper configuration
agent = CodeAgent(
    tools=[
        final_answer,              # CRITICAL: Tool to end the task
        detect_objects_from_latest_image,
        describe_image_with_blip,
        # Note: Removed RAG loading tools since context is pre-loaded
    ],
    model=model,
    max_steps=MAX_STEPS,          # Prevents infinite loops
    verbosity_level=2             # Show detailed step-by-step execution
)

print("[Agent] CodeAgent initialized")
print(f"[Agent] Tools available: {len(agent.tools)}")
print("-" * 70)

# =============================================================================
# BUILD TASK PROMPT
# =============================================================================

# Build a clear, directive task prompt
task_parts = []

# Add RAG context if available
if rag_context.strip():
    task_parts.append(f"CONTEXT INFORMATION:\n{rag_context}\n")

# Add clear instructions
task_parts.append("""TASK:
1. Analyze the latest image in the input_images/ folder
2. Use the provided context information to inform your analysis
3. If the context suggests something different from what you see, trust the context
4. Provide a clear, definitive conclusion using the final_answer tool

IMPORTANT: You MUST call final_answer(answer="your conclusion") when you're ready to provide your answer.""")

task_prompt = "\n".join(task_parts)

# =============================================================================
# RUN AGENT
# =============================================================================

print("[Agent] Starting execution...")
print("=" * 70)

try:
    result = agent.run(task_prompt)

    print("=" * 70)
    print("[Agent] Task completed successfully!")
    print("=" * 70)
    print("\nFINAL RESULT:")
    print("-" * 70)
    print(result)
    print("-" * 70)

except Exception as e:
    print("=" * 70)
    print("[Agent] Error occurred:")
    print("=" * 70)
    print(f"{type(e).__name__}: {e}")
    print("-" * 70)
    print("\nTroubleshooting:")
    print("1. Ensure Ollama is running: ollama serve")
    print(f"2. Ensure {MODEL_ID} is installed: ollama pull {MODEL_ID}")
    print("3. Check if an image exists in input_images/")
    print("4. Review the error message above for specific issues")
