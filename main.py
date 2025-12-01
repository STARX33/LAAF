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
from document_tools import (
    extract_text_from_pdf,
    ocr_pdf,
    ocr_image,
    parse_csv_file,
    summarize_document,
    list_documents_in_folder,
    analyze_document_in_folder,
    search_document_section,  # NEW: Search for specific sections in PDFs
    extract_key_elements,
    save_document_summary,
    check_document_tools_status
)
from web_search_tool import (
    search_web_for_research,
    find_research_sources
)
from ollama_model import OllamaModel, get_agentic_system_prompt
from rag_loader import _load_text_rag_context_impl, _load_image_rag_captions_impl

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION - Easy model upgrades here!
# =============================================================================

# Model Configuration
# Accuracy test results (2025-11-29):
#   - qwen2:7b: 100% (4/4 tests) - RECOMMENDED
#   - llama3:   50%  (2/4 tests) - baseline
# qwen2:7b shows better tool selection, less hallucination, better metadata extraction
MODEL_ID = "qwen2:7b"  # Options: llama3, qwen2:7b, qwen:7b, llama3.2
TEMPERATURE = 0.3    # Lower = more focused (0.3 optimal for tool reasoning)
MAX_STEPS = 6        # Maximum agent steps before stopping (increased for multi-step workflows)

# RAG Configuration
ENABLE_TEXT_RAG = True   # Load text documents from rag/text/
ENABLE_IMAGE_RAG = True  # Load and caption images from rag/images/

# Document Processing Mode
DOCUMENT_MODE = True     # Enable document processing tools (PDF, CSV, OCR)

# Phase 1 Configuration
ENABLE_WEB_RESEARCH = True   # Enable web search for research validation
ENABLE_KEY_EXTRACTION = True # Enable semantic key element extraction
AUTO_SAVE_SUMMARIES = True   # Automatically save document summaries

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

# Check document processing capabilities
if DOCUMENT_MODE:
    check_document_tools_status()

# Initialize Ollama model with proper smolagents compatibility
model = OllamaModel(
    model_id=MODEL_ID,
    temperature=TEMPERATURE
)

# Build tool list based on configuration
tools_list = [
    final_answer,              # CRITICAL: Tool to end the task
]

# Add vision tools
tools_list.extend([
    detect_objects_from_latest_image,
    describe_image_with_blip,
])

# Add document processing tools
if DOCUMENT_MODE:
    tools_list.extend([
        search_document_section,     # BEST: Search for specific section in a PDF
        extract_text_from_pdf,       # Extract full text from PDF
        analyze_document_in_folder,  # Auto-discovery when filename unknown
        list_documents_in_folder,    # List available documents
        ocr_pdf,
        ocr_image,
        parse_csv_file,
        summarize_document,
    ])

# Add Phase 1 intelligence tools
if ENABLE_KEY_EXTRACTION:
    tools_list.append(extract_key_elements)
    tools_list.append(save_document_summary)

if ENABLE_WEB_RESEARCH:
    tools_list.extend([
        search_web_for_research,
        find_research_sources
    ])

# Create Alfred agent with proper configuration
agent = CodeAgent(
    tools=tools_list,
    model=model,
    max_steps=MAX_STEPS,          # Prevents infinite loops
    verbosity_level=2             # Show detailed step-by-step execution
)

print("[Agent] CodeAgent initialized (Phase 1: Master Document Parser)")
print(f"[Agent] Tools available: {len(agent.tools)}")
if ENABLE_WEB_RESEARCH:
    print("[Phase 1] Web research enabled")
if ENABLE_KEY_EXTRACTION:
    print("[Phase 1] Key element extraction enabled")
if AUTO_SAVE_SUMMARIES:
    print("[Phase 1] Auto-save summaries enabled")
print("-" * 70)

# =============================================================================
# INTERACTIVE AGENT LOOP
# =============================================================================

def build_task_prompt(user_query: str) -> str:
    """Build a complete task prompt from user query and RAG context."""
    task_parts = []

    # Add RAG context if available
    if rag_context.strip():
        task_parts.append(f"CONTEXT INFORMATION:\n{rag_context}\n")

    # Add user query
    task_parts.append(f"USER REQUEST:\n{user_query}\n")

    # Add instructions
    task_parts.append("""INSTRUCTIONS:
1. Understand what the user is asking for
2. Use appropriate tools to fulfill the request (document analysis, web research, etc.)
3. Cross-reference with the provided context information
4. Provide a comprehensive, structured response
5. MUST call final_answer(answer="your response") when complete

Remember your episodic memory mission: capture WHEN, WHERE, and WHY for memory events.""")

    return "\n".join(task_parts)


def run_agent_task(user_query: str):
    """Run the agent with a user query."""
    print("\n" + "=" * 70)
    print("[Alfred] Processing your request...")
    print("=" * 70 + "\n")

    try:
        # Build the task prompt
        task_prompt = build_task_prompt(user_query)

        # Run the agent
        result = agent.run(task_prompt)

        # Display result
        print("\n" + "=" * 70)
        print("[Alfred] Task completed!")
        print("=" * 70)
        print("\nRESPONSE:")
        print("-" * 70)
        print(result)
        print("-" * 70 + "\n")

        return True

    except Exception as e:
        print("\n" + "=" * 70)
        print("[Alfred] Error occurred:")
        print("=" * 70)
        print(f"{type(e).__name__}: {e}")
        print("-" * 70)
        print("\nTroubleshooting:")
        print("1. Ensure Ollama is running: ollama serve")
        print(f"2. Ensure {MODEL_ID} is installed: ollama pull {MODEL_ID}")
        print("3. Check your request and try again")
        print("-" * 70 + "\n")
        return False


# =============================================================================
# MAIN INTERACTIVE LOOP
# =============================================================================

print("\n" + "=" * 70)
print("  ALFRED - EPISODIC MEMORY ASSISTANT")
print("=" * 70)
print("\nHey there again! I'm Alfred, your Master Document Analyst")
print("and Research Specialist.")
print("\nI can help you with:")
print("  - Document analysis (PDF, CSV, images)")
print("  - Extracting key semantic elements")
print("  - Web research and source validation")
print("  - Creating episodic memories with context")
print("\nType 'exit' or 'quit' to end the session.")
print("=" * 70 + "\n")

# Interactive loop
while True:
    try:
        # Get user input
        user_input = input("How can I help you, legend? > ").strip()

        # Check for exit commands
        if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
            print("\n" + "=" * 70)
            print("Thanks for using Alfred! Your episodic memories are saved.")
            print("See you next time, legend!")
            print("=" * 70 + "\n")
            break

        # Skip empty input
        if not user_input:
            continue

        # Run the agent task
        run_agent_task(user_input)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("Session interrupted. Your memories are safe!")
        print("=" * 70 + "\n")
        break

    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Please try again or type 'exit' to quit.\n")
