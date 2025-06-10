from dotenv import load_dotenv
from smolagents import CodeAgent
from tools import suggest_menu
from vision_tools import (
    detect_objects_from_latest_image,
    detect_items_with_owlvit,
    describe_image_with_blip
)
from ollama_model import OllamaModel
from rag_loader import load_text_rag_context, load_image_rag_captions  # 👈 NEW

# Load .env if needed
load_dotenv()

# Load RAG context (text + image captions from RAG folders)
rag_text = load_text_rag_context()
rag_images = load_image_rag_captions()
rag_context = rag_text + rag_images

if not rag_context.strip():
    print("[RAG] ⚠️ No RAG context loaded. Proceeding with image-only prompt.")
else:
    print("[RAG] ✅ Context successfully loaded and injected into prompt.")

# Create Alfred agent with local Ollama model
agent = CodeAgent(
    tools=[
        suggest_menu,
        detect_objects_from_latest_image,
        detect_items_with_owlvit,
        describe_image_with_blip
    ],
    model=OllamaModel(model_id="llama3")
)

# Run Alfred with full RAG context + vision task
agent.run(f"""{rag_context}

Use the context above to decide what this image truly represents. Think beyond what you see. Is this a Dog?.""")
