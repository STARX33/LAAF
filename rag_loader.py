import os
from PIL import Image
import torch
from vision_tools import blip_processor, blip_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_text_rag_context(rag_path="rag/text"):
    """
    Scans all subfolders in rag/text and loads text content into a single context string.
    """
    context_blocks = []

    for folder in os.listdir(rag_path):
        folder_path = os.path.join(rag_path, folder)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if file.endswith((".txt", ".md", ".csv")):
                file_path = os.path.join(folder_path, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        context_blocks.append(f"\n\n---\n# From: {file}\n{content}")
                except Exception as e:
                    print(f"[RAG/Text] Error loading {file_path}: {e}")

    return "\n".join(context_blocks)


def load_image_rag_captions(rag_path="rag/images"):
    """
    Uses BLIP to caption images from rag/images/** and returns them as text blocks.
    """
    image_blocks = []

    for folder in os.listdir(rag_path):
        folder_path = os.path.join(rag_path, folder)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(folder_path, file)
                try:
                    image = Image.open(file_path).convert("RGB")
                    inputs = blip_processor(images=image, return_tensors="pt").to(device)
                    out = blip_model.generate(**inputs)
                    caption = blip_processor.decode(out[0], skip_special_tokens=True)
                    image_blocks.append(f"\n\n---\n# From image: {file}\n{caption}")
                except Exception as e:
                    print(f"[RAG/Image] Error processing {file_path}: {e}")

    return "\n".join(image_blocks)
