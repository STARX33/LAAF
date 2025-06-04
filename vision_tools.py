import os
from datetime import datetime
import csv
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from smolagents import tool

# Load BLIP model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)


def get_latest_image(folder_path="input_images/") -> str:
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        return None
    images.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=True)
    return os.path.join(folder_path, images[0])


@tool
def describe_image_with_blip() -> str:
    """
    Generates a caption for the most recent image in the 'input_images/' folder using the BLIP model.

    Returns:
        A string description of the image content.
    """
    image_path = get_latest_image()
    if not image_path:
        return "No image found."
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption


@tool
def detect_objects_from_latest_image() -> str:
    """
    Captions the most recent image and logs the description with a timestamp to 'results/inventory_log.csv'.

    Returns:
        The generated caption string.
    """
    caption = describe_image_with_blip()
    filename = "results/inventory_log.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as csvfile:
        fieldnames = ['timestamp', 'project', 'caption']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'timestamp': datetime.now().isoformat(),
            'project': "WarehouseVision",
            'caption': caption
        })
    return caption


@tool
def detect_items_with_owlvit() -> dict:
    """
    Placeholder for OWL-ViT detection functionality.

    Returns:
        A dictionary indicating this feature is not yet available.
    """
    return {"status": "OWL-ViT detection is currently disabled in local mode."}
