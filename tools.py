# This file originated from the Hugging Face Agents course.
# Functionality like `suggest_menu` was part of the original Alfred agent demo.
# LAAF builds upon that foundation with local RAG, vision tools, and custom runtime.

from smolagents import tool

@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggests a menu based on the type of occasion.

    Args:
        occasion: The type of event, such as 'casual', 'formal', or 'superhero'.

    Returns:
        A string description of the suggested menu.
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    elif occasion == "formal":
        return "Three-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu curated by Alfred the butler."
