"""Compatibility entrypoint for OpenAI CLIP audit."""
from model_audit import main

if __name__ == "__main__":
    main("openai/clip-vit-base-patch32", "OpenAI_CLIP")
