"""Compatibility entrypoint for OpenCLIP audit."""
from model_audit import main

if __name__ == "__main__":
    main("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", "OpenCLIP")
