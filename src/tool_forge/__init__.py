"""Welcome to the Tool Forge Project

This project was created to demonstrate post training 
an LLM for tool calling functionality on a limited 12GB RTX 4070 GPU
"""

from .const import MODEL_NAME

__all__ = ["MODEL_NAME"]


def main() -> None:
    print("Hello from tool-forge!")