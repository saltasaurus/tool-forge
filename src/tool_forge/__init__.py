"""Welcome to the Tool Forge Project

This project was created to demonstrate post training 
an LLM for tool calling functionality on a limited 12GB RTX 4070 GPU
"""

from . import dataset, format
from .const import QWEN_4B_BASE, QWEN_4B_INSTRUCT

__all__ = ["QWEN_4B_BASE", "QWEN_4B_INSTRUCT", "dataset", "format"]


def main() -> None:
    print("Hello from tool-forge!")