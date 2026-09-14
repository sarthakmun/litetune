"""
LiteTune: High-Performance Large Language Model Fine-Tuning & Quantization Engine.
Engineered by Sarthak Mun (IIT Kharagpur) <sarthak.mun03@gmail.com>.
"""

__version__ = "1.0.0"
__author__ = "Sarthak Mun"
__email__ = "sarthak.mun03@gmail.com"

# Re-export core LitGPT components for drop-in high-performance usage
try:
    from litgpt.api import LLM
except (ImportError, ModuleNotFoundError):
    LLM = None
from litgpt.config import Config
from litgpt.model import GPT
from litgpt.prompts import PromptStyle
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    CheckpointValidationResult,
    estimate_model_memory,
    validate_checkpoint,
)

__all__ = [
    "LLM",
    "GPT",
    "Config",
    "PromptStyle",
    "Tokenizer",
    "CheckpointValidationResult",
    "validate_checkpoint",
    "estimate_model_memory",
    "__version__",
    "__author__",
    "__email__",
]
