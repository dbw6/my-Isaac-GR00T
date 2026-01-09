# Focus integration for Qwen3 models
from .modeling_qwen3 import (
    Qwen3DecoderLayer_focus_forward,
    Qwen3Model_focus_forward,
    Qwen3Attention_focus_forward,
    Qwen3MLP_focus_forward,
)

__all__ = [
    "Qwen3DecoderLayer_focus_forward",
    "Qwen3Model_focus_forward",
    "Qwen3Attention_focus_forward",
    "Qwen3MLP_focus_forward",
]

