"""
Unit tests for LiteTune API and core components.
"""
import pytest
import litetune
from litetune import (
    GPT,
    Config,
    PromptStyle,
    Tokenizer,
    estimate_model_memory,
)


def test_litetune_metadata():
    """Verify package version, author and email metadata."""
    assert litetune.__version__ == "1.0.0"
    assert litetune.__author__ == "Sarthak Mun"
    assert litetune.__email__ == "sarthak.mun03@gmail.com"


def test_config_initialization():
    """Test standard Config class initialization."""
    config = Config(
        name="custom-test-model",
        block_size=1024,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=4,
        n_head=4,
        n_embd=128,
        intermediate_size=512,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
    )
    assert config.name == "custom-test-model"
    assert config.block_size == 1024
    assert config.n_layer == 4
    assert config.n_embd == 128
    assert config.intermediate_size == 512


def test_prompt_style():
    """Test prompt formatting styles."""
    prompt_style = PromptStyle.from_name("alpaca")
    prompt_formatted = prompt_style.apply(
        prompt="Write a quicksort algorithm in Python.",
        input="List of integers"
    )
    assert "quicksort" in prompt_formatted
    assert "List of integers" in prompt_formatted


def test_estimate_model_memory():
    """Test memory estimation calculation for a configuration."""
    config = Config(
        name="test-model",
        block_size=512,
        vocab_size=1000,
        n_layer=2,
        n_head=2,
        n_embd=64,
        intermediate_size=256,
        rotary_percentage=1.0,
    )
    memory_estimate = estimate_model_memory(config)
    assert isinstance(memory_estimate, dict)
    assert len(memory_estimate) > 0
