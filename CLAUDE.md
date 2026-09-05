# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a minimal LLM (Large Language Model) implementation called "MyMiniMind" built with PyTorch and Hugging Face Transformers. The project focuses on implementing a transformer-based language model with optional Mixture of Experts (MoE) architecture.

## Key Files and Structure

- `model/model_myminimind.py`: Main model implementation containing:
  - `MyMiniMindConfig`: Configuration class extending PretrainedConfig
  - `RMSNorm`: Custom RMS normalization layer
  - `Attention`: Multi-head attention mechanism with RoPE and YaRN support
  - RoPE (Rotary Position Embedding) and YaRN (Yet another RoPE NeRd) implementations

- `pyproject.toml`: Project configuration with dependencies including:
  - Core ML: `transformers`, `torch`, `datasets`, `trl`
  - Utilities: `numpy`, `einops`, `wandb`, `swanlab`
  - Development: `matplotlib`, `peft` (optional)

## Development Environment

The project uses `uv` for package management with GPU support:

```bash
# Install dependencies
uv sync

# Install with GPU support
uv sync --extra gpu

# Install development dependencies
uv sync --extra dev
```

## Model Architecture

The model implements:
- Configurable hidden size and number of layers
- Optional Mixture of Experts (MoE) architecture
- Rotary Position Embeddings (RoPE) with YaRN length extrapolation
- RMS normalization
- Multi-head attention with grouped query attention support
- Flash attention optimization when available

## Key Technical Components

### RoPE and YaRN Implementation
The model includes custom implementations of:
- `precompute_cis()`: Pre-computes cosine and sine embeddings for RoPE/YaRN
- `apply_rotary_pos_emb()`: Applies rotary position embeddings to query and key tensors
- YaRN scaling for extended context length support

### Configuration Options
- Standard transformer parameters (hidden_size, num_layers, etc.)
- MoE-specific parameters (num_experts, num_experts_per_tok)
- RoPE scaling configuration for inference
- Flash attention toggle

## Common Development Tasks

### Testing RoPE Implementation
```bash
# Create test files to validate RoPE and YaRN functions
python -c "
import torch
import sys
sys.path.append('model')
from model_myminimind import precompute_cis, apply_rotary_pos_emb

# Test basic functionality
cos, sin = precompute_cis(64, 128)
print(f'RoPE shapes: cos={cos.shape}, sin={sin.shape}')
"
```

### Model Configuration
```python
from model.model_myminimind import MyMiniMindConfig

# Standard configuration
config = MyMiniMindConfig(hidden_size=768, num_hidden_layers=8)

# MoE configuration
config_moe = MyMiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=True)

# Extended context with YaRN
config_yarn = MyMiniMindConfig(
    hidden_size=768, 
    num_hidden_layers=8,
    inference_rope_scaling=True
)
```

## Important Notes

1. **Do not modify** the core model implementation in `model_myminimind.py` unless explicitly requested
2. The RoPE implementation includes both basic RoPE and YaRN length extrapolation
3. The model supports both standard transformer and MoE architectures
4. Flash attention is enabled by default when available
5. The project uses Chinese-friendly package index (Tsinghua University mirror)

## Dependencies

The project requires several key packages:
- PyTorch 2.6.0 (with GPU support recommended)
- Transformers 4.57.6
- Additional ML utilities (datasets, trl, etc.)

Use `uv sync --extra gpu` to install with GPU support for optimal performance.