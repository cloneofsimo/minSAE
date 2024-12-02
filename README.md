# Minimal SAE Trainer

A principled implementation for training Sparse Autoencoders (SAEs) designed to extract interpretable features from Large Language Model (LLM) activations. This repository prioritizes implementation clarity and computational efficiency.

## Overview

- Efficient activation caching with sharded storage
- Rigorous validation set evaluation for Reconstruction loss.
- W&B Logging
- FineWeb dataset

The implementation is particularly suited for researchers studying mechanistic interpretability who require a robust yet maintainable codebase.

## Dependencies

```bash
pip install torch transformers wandb safetensors datasets click tqdm
```

## Usage

### Basic Training
Execute training with default parameters:

```bash
export HF_HF_HUB_ENABLE_HF_TRANSFER=1 # Enable HF transfer for faster fineweb download.

python sae.py \
    --model-name meta-llama/Llama-2-7b-hf \
    --layer-idx 8 \
    --cache-dir activation_cache \
    --d-hidden 2048 \
    --learning-rate 1e-3 \
    --batch-size 2 \
    --wandb-project test_sae_project \
    --num-train-samples 10000 \
    --num-val-samples 100 \
    --overwrite-cache
```

## Citation

```bibtex
@software{minSAE,
  author = {Simo Ryu},
  title = {Minimal SAE Trainer},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/cloneofsimo/minSAE}
}
```