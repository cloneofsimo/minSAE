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
    --fineweb-version 10B \
    --cache-dir activation_cache \
    --d-hidden 2048 \
    --learning-rate 1e-3 \
    --batch-size 1024 \
    --wandb-project my-sae-project
```

### Advanced Configuration
For fine-grained control, the system exposes the following parameters:

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| model_name | HuggingFace model identifier | meta-llama/Llama-2-7b-hf | Any compatible transformer |
| layer_idx | Target layer for activation extraction | 8 | Zero-indexed |
| fineweb_version | Dataset scale selection | 10B | Options: 10B, 100B |
| d_hidden | Number of sparse features | 4x input | Automatically scaled if unspecified |
| batch_size | Training batch size | 1024 | Memory dependent |
| l1_coef | L1 regularization strength | 1e-3 | Controls sparsity |

### Programmatic Interface


## Dataset Integration

The implementation leverages the FineWeb dataset, providing:
- **10B Version**: Approximately 10 billion tokens of filtered web text
- **100B Version**: Expanded dataset with ~100 billion tokens
- Automatic train/validation splitting
- Efficient streaming interface

## Performance Metrics

The system tracks comprehensive metrics via Weights & Biases:
- Training and validation loss trajectories
- Reconstruction fidelity measurements
- Feature sparsity statistics
- Model checkpoint management

Example training progression:
```
Epoch 1
Training Loss: 0.2345 ± 0.0123
Validation Loss: 0.2123 ± 0.0098
Best Validation Loss: 0.2123

Epoch 10 
Training Loss: 0.0456 ± 0.0034
Validation Loss: 0.0445 ± 0.0028
Best Validation Loss: 0.0423
```

### Citation

```bibtex
@software{minSAE,
  author = {Simo Ryu},
  title = {Minimal SAE Trainer},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/cloneofsimo/minSAE}
}
```