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

### Choosing the Sparse Autoencoder Type

You can choose between two types of sparse autoencoders: ReLU and TopK. Use the `--sae-type` option to specify which one to use:

- **ReLU Autoencoder**: This is the default option. It uses a ReLU activation function and an L1 regularization term to encourage sparsity.
  
  ```bash
  python sae.py --sae-type relu
  ```

- **TopK Autoencoder**: This autoencoder selects the top K activations, setting the rest to zero, which can be specified with the `--topk` option.

  ```bash
  python sae.py --sae-type topk --topk 100
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