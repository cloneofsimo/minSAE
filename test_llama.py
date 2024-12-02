import json
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_llama_hooks():
    """Test that we can properly hook LLama activations"""
    print("Testing LLama hooks...")

    # Load small model for testing
    model_name = "meta-llama/Llama-2-7b-hf"  # Change to a smaller model for testing
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Track activations
    activations = []

    def hook_fn(module, input, output):
        # Store MLP output activations
        activations.append(output.detach().cpu())

    # Register hook on MLP output
    layer_idx = 8
    hook_found = False
    for name, module in model.named_modules():
        if f"layers.{layer_idx}.mlp.down_proj" in name:
            module.register_forward_hook(hook_fn)
            hook_found = True
            break

    assert hook_found, f"Could not find layer {layer_idx} MLP"

    # Run a forward pass
    input_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        model(**inputs)

    print(f"Activations: {activations}")
    # Check activations were collected
    assert len(activations) > 0, "No activations collected"
    assert activations[0].ndim == 3, f"Wrong activation shape: {activations[0].shape}"
    print("LLama hook test passed!")


def test_activation_caching():
    """Test activation caching functionality with real LLama activations"""
    print("Testing activation caching...")

    # Create temporary cache directory
    tmp_dir = tempfile.mkdtemp()
    try:
        # Load model and tokenizer
        model_name = "meta-llama/Llama-2-7b-hf"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Sample texts
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "A journey of a thousand miles begins with a single step.",
            "To be or not to be, that is the question.",
        ]

        # Track activations
        activations = []

        def hook_fn(module, input, output):
            activations.append(output.detach().cpu())

        # Register hook on the MLP output
        layer_idx = 8
        for name, module in model.named_modules():
            if f"layers.{layer_idx}.mlp.down_proj" in name:
                module.register_forward_hook(hook_fn)
                break

        # Generate activations
        with torch.no_grad():
            for text in tqdm(texts, desc="Collecting activations"):
                inputs = tokenizer(text, return_tensors="pt").to(device)
                model(**inputs)

        # Save activations in shards
        shard_dir = Path(tmp_dir) / "train"
        shard_dir.mkdir(parents=True)

        for i, act in enumerate(activations):
            # Save shard
            shard_path = shard_dir / f"shard_{i:05d}.safetensors"
            save_file({"activations": act}, str(shard_path))

        # Save metadata
        metadata = {
            "model_name": model_name,
            "layer_idx": layer_idx,
            "activation_dim": activations[0].shape[-1],
            "num_shards": len(activations),
            "context_size": max(act.shape[1] for act in activations),
        }
        with open(shard_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Verify saved files
        assert (shard_dir / "metadata.json").exists(), "Metadata not saved"
        assert len(list(shard_dir.glob("shard_*.safetensors"))) == len(
            activations
        ), "Wrong number of shards"

        # Load and verify a shard
        shard_path = shard_dir / "shard_00000.safetensors"
        loaded_act = load_file(str(shard_path))["activations"]
        assert torch.allclose(
            loaded_act, activations[0]
        ), "Loaded activation doesn't match"

        # Verify activation properties
        d_model = activations[0].shape[-1]
        for act in activations:
            # Check dimensions
            assert act.ndim == 3, f"Wrong activation dimensions: {act.ndim}"
            assert (
                act.shape[-1] == d_model
            ), f"Inconsistent hidden dimension: {act.shape[-1]} vs {d_model}"

            # Check numerical properties
            assert not torch.isnan(act).any(), "NaN in activations"
            assert not torch.isinf(act).any(), "Inf in activations"
            assert torch.isfinite(act).all(), "Non-finite values in activations"

        print(
            f"Generated {len(activations)} activation shards of shape {activations[0].shape}"
        )

    finally:
        # Cleanup
        for path in Path(tmp_dir).rglob("*"):
            if path.is_file():
                path.unlink()

        shutil.rmtree(tmp_dir)

    print("Activation caching test passed!")


if __name__ == "__main__":
    print("\nRunning LLama and activation tests...")
    test_llama_hooks()
    test_activation_caching()
    print("\nAll tests passed! 🎉\n")
