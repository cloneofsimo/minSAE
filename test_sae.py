import os

import torch
from safetensors.torch import save_file

from sae import ActivationLoader, SparseAutoencoder


def test_sae():
    """Test basic SAE functionality"""
    print("Testing SAE...")

    # Initialize SAE
    d_input = 768
    d_hidden = 1024
    batch_size = 32
    sae = SparseAutoencoder(d_input, d_hidden)

    # Test forward pass dimensions
    batch = torch.randn(batch_size, d_input)
    decoded, encoded = sae(batch)
    assert decoded.shape == (
        batch_size,
        d_input,
    ), f"Wrong decode shape: {decoded.shape}"
    assert encoded.shape == (
        batch_size,
        d_hidden,
    ), f"Wrong encode shape: {encoded.shape}"

    # Test sparsity
    assert torch.any(encoded == 0), "No zero activations found"
    sparsity = (encoded == 0).float().mean()
    assert sparsity > 0.1, f"Not sparse enough: {sparsity}"

    # Test gradients
    sae.zero_grad()
    loss = torch.nn.functional.mse_loss(decoded, batch)
    loss.backward()

    # Check gradients exist and are non-zero
    for name, param in sae.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    print("SAE tests passed!")


def test_data_loading():
    """Test activation loading functionality"""
    print("Testing data loading...")

    # Create temporary directory
    tmp_dir = "./tmp"

    # Create dummy data
    d_input = 768
    n_samples = 256
    data = torch.randn(n_samples, d_input)

    # Save dummy shard
    os.makedirs(os.path.join(tmp_dir, "train"), exist_ok=True)
    save_file(
        {"activations": data}, os.path.join(tmp_dir, "train", "shard_00000.safetensors")
    )

    # Save metadata
    with open(os.path.join(tmp_dir, "train", "metadata.json"), "w") as f:
        f.write('{"activation_dim": 768}')

    # Test loading without shuffle
    loader = ActivationLoader(tmp_dir, "train", batch_size=32, shuffle=False)
    loaded_data = torch.cat([batch for batch in loader])
    assert torch.allclose(loaded_data, data), "Data loading mismatch"

    # Test loading with shuffle
    loader = ActivationLoader(tmp_dir, "train", batch_size=32, shuffle=True)
    shuffled_data = torch.cat([batch for batch in loader])
    assert not torch.allclose(shuffled_data, data), "Data wasn't shuffled"

    # But should contain same values
    assert torch.allclose(
        torch.sort(shuffled_data.flatten())[0], torch.sort(data.flatten())[0]
    ), "Shuffled data values don't match"

    print("Data loading tests passed!")


def test_training():
    """Test basic training loop"""
    print("Testing training loop...")

    # Create dummy data and model
    d_input = 768
    d_hidden = 1024
    n_samples = 1000
    data = torch.randn(n_samples, d_input)
    sae = SparseAutoencoder(d_input, d_hidden)

    # Get initial loss
    with torch.no_grad():
        decoded, _ = sae(data)
        initial_loss = torch.nn.functional.mse_loss(decoded, data).item()

    # Train for a few steps
    optimizer = torch.optim.Adam(sae.parameters())
    for _ in range(10):
        optimizer.zero_grad()
        decoded, encoded = sae(data)
        loss = torch.nn.functional.mse_loss(decoded, data)
        loss.backward()
        optimizer.step()

    # Check final loss
    with torch.no_grad():
        decoded, _ = sae(data)
        final_loss = torch.nn.functional.mse_loss(decoded, data).item()

    assert (
        final_loss < initial_loss
    ), f"Loss didn't improve: {initial_loss} -> {final_loss}"
    print("Training tests passed!")


if __name__ == "__main__":
    print("\nRunning SAE tests...")
    test_sae()
    test_data_loading()
    test_training()
    print("\nAll tests passed! 🎉\n")
