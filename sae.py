import json
from pathlib import Path
from typing import Optional, Tuple

import click
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb


def cache_activations(
    model_name: str,
    layer_idx: int,
    output_dir: str,
    context_length: int = 2048,
    num_train_samples: Optional[int] = None,
    num_val_samples: Optional[int] = None,
    shard_size: int = 10000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: torch.dtype = torch.float16,
    split: str = "train",
) -> None:
    """Cache activations from a specific layer of the LLama model."""

    print(f"\nCaching activations from {model_name} layer {layer_idx}")
    print(f"Saving to: {output_dir}")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map={"": device}
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    d_model = model.config.hidden_size

    # Load FineWeb dataset
    dataset = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=False
    )

    if split == "train" and num_train_samples:
        dataset = dataset.take(num_train_samples)
    elif split == "validation" and num_val_samples:
        dataset = dataset.skip(num_train_samples).take(num_val_samples)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Storage for current shard
    current_activations = []
    shard_idx = 0
    samples_processed = 0
    total_tokens_processed = 0

    def save_shard(acts, idx: int, split: str = "train") -> None:
        """Save current activations as a shard."""
        if not acts:
            return

        shard_dir = output_dir / split
        shard_dir.mkdir(exist_ok=True)
        shard_path = shard_dir / f"shard_{idx:05d}.safetensors"

        # Stack and save
        acts_tensor = torch.cat(acts, dim=0)
        save_file({"activations": acts_tensor}, str(shard_path))
        return []

    def hook_fn(module, input, output):
        """Hook to capture activations."""
        act = output.detach().cpu()

        # Validate activation
        assert not torch.isnan(act).any(), "NaN in activation"
        assert not torch.isinf(act).any(), "Inf in activation"
        assert torch.isfinite(act).all(), "Non-finite values in activation"
        assert (
            act.shape[-1] == d_model
        ), f"Wrong activation dimension: {act.shape[-1]} vs {d_model}"

        current_activations.append(act)

    # Register hook on the MLP output
    hook_registered = False
    for name, module in model.named_modules():
        if f"layers.{layer_idx}.mlp.down_proj" in name:
            module.register_forward_hook(hook_fn)
            hook_registered = True
            break

    assert hook_registered, f"Could not find layer {layer_idx}"

    print("\nCollecting activations...")
    try:
        with torch.no_grad():
            for sample in tqdm(dataset):
                # Tokenize
                inputs = tokenizer(
                    sample["text"],
                    max_length=context_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Forward pass
                model(**inputs)

                # Update counters
                samples_processed += 1
                total_tokens_processed += inputs["input_ids"].numel()

                # Save shard if enough activations
                if len(current_activations) >= shard_size:
                    current_activations = save_shard(
                        current_activations, shard_idx, split
                    )
                    shard_idx += 1

                # Optional early stopping
                if (
                    split == "train"
                    and num_train_samples
                    and samples_processed >= num_train_samples
                ):
                    break
                elif (
                    split == "validation"
                    and num_val_samples
                    and samples_processed >= num_val_samples
                ):
                    break

    except KeyboardInterrupt:
        print("\nInterrupted! Saving final shard...")

    finally:
        # Save final shard if any activations remain
        if current_activations:
            save_shard(current_activations, shard_idx, split)

        # Save metadata
        metadata = {
            "model_name": model_name,
            "layer_idx": layer_idx,
            "corpus": "HuggingFaceFW/fineweb",
            "activation_dim": d_model,
            "num_shards": shard_idx + 1,
            "samples_processed": samples_processed,
            "total_tokens_processed": total_tokens_processed,
            "context_length": context_length,
            "shard_size": shard_size,
        }

        with open(output_dir / split / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"\nCaching complete!")
    print(f"Saved {shard_idx + 1} shards")
    print(f"Processed {samples_processed:,} samples")
    print(f"Total tokens: {total_tokens_processed:,}")


class ActivationLoader:
    """Simple activation data loader that loads and shuffles shards."""

    def __init__(
        self, cache_dir: str, split: str, batch_size: int, shuffle: bool = True
    ):
        self.cache_dir = Path(cache_dir) / split
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Load metadata
        with open(self.cache_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        self.shard_paths = sorted(self.cache_dir.glob("shard_*.safetensors"))
        self.current_shard = None
        self.current_indices = None

    def load_shard(self, shard_path: Path) -> torch.Tensor:
        """Load a single shard of activations."""
        data = load_file(str(shard_path))
        return data["activations"]

    def __iter__(self):
        # Shuffle shard order if requested
        if self.shuffle:
            np.random.shuffle(self.shard_paths)

        for shard_path in self.shard_paths:
            # Load and shuffle shard
            self.current_shard = self.load_shard(shard_path)
            self.current_indices = (
                torch.randperm(len(self.current_shard)) if self.shuffle else None
            )

            # Yield batches from current shard
            for start_idx in range(0, len(self.current_shard), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(self.current_shard))

                if self.current_indices is not None:
                    indices = self.current_indices[start_idx:end_idx]
                    batch = self.current_shard[indices]
                else:
                    batch = self.current_shard[start_idx:end_idx]

                yield batch


class SparseAutoencoder(nn.Module):
    """Sparse autoencoder using nn.Linear layers."""

    def __init__(self, d_input: int, d_hidden: int):
        super().__init__()
        self.encoder = nn.Linear(d_input, d_hidden)
        self.decoder = nn.Linear(d_hidden, d_input)
        self.activation = nn.ReLU()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.encoder(x))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        return self.decode(h), h


@click.command()
@click.option("--model-name", default="meta-llama/Llama-2-7b-hf", help="Model name")
@click.option("--layer-idx", default=8, help="Layer index")
@click.option("--cache-dir", default="activation_cache", help="Cache directory")
@click.option("--d-hidden", default=None, type=int, help="Hidden dimension")
@click.option(
    "--expansion-factor", default=4, help="Expansion factor if d_hidden not specified"
)
@click.option("--learning-rate", default=1e-3, help="Learning rate")
@click.option("--batch-size", default=1024, help="Batch size")
@click.option("--shard-size", default=10000, help="Activations per shard")
@click.option("--l1-coef", default=1e-3, help="L1 loss coefficient")
@click.option("--num-epochs", default=10, help="Number of epochs")
@click.option(
    "--num-train-samples", default=None, type=int, help="Number of training samples"
)
@click.option(
    "--num-val-samples", default=None, type=int, help="Number of validation samples"
)
@click.option("--wandb-project", default="sae-training", help="W&B project name")
@click.option("--wandb-run-name", default=None, help="W&B run name")
@click.option("--overwrite-cache", is_flag=True, help="Overwrite cache")
def train_sae(
    model_name: str,
    layer_idx: int,
    cache_dir: str,
    d_hidden: Optional[int],
    expansion_factor: int,
    learning_rate: float,
    batch_size: int,
    shard_size: int,
    l1_coef: float,
    num_epochs: int,
    num_train_samples: Optional[int],
    num_val_samples: Optional[int],
    wandb_project: str,
    wandb_run_name: str,
    overwrite_cache: bool = False,
) -> None:
    """Train a sparse autoencoder on LLama activations with validation."""

    # Initialize wandb
    config = {
        "model_name": model_name,
        "layer_idx": layer_idx,
        "d_hidden": d_hidden,
        "expansion_factor": expansion_factor,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "shard_size": shard_size,
        "l1_coef": l1_coef,
        "num_epochs": num_epochs,
        "num_train_samples": num_train_samples,
        "num_val_samples": num_val_samples,
    }
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(cache_dir)

    # Cache activations if not already cached
    if not (cache_dir / "train").exists() or overwrite_cache:
        cache_activations(
            model_name=model_name,
            layer_idx=layer_idx,
            output_dir=cache_dir,
            context_length=2048,
            shard_size=shard_size,
            num_train_samples=num_train_samples,
            num_val_samples=None,
            device=device,
            dtype=torch.bfloat16,
            split="train",
        )

    if not (cache_dir / "validation").exists() or overwrite_cache:
        cache_activations(
            model_name=model_name,
            layer_idx=layer_idx,
            output_dir=cache_dir,
            context_length=2048,
            shard_size=shard_size,
            num_train_samples=num_train_samples,
            num_val_samples=num_val_samples,
            device=device,
            dtype=torch.bfloat16,
            split="validation",
        )

    # Load metadata
    with open(cache_dir / "train" / "metadata.json") as f:
        metadata = json.load(f)

    # Initialize SAE
    d_input = metadata["activation_dim"]
    if d_hidden is None:
        d_hidden = d_input * expansion_factor

    print(f"Initializing SAE with input dim {d_input} and hidden dim {d_hidden}")
    sae = SparseAutoencoder(d_input, d_hidden).to(device)

    # Create data loaders
    train_loader = ActivationLoader(cache_dir, "train", batch_size=batch_size)
    val_loader = ActivationLoader(
        cache_dir, "validation", batch_size=batch_size, shuffle=False
    )

    # Training setup
    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    ctx = torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)

    def evaluate():
        sae.eval()
        total_recon_loss = 0.0
        total_l1_loss = 0.0
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                with ctx:
                    batch = batch.to(device, dtype=torch.bfloat16)
                    decoded, encoded = sae(batch)

                recon = mse_loss(decoded.float(), batch.float())
                l1 = l1_coef * encoded.abs().mean()
                loss = recon + l1

                total_recon_loss += recon.item()
                total_l1_loss += l1.item()
                total_loss += loss.item()
                num_batches += 1

        return {
            "val_loss": total_loss / num_batches,
            "val_recon_loss": total_recon_loss / num_batches,
            "val_l1_loss": total_l1_loss / num_batches,
        }

    # Training loop
    print("Starting training...")
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        sae.train()
        total_loss = 0.0
        recon_loss = 0.0
        l1_loss = 0.0
        batch_count = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()

            with ctx:
                batch = batch.to(device, dtype=torch.bfloat16)

                decoded, encoded = sae(batch)

            # Compute losses
            recon = mse_loss(decoded.float(), batch.float())
            l1 = l1_coef * encoded.abs().mean()
            loss = recon + l1

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            recon_loss += recon.item()
            l1_loss += l1.item()
            batch_count += 1

            # Log training metrics periodically
            if batch_count % 100 == 1:
                metrics = {
                    "train_loss": loss.item(),
                    "train_recon_loss": recon.item(),
                    "train_l1_loss": l1.item(),
                    "epoch": epoch,
                    "batch": batch_count,
                }
                wandb.log(metrics)

        # Evaluate on validation set
        val_metrics = evaluate()
        val_metrics["epoch"] = epoch
        wandb.log(val_metrics)

        # Save best model
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            save_path = cache_dir / "best_sae.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": sae.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": best_val_loss,
                    "config": config,
                },
                save_path,
            )

            # Log best model metrics
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_model_epoch"] = epoch

        # Print epoch statistics
        avg_train_loss = total_loss / batch_count
        print(f"Epoch {epoch+1}")
        print(f"Train - Loss: {avg_train_loss:.4f}")
        print(f"Val - Loss: {val_metrics['val_loss']:.4f}")
        print(f"Best val loss: {best_val_loss:.4f}")

    # Save final model
    final_save_path = cache_dir / "final_sae.pt"
    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": sae.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_loss": val_metrics["val_loss"],
            "config": config,
        },
        final_save_path,
    )

    print(f"Training complete!")
    print(f"Best model saved to {save_path}")
    print(f"Final model saved to {final_save_path}")

    # Close wandb run
    wandb.finish()


if __name__ == "__main__":
    train_sae()
