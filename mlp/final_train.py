"""Fit a fixed configuration on all development folds and save/reload it.

No validation checkpoint selection or test loading occurs here. Choose settings
and the epoch count from development experiments before calling this module.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mlp import data
from mlp.model import SpectrumMLP
from mlp.train import train_one_epoch


def train_final_model(
    spectra_root, level=1, *, hidden_dims=(64, 32), epochs=11,
    learning_rate=1e-3, weight_decay=1e-4, batch_size=64,
    seed=42, shuffle_seed=42,
):
    """Train a fresh model for a fixed duration on folds 1–5 only."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative.")

    X, y, ids = data.load_development(spectra_root, level)
    if X.shape != (len(ids), 50) or y.shape != (len(ids), 2):
        raise ValueError("Expected 50 input features and two targets per map.")

    # Fit new scalers using the complete development training set.
    feature_mean, feature_std = data.fit_feature_scaler(X)
    target_mean, target_std = data.fit_target_scaler(y)
    X_tensor = torch.from_numpy(data.transform_features(X, feature_mean, feature_std))
    y_tensor = torch.from_numpy(data.transform_targets(y, target_mean, target_std))
    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(shuffle_seed),
    )

    torch.manual_seed(seed)
    model = SpectrumMLP(hidden_dims=hidden_dims)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    loss_fn = nn.MSELoss()
    train_history = []
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, loss_fn)
        if not np.isfinite(loss):
            raise RuntimeError(f"Nonfinite training loss at epoch {epoch}.")
        train_history.append(loss)
        print(f"Epoch {epoch:02d}/{epochs} | Training MSE: {loss:.4f}")

    # Fixed-duration training keeps the final epoch, ready for later inference.
    model.eval()
    return {
        "model": model,
        "settings": {
            "level": level,
            "hidden_dims": list(hidden_dims),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "seed": seed,
            "shuffle_seed": shuffle_seed,
            "input_representation": "log_power_spectra.npy",
            "training_folds": [1, 2, 3, 4, 5],
        },
        "scalers": {
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "target_mean": target_mean,
            "target_std": target_std,
        },
        "target_names": ["Omega_m", "sigma_8"],
        "train_history": train_history,
        "training_maps": len(X),
        "training_simulation_ids": np.unique(ids).tolist(),
    }


def save_final_model(result, checkpoint_path):
    """Save weights, scalers, settings, and training history without overwriting."""
    path = Path(checkpoint_path)
    payload = {
        "format_version": 1,
        "torch_version": str(torch.__version__),
        "model_state": result["model"].state_dict(),
        "scalers": {
            name: torch.from_numpy(value.copy())
            for name, value in result["scalers"].items()
        },
        **{key: result[key] for key in (
            "settings", "target_names", "train_history",
            "training_maps", "training_simulation_ids",
        )},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        torch.save(payload, file)
    return path


def load_final_model(checkpoint_path):
    """Restore a CPU model and its saved development-fitted preprocessing."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["format_version"] != 1:
        raise ValueError("Unsupported final-model checkpoint format.")
    model = SpectrumMLP(hidden_dims=checkpoint["settings"]["hidden_dims"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return {
        "model": model,
        "scalers": {
            name: value.numpy().copy() for name, value in checkpoint["scalers"].items()
        },
        **{key: checkpoint[key] for key in (
            "settings", "target_names", "train_history",
            "training_maps", "training_simulation_ids",
        )},
    }
