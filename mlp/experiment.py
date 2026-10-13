"""Build a reusable experiment one small learning task at a time."""

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from mlp import data
from mlp.model import SpectrumMLP
from mlp.train import train_one_epoch
from mlp.evaluate import evaluate_loss, average_by_simulation

from torch.utils.data import TensorDataset, DataLoader

def run_experiment(
    spectra_root, level, validation_fold, *, hidden_dims=(32, 16),
    learning_rate=1e-3, weight_decay=1e-4, batch_size=64, epochs=20,
    seed=42, shuffle_seed=None,
):
    """Train one split and return its best model, scalers, histories, and metrics.

    Metric arrays use target order [Omega_m, sigma_8] and original units.
    Validation and baseline losses are standardized-target MSE. The returned
    model has the best validation weights restored; nothing is saved to disk.
    Defaults preserve the original experiment. Set shuffle_seed to use a
    separate batch-order generator for fair comparisons across architectures.
    """
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative.")
    X_train, y_train, ids_train, X_val, y_val, ids_val = (
        data.load_train_validation(
            spectra_root, level, validation_fold
        )
    )

    # Keep related maps from one simulation in the same split.
    assert np.intersect1d(ids_train, ids_val).size == 0, (
        "Training and validation share ID's"
    )

    # Fit feature scaling on training rows and reuse it for validation.
    feature_mean, feature_std = data.fit_feature_scaler(X_train)

    X_train_scaled = data.transform_features(X_train, feature_mean, feature_std)
    X_val_scaled = data.transform_features(X_val, feature_mean, feature_std)

    # Fit target scaling on training rows and reuse it for validation.
    target_mean, target_std = data.fit_target_scaler(y_train)

    y_train_scaled = data.transform_targets(y_train, target_mean, target_std)
    y_val_scaled = data.transform_targets(y_val, target_mean, target_std)

    # Convert scaled arrays to float32 tensors for the model.
    X_train_tensor = torch.from_numpy(X_train_scaled)
    y_train_tensor = torch.from_numpy(y_train_scaled)

    X_val_tensor = torch.from_numpy(X_val_scaled)
    y_val_tensor = torch.from_numpy(y_val_scaled)

    # Keep each spectrum paired with its target.
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # Shuffle training batches; preserve validation row order.
    shuffle_generator = None
    if shuffle_seed is not None:
        shuffle_generator = torch.Generator().manual_seed(shuffle_seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        generator=shuffle_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Start each experiment with fresh weights and optimizer state.
    torch.manual_seed(seed)
    model = SpectrumMLP(hidden_dims=hidden_dims)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Record training and validation losses for each epoch.
    train_history = []
    val_history = []

    best_val_loss = float("inf")
    best_epoch = None
    best_state = None


    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss = evaluate_loss(model, val_loader, loss_fn)

        # Keep an independent copy of the best validation checkpoint.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

        train_history.append(train_loss)
        val_history.append(val_loss)

        print(
            f"Epoch {epoch:02d} | "
            f"Train MSE: {train_loss:.4f} | "
            f"Validation MSE: {val_loss:.4f}"
        )

    # Restore the best weights and verify their validation loss.
    model.load_state_dict(best_state)
    val_loss = evaluate_loss(model, val_loader, loss_fn)

    assert np.isclose(val_loss, best_val_loss), \
        "Restored model loss does not match the best recorded loss."

    print("Restored best epoch:", best_epoch)
    print("Best model validation MSE:", val_loss)

    # Training target means correspond to zeros in standardized space.
    baseline_predictions = torch.zeros_like(y_val_tensor)
    baseline_loss = loss_fn(baseline_predictions, y_val_tensor)

    print("Training-mean baseline MSE:", baseline_loss.item())
    print("Model validation MSE:", val_loss)

    if val_loss < baseline_loss.item():
        print("The model beats the training-mean baseline.")
    else:
        print("The model does not beat the training-mean baseline.")

    # Notebook section 15. Measure map-level error in original units
    model.eval()

    with torch.no_grad():
        predictions_scaled = model(X_val_tensor).numpy()

    predictions_original = data.inverse_transform_targets(
        predictions_scaled, target_mean, target_std
    )

    errors = predictions_original - y_val

    rmse = np.sqrt(np.mean(errors**2, axis=0))

    mae = np.mean(np.abs(errors), axis=0)

    target_names = ["Omega_m", "sigma_8"]

    for i, name in enumerate(target_names):
        print(f"{name}: RMSE={rmse[i]:.4f}, MAE={mae[i]:.4f}")

    # Notebook section 16. Compare map-level RMSE with the baseline
    baseline_errors = target_mean - y_val
    baseline_rmse = np.sqrt(np.mean(baseline_errors ** 2, axis=0))
    baseline_mae = np.mean(np.abs(baseline_errors), axis=0)

    for i, name in enumerate(target_names):
        improvement = (
            100 * (1 - rmse[i] / baseline_rmse[i])
            if baseline_rmse[i] > 0 else float("nan")
        )
        print(
            f"{name}: baseline RMSE={baseline_rmse[i]:.4f}, "
            f"MLP RMSE={rmse[i]:.4f}, RMSE improvement={improvement:.1f}%"
        )

    # Notebook section 17. Average predictions within each simulation
    sim_predictions, sim_targets, sim_ids = average_by_simulation(
        predictions_original, y_val, ids_val
    )

    assert sim_predictions.shape == (len(np.unique(ids_val)), 2)
    assert sim_targets.shape == sim_predictions.shape
    assert sim_ids.shape == (len(sim_predictions),)
    assert np.isfinite(sim_predictions).all()

    for i, simulation_id in enumerate(sim_ids):
        mask = ids_val == simulation_id
        assert np.allclose(sim_predictions[i], predictions_original[mask].mean(axis=0))
        assert np.all(y_val[mask] == sim_targets[i])

    print("Simulation prediction shape:", sim_predictions.shape)
    print("Simulation target shape:", sim_targets.shape)
    print("Simulation averaging checks passed.")

    # Notebook section 18. Measure simulation-level performance
    sim_errors = sim_predictions - sim_targets
    sim_rmse = np.sqrt(np.mean(sim_errors ** 2, axis=0))
    sim_mae = np.mean(np.abs(sim_errors), axis=0)

    sim_baseline_errors = target_mean - sim_targets
    sim_baseline_rmse = np.sqrt(np.mean(sim_baseline_errors ** 2, axis=0))
    sim_baseline_mae = np.mean(np.abs(sim_baseline_errors), axis=0)

    assert sim_rmse.shape == sim_mae.shape == (2,)
    assert np.isfinite(sim_rmse).all()
    assert np.isfinite(sim_mae).all()
    # With equal map counts per simulation, averaging cannot increase squared error.
    _, maps_per_simulation = np.unique(ids_val, return_counts=True)
    if np.all(maps_per_simulation == maps_per_simulation[0]):
        assert np.all(sim_rmse <= rmse + 1e-6)

    print("Simulation-level validation metrics (original units):")
    for i, name in enumerate(target_names):
        improvement = (
            100 * (1 - sim_rmse[i] / sim_baseline_rmse[i])
            if sim_baseline_rmse[i] > 0 else float("nan")
        )
        print(
            f"{name}: MLP RMSE={sim_rmse[i]:.4f}, MAE={sim_mae[i]:.4f}; "
            f"baseline RMSE={sim_baseline_rmse[i]:.4f}, "
            f"MAE={sim_baseline_mae[i]:.4f}; "
            f"RMSE improvement={improvement:.1f}%"
        )

    # Return values so callers can collect folds and plot their histories.
    return {
        "level": level,
        "validation_fold": validation_fold,
        "settings": {
            "hidden_dims": list(hidden_dims),
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "epochs": epochs,
            "seed": seed,
            "shuffle_seed": shuffle_seed,
        },
        "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "validation_simulations": len(sim_ids),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "baseline_loss": baseline_loss.item(),
        "train_history": train_history,
        "val_history": val_history,
        "target_names": target_names,
        "map_metrics": {
            "rmse": rmse,
            "mae": mae,
            "baseline_rmse": baseline_rmse,
            "baseline_mae": baseline_mae,
        },
        "simulation_metrics": {
            "rmse": sim_rmse,
            "mae": sim_mae,
            "baseline_rmse": sim_baseline_rmse,
            "baseline_mae": sim_baseline_mae,
        },
        "model": model,
        "scalers": {
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "target_mean": target_mean,
            "target_std": target_std,
        },
    }
