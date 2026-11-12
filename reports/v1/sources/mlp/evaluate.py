# Restore the trained model and training-fitted scaling statistics.
# Convert predictions back to original target units before computing metrics.
# Report per-target RMSE, MAE, and R2 at map and simulation levels.
# Average map predictions within each simulation for simulation-level evaluation.

import torch
import numpy as np
from pathlib import Path
from hashlib import sha256

def evaluate_loss(model, data_loader, loss_fn):
    model.eval()

    total_squared_error = 0.0
    total_target_values = 0

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            # Your part: run the model on the inputs.
            predictions = model(batch_X)

            # Your part: calculate the batch loss.
            batch_loss = loss_fn(predictions, batch_y)

            total_squared_error += batch_loss.item() * batch_y.numel()
            total_target_values += batch_y.numel()

    return total_squared_error / total_target_values

def average_by_simulation(predictions, targets, simulation_ids):
    unique_ids = np.unique(simulation_ids)

    simulation_predictions = []
    simulation_targets = []

    for simulation_id in unique_ids:
        mask = simulation_ids == simulation_id

        # Average the predictions from this simulation's maps.
        mean_prediction = predictions[mask].mean(axis=0)

        simulation_predictions.append(mean_prediction)

        # All maps from this simulation share the same target.
        simulation_targets.append(targets[mask][0])

    return (
        np.stack(simulation_predictions),
        np.stack(simulation_targets),
        unique_ids,
    )


def evaluate_test_model(checkpoint_path, spectra_root):
    """Evaluate one fixed checkpoint with its saved scalers; never train or fit.

    The checkpoint chooses the level. Baseline predictions use the saved
    development target means. Metrics follow [Omega_m, sigma_8] in original units.
    """
    from mlp import data
    from mlp.final_train import load_final_model

    checkpoint_path = Path(checkpoint_path)
    checkpoint_hash = sha256(checkpoint_path.read_bytes()).hexdigest()
    saved = load_final_model(checkpoint_path)
    level = saved["settings"]["level"]
    if saved["target_names"] != ["Omega_m", "sigma_8"]:
        raise ValueError("Unexpected target order in checkpoint.")
    if saved["settings"]["input_representation"] != "log_power_spectra.npy":
        raise ValueError("Expected saved log-power features.")

    leaf = Path(spectra_root) / f"level{level}" / "test"
    X = np.load(leaf / "log_power_spectra.npy", allow_pickle=False)
    y = np.load(leaf / "labels.npy", allow_pickle=False)
    ids = np.load(leaf / "simulation_ids.npy", allow_pickle=False)
    if ids.ndim != 1 or X.shape != (len(ids), 50) or y.shape != (len(ids), 2):
        raise ValueError("Invalid test array shapes.")
    if len(ids) == 0 or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Test data must be nonempty and finite.")
    if np.intersect1d(ids, saved["training_simulation_ids"]).size:
        raise ValueError("Test simulations overlap with final training simulations.")
    for simulation_id in np.unique(ids):
        targets = y[ids == simulation_id]
        if not np.all(targets == targets[0]):
            raise ValueError("Maps in one simulation must share the same targets.")

    scalers = saved["scalers"]
    X_scaled = data.transform_features(X, scalers["feature_mean"], scalers["feature_std"])
    y_scaled = data.transform_targets(y, scalers["target_mean"], scalers["target_std"])
    model = saved["model"]
    model.eval()
    weights_before = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        predictions_scaled = model(torch.from_numpy(X_scaled)).numpy()
    if not np.isfinite(predictions_scaled).all():
        raise RuntimeError("Model predictions are nonfinite.")
    predictions = data.inverse_transform_targets(
        predictions_scaled, scalers["target_mean"], scalers["target_std"],
    )
    sim_predictions, sim_targets, sim_ids = average_by_simulation(predictions, y, ids)

    def metrics(predicted, targets):
        errors = predicted - targets
        baseline_errors = scalers["target_mean"] - targets
        return {
            "rmse": np.sqrt(np.mean(errors ** 2, axis=0)),
            "mae": np.mean(np.abs(errors), axis=0),
            "baseline_rmse": np.sqrt(np.mean(baseline_errors ** 2, axis=0)),
            "baseline_mae": np.mean(np.abs(baseline_errors), axis=0),
        }

    assert all(torch.equal(value, weights_before[name]) for name, value in model.state_dict().items()), \
        "Model weights changed during evaluation."
    assert sha256(checkpoint_path.read_bytes()).hexdigest() == checkpoint_hash, \
        "Checkpoint file changed during evaluation."

    return {
        "level": level,
        "checkpoint_sha256": checkpoint_hash,
        "settings": saved["settings"],
        "target_names": saved["target_names"],
        "test_maps": len(X),
        "test_simulations": len(sim_ids),
        "test_loss": float(np.mean((predictions_scaled - y_scaled) ** 2)),
        "baseline_loss": float(np.mean(y_scaled ** 2)),
        "map_metrics": metrics(predictions, y),
        "simulation_metrics": metrics(sim_predictions, sim_targets),
        "map_predictions": predictions,
        "map_targets": y,
        "map_simulation_ids": ids,
        "simulation_predictions": sim_predictions,
        "simulation_targets": sim_targets,
        "simulation_ids": sim_ids,
        "baseline_target_mean": scalers["target_mean"],
    }
