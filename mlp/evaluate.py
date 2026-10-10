# Restore the trained model and training-fitted scaling statistics.
# Convert predictions back to original target units before computing metrics.
# Report per-target RMSE, MAE, and R2 at map and simulation levels.
# Average map predictions within each simulation for simulation-level evaluation.

import torch
import numpy as np

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
