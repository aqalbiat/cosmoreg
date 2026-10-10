"""Run with: python -m mlp.check_data

Use python -i -m mlp.check_data to keep the variables available in a
Python console afterward. Sections execute top to bottom, like notebook cells.
"""

# %% Imports and settings
from pathlib import Path

from copy import deepcopy

import numpy as np
import torch

from mlp import data

SPECTRA_ROOT = Path(__file__).resolve().parents[1] / "powerspectrum"
LEVEL = 1
VALIDATION_FOLD = 5

# %% Load the training and validation folds
X_train, y_train, ids_train, X_val, y_val, ids_val = (
    data.load_train_validation(SPECTRA_ROOT, LEVEL, VALIDATION_FOLD)
)

assert X_train.ndim == X_val.ndim == 2
assert X_train.shape[1] == X_val.shape[1] == 50
assert y_train.shape == (X_train.shape[0], 2)
assert y_val.shape == (X_val.shape[0], 2)
assert ids_train.shape == (X_train.shape[0],)
assert ids_val.shape == (X_val.shape[0],)
assert np.intersect1d(ids_train, ids_val).size == 0, "Simulation leakage!"

print("Training shapes:", X_train.shape, y_train.shape, ids_train.shape)
print("Validation shapes:", X_val.shape, y_val.shape, ids_val.shape)
print("Unique simulations:", len(np.unique(ids_train)), "train,",
      len(np.unique(ids_val)), "validation")
print("Loading and split checks passed.")

# %% Fit feature scaling on training data and transform both groups
feature_mean, feature_std = data.fit_feature_scaler(X_train)
assert feature_mean.shape == feature_std.shape == (50,)
assert np.isfinite(feature_mean).all()
assert np.isfinite(feature_std).all()
assert np.all(feature_std > 0)

X_train_scaled = data.transform_features(X_train, feature_mean, feature_std)
X_val_scaled = data.transform_features(X_val, feature_mean, feature_std)

assert X_train_scaled.shape == X_train.shape
assert X_val_scaled.shape == X_val.shape
assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-5)
nonconstant_features = X_train.std(axis=0) > 0
assert np.allclose(
    X_train_scaled.std(axis=0)[nonconstant_features], 1, atol=1e-5
)
print("Feature scaling checks passed.")

# %% Fit target scaling on training data and transform both groups
target_mean, target_std = data.fit_target_scaler(y_train)
assert target_mean.shape == target_std.shape == (2,)
assert np.isfinite(target_mean).all()
assert np.isfinite(target_std).all()
assert np.all(target_std > 0)

y_train_scaled = data.transform_targets(y_train, target_mean, target_std)
y_val_scaled = data.transform_targets(y_val, target_mean, target_std)

assert y_train_scaled.shape == y_train.shape
assert y_val_scaled.shape == y_val.shape
assert np.allclose(y_train_scaled.mean(axis=0), 0, atol=1e-5)
nonconstant_targets = y_train.std(axis=0) > 0
assert np.allclose(
    y_train_scaled.std(axis=0)[nonconstant_targets], 1, atol=1e-5
)

y_restored = data.inverse_transform_targets(y_train_scaled, target_mean, target_std)
assert np.allclose(y_restored, y_train, rtol=1e-5, atol=1e-6)
print("Target scaling and inverse transformation checks passed.")

# %% Convert the scaled arrays to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train_scaled)
y_train_tensor = torch.from_numpy(y_train_scaled)
X_val_tensor = torch.from_numpy(X_val_scaled)
y_val_tensor = torch.from_numpy(y_val_scaled)

for tensor in (X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor):
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all().item()

print("Training tensors:", X_train_tensor.shape, y_train_tensor.shape)
print("Validation tensors:", X_val_tensor.shape, y_val_tensor.shape)
print("All data checks passed. No model training has been run.")

# %% Pair training inputs with targets
from torch.utils.data import TensorDataset

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

sample_X, sample_y = train_dataset[0]

print("Training dataset size:", len(train_dataset))
print("One input shape:", sample_X.shape)
print("One target shape:", sample_y.shape)

assert len(train_dataset) == X_train_tensor.shape[0]
assert torch.equal(sample_X, X_train_tensor[0])
assert torch.equal(sample_y, y_train_tensor[0])

# %% Group training samples into batches
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size= 64,
    shuffle= True,
)

batch_X, batch_y = next(iter(train_loader))

print("Input batch:", batch_X.shape)
print("Target batch:", batch_y.shape)

assert batch_X.shape == (64, 50)
assert batch_y.shape == (64, 2)

# %% Prepare validation batches
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
)

val_batch_X, val_batch_y = next(iter(val_loader))

print("Validation input batch:", val_batch_X.shape)
print("Validation target batch:", val_batch_y.shape)

assert val_batch_X.shape == (64, 50)
assert val_batch_y.shape == (64, 2)

# %% Check the model before training
from mlp.model import SpectrumMLP

model = SpectrumMLP()

with torch.no_grad():
    predictions = model(batch_X)

parameter_count = sum(p.numel() for p in model.parameters())

print(model)
print("Prediction shape:", predictions.shape)
print("Trainable parameters:", parameter_count)

assert predictions.shape == batch_y.shape
assert torch.isfinite(predictions).all().item()
assert parameter_count == 2194

# %% Calculate the initial loss
from torch import nn

loss_fn = nn.MSELoss()

# Run a fresh forward pass with gradient tracking enabled.
predictions = model(batch_X)

# Your part: compare predictions with the batch's targets.
loss = loss_fn(predictions, batch_y)

print("Initial batch MSE:", loss.item())

assert loss.ndim == 0
assert torch.isfinite(loss).item()
assert loss.requires_grad

# %% Calculate gradients without updating weights
model.zero_grad()

# Your part: call backward() on the loss.
loss.backward()


first_layer = model.network[0]
weight_gradients = first_layer.weight.grad

assert weight_gradients is not None
assert weight_gradients.shape == first_layer.weight.shape
assert torch.isfinite(weight_gradients).all().item()

print("First-layer gradient shape:", weight_gradients.shape)
print("Mean absolute gradient:", weight_gradients.abs().mean().item())

# %% Update the weights once
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)

# Keep a copy so we can check whether the weights changed.
weights_before = first_layer.weight.detach().clone()
loss_before = loss.item()

# Your part: ask the optimizer to perform one update.
optimizer.step()

weights_after = first_layer.weight.detach()
assert not torch.equal(weights_before, weights_after)

# Measure the loss again on exactly the same batch.
with torch.no_grad():
    predictions_after = model(batch_X)
    loss_after = loss_fn(predictions_after, batch_y).item()

print("Loss before update:", loss_before)
print("Loss after update:", loss_after)

# %% Train for one complete epoch
# model.train()
#
# total_squared_error = 0.0
# total_target_values = 0
#
# for batch_X, batch_y in train_loader:
#     optimizer.zero_grad()
#
#     predictions = model(batch_X)
#     loss = loss_fn(predictions, batch_y)
#
#     # Your part: calculate gradients.
#     loss.backward()
#     # Your part: update the parameters.
#     optimizer.step()
#
#     total_squared_error += loss.item() * batch_y.numel()
#     total_target_values += batch_y.numel()
#
# epoch_loss = total_squared_error / total_target_values
#
# print("Training MSE for one epoch:", epoch_loss)
#

best_val_loss = float("inf")
best_epoch = None
best_state = None

# %% Train a fresh model for multiple epochs
from mlp.train import train_one_epoch
from mlp.evaluate import evaluate_loss

torch.manual_seed(42)

# Start fresh so the earlier demonstration update doesn't carry over.
model = SpectrumMLP()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)

train_history = []
val_history = []

for epoch in range(1, 21):
    # Your part: call the training function.
    train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)

    # Your part: call the evaluation function on validation data.
    val_loss = evaluate_loss(model, val_loader, loss_fn)

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



model.load_state_dict(best_state)
val_loss = evaluate_loss(model, val_loader, loss_fn)

assert np.isclose(val_loss, best_val_loss), \
    "Restored model loss does not match the best recorded loss."

print("Restored best epoch:", best_epoch)
print("Best model validation MSE:", val_loss)

# %% Compare validation performance with a training-mean baseline
baseline_predictions = torch.zeros_like(y_val_tensor)

with torch.no_grad():
    # Your part: calculate MSE between baseline predictions and validation targets.
    baseline_loss = loss_fn(baseline_predictions, y_val_tensor)

print("Training-mean baseline MSE:", baseline_loss.item())
print("Model validation MSE:", val_loss)

if val_loss < baseline_loss.item():
    print("The model beats the training-mean baseline.")
else:
    print("The model does not yet beat the training-mean baseline.")

# %% Plot training and validation losses
import matplotlib.pyplot as plt

epochs = range(1, len(train_history) + 1)

fig, ax = plt.subplots()

ax.plot(epochs, train_history, label="Training")

# Your part: plot val_history against epochs, labelled "Validation".
ax.plot(epochs, val_history, label="Validation")

ax.axhline(
    baseline_loss.item(),
    color="gray",
    linestyle="--",
    label="Training-mean baseline",
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Standardized-target MSE")
ax.set_title("MLP learning curves")
ax.legend()

figure_path = Path(__file__).resolve().parents[1] / "mlp_learning_curves.png"
fig.savefig(figure_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print("Plot saved to:", figure_path)

# %% Evaluate each target in original units
model.eval()

with torch.no_grad():
    predictions_scaled = model(X_val_tensor).numpy()

predictions_original = data.inverse_transform_targets(
    predictions_scaled, target_mean, target_std
)

errors = predictions_original - y_val

# Your part: calculate RMSE separately for each target column.
rmse = np.sqrt(np.mean(errors**2, axis=0))

# Your part: calculate MAE separately for each target column.
mae = np.mean(np.abs(errors), axis=0)

target_names = ["Omega_m", "sigma_8"]

for i, name in enumerate(target_names):
    print(f"{name}: RMSE={rmse[i]:.4f}, MAE={mae[i]:.4f}")

# %% Compare each target with the training-mean baseline
baseline_errors = target_mean - y_val
baseline_rmse = np.sqrt(np.mean(baseline_errors ** 2, axis=0))

for i, name in enumerate(target_names):
    improvement = (
        100 * (1 - rmse[i] / baseline_rmse[i])
        if baseline_rmse[i] > 0 else float("nan")
    )
    print(
        f"{name}: baseline RMSE={baseline_rmse[i]:.4f}, "
        f"MLP RMSE={rmse[i]:.4f}, RMSE improvement={improvement:.1f}%"
    )

# %% Average map predictions within each validation simulation
from mlp.evaluate import average_by_simulation

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

# %% Evaluate simulation-level predictions in original units
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
