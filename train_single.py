import os
import json
import argparse
import importlib
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Import data loader
from loaddata import load_level_data

SEED = 42

# Model registry
MODEL_VARIANTS = {
    4: 5,    # Danial
    6: 5,    # Danial
    8: 5,    # Danial
    10: 5,   # Danial
    12: 6,   # Alen
    14: 6,   # Alen
    16: 8,   # Alen
}

# "model_12_1" -> ("models.cnn12layer1", "CNN12Steps_Model1")
MODEL_REGISTRY = {
    f"model_{depth}_{i}": (f"models.cnn{depth}layer1", f"CNN{depth}Steps_Model{i}")
    for depth, n_variants in MODEL_VARIANTS.items()
    for i in range(1, n_variants + 1)
}

def get_model_class(model_name):
    """Return the model class for a registry name such as 'model_12_1'."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' not found in registry! "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    module_name, class_name = MODEL_REGISTRY[model_name]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise ImportError(
            f"Could not import '{module_name}' for {model_name}. "
            f"Expected file: {module_name.replace('.', '/')}.py"
        ) from e
    if not hasattr(module, class_name):
        raise ImportError(f"'{module_name}' has no class named '{class_name}'.")
    return getattr(module, class_name)



# Setup helpers
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Specific to Mac Apple Silicon (mps)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def get_device():
    # NVIDIA GPU (Colab / Kaggle) -> Apple Silicon -> CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



# Training
def train_one_model(model_name, level=1, epochs=25, lr=3e-4, results_file=None, save=True, data_root="maps2d"):
    """Train one model on one level and return its best validation MSE.

    results_file: JSON file the score is written to. Defaults to the shared
                  results_level{level}.json.
    save:         if False, nothing is written (no weights, no score). Use it
                  for timing or test runs.
    """
    model_class = get_model_class(model_name)

    set_seed()

    device = get_device()
    print("\n==========================================")
    print(f"Training: {model_name} on Level {level} using {device}")
    print(f"Data Root: {data_root}")
    print("==========================================")

    # Load Data
    X_train, Y_train, X_val, Y_val = load_level_data(level=level, val_fold=5, root=data_root)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32)),
        batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32)),
        batch_size=32, shuffle=False
    )

    # Initialize Model
    model = model_class().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # always holds the best epoch.
    weights_path = f"checkpoints/{model_name}_level{level}.pth"
    if save:
        os.makedirs("checkpoints", exist_ok=True)

    best_val_loss = float("inf")
    train_history = []
    val_history = []

    # Training Loop
    for epoch in range(1, epochs + 1):
        # TRAINING STEP
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # maybe bad
            optimizer.step()
            train_loss += loss.item() * bx.size(0)

        scheduler.step()
        epoch_train_loss = train_loss / len(train_loader.dataset)
        train_history.append(epoch_train_loss)

        # VALIDATION STEP
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(model(bx), by).item() * bx.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_history.append(epoch_val_loss)

        # SAVE WEIGHTS WHEN VALIDATION IMPROVES
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            if save:
                torch.save(model.state_dict(), weights_path)

        # PRINT BOTH SIDE-BY-SIDE
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs} - Train Loss: {epoch_train_loss:.5f} | Val Loss: {epoch_val_loss:.5f}")

    # Final Result
    print(f"\n-> Best Validation MSE for {model_name}: {best_val_loss:.6f}")
    if not save:
        print("-> Nothing saved (save=False)\n")
        return best_val_loss
    print(f"-> Best weights saved to: {weights_path}")

    # Save/Update Score in the JSON Leaderboard file
    if results_file is None:
        results_file = f"results_level{level}.json"
    scores = {}
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            scores = json.load(f)

    # Update or add this model's score
    scores[model_name] = best_val_loss

    with open(results_file, "w") as f:
        json.dump(scores, f, indent=4)
    print(f"-> Updated {results_file} successfully!\n")

    return best_val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Name of model to train (e.g. model_4_1 or model_12_1)")
    parser.add_argument("--level", type=int, default=1, help="Level to train on (1, 2, or 3)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--results-file", type=str, default=None,
                        help="JSON file for the score (default: results_level{level}.json)")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not write weights or scores (for timing or test runs)")
    parser.add_argument("--data-root", type=str, default="maps2d", help="Folder containing the data")
    args = parser.parse_args()

    train_one_model(args.model, level=args.level, epochs=args.epochs,
                    results_file=args.results_file, save=not args.no_save, data_root=args.data_root)