"""
Final training of the two best CNNs on all 5 folds of level 3.

    model_10_5  best of the 4-10 step CNNs (Danial)
    model_16_5  best of the 12-16 step CNNs (Alen)

Both were chosen on validation scores (run_levels.py, level 3, fold 5). This
follows the protocol of the Ridge and MLP branches: retrain the chosen model
on all 5 development folds, then evaluate it once on the held-out test group
(cnn_test/cnn_test.ipynb).

Training uses exactly the settings of train_single.py (AdamW, lr 3e-4,
weight decay 1e-4, cosine schedule, batch 32, gradient clipping, seed 42)
for the same 25 epochs. With no validation fold left, the weights after the
last epoch are kept.

Usage (from the repo root):

    python final_train_cnn.py                      # both models
    python final_train_cnn.py --models model_16_5  # only one (e.g. after a disconnect)

Each model is saved to checkpoints/final/{model}_level3.pt together with the
input normalisation and the training-label mean, so the test notebook needs
nothing else. A model that already has a final checkpoint is skipped unless
--force is given.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train_single import get_device, get_model_class, set_seed

LEVEL = 3
FINAL_MODELS = ["model_10_5", "model_16_5"]
FOLDS = [1, 2, 3, 4, 5]
FINAL_DIR = Path("checkpoints/final")
SETTINGS = {"epochs": 25, "lr": 3e-4, "weight_decay": 1e-4, "batch_size": 32,
            "max_grad_norm": 1.0, "seed": 42, "training_folds": FOLDS}


def final_path(model_name, level=LEVEL):
    return FINAL_DIR / f"{model_name}_level{level}.pt"


def log_maps(leaf):
    """Same transform as loaddata.load_level_data: float32, log10, channel axis."""
    X = np.array(np.load(Path(leaf) / "maps.npy", mmap_mode="r"), dtype=np.float32)
    X = np.log10(np.maximum(X, 1e-5))
    return np.expand_dims(X, axis=1)


def load_all_folds(level=LEVEL, data_root="maps2d"):
    """All 5 development folds, log-transformed and standardised like loaddata.py."""
    leaves = [Path(data_root) / f"level{level}" / f"fold{f}" for f in FOLDS]
    X = np.concatenate([log_maps(leaf) for leaf in leaves], axis=0)
    y = np.concatenate([np.load(leaf / "labels.npy") for leaf in leaves]).astype(np.float32)
    sim_ids = np.concatenate([np.load(leaf / "simulation_ids.npy") for leaf in leaves])
    mean, std = float(np.mean(X)), float(np.std(X))
    X -= mean          # in place, to keep memory low
    X /= std
    return X, y, sim_ids, mean, std


def train_final(model_name, data, level=LEVEL, epochs=SETTINGS["epochs"]):
    X, y, sim_ids, mean, std = data
    s = dict(SETTINGS, epochs=epochs)

    set_seed(s["seed"])
    device = get_device()
    print(f"\nFinal training: {model_name} on level {level}, {len(y)} maps from "
          f"{len(np.unique(sim_ids))} simulations, {s['epochs']} epochs on {device}")

    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                        batch_size=s["batch_size"], shuffle=True)
    model = get_model_class(model_name)().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=s["lr"], weight_decay=s["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=s["epochs"], eta_min=1e-5)

    start = time.time()
    for epoch in range(1, s["epochs"] + 1):
        model.train()
        total = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=s["max_grad_norm"])
            optimizer.step()
            total += loss.item() * bx.size(0)
        scheduler.step()
        if epoch % 5 == 0 or epoch == s["epochs"]:
            print(f"Epoch {epoch:02d}/{s['epochs']} - Train Loss: {total / len(y):.5f}")

    path = final_path(model_name, level)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_name": model_name,
        "level": level,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "input_mean": mean,
        "input_std": std,
        "target_mean": y.mean(axis=0).tolist(),
        "training_simulation_ids": np.unique(sim_ids).tolist(),
        "training_maps": int(len(y)),
        "settings": s,
        "target_names": ["Omega_m", "sigma_8"],
    }, path)
    print(f"-> Saved {path} ({(time.time() - start) / 60:.1f} min)")


def main():
    parser = argparse.ArgumentParser(description="Train the final CNNs on all 5 folds of level 3.")
    parser.add_argument("--models", nargs="+", default=FINAL_MODELS, choices=FINAL_MODELS,
                        help=f"Which of the final models to train (default: both)")
    parser.add_argument("--force", action="store_true", help="Retrain even if a final checkpoint exists")
    args = parser.parse_args()

    todo = [m for m in args.models if args.force or not final_path(m).exists()]
    for m in args.models:
        if m not in todo:
            print(f"Skipping {m}: {final_path(m)} already exists (use --force to retrain)")
    if not todo:
        return

    print(f"Loading all 5 folds of level {LEVEL}...")
    data = load_all_folds()
    for name in todo:
        train_final(name, data)


if __name__ == "__main__":
    main()
