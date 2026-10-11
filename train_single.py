import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random
import numpy as np


# Import your data loader
from loaddata import load_level_data

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Specific to Mac Apple Silicon (mps)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)

# Import all your model classes here
from models.cnn12layer1 import CNN12Steps_Model1
from models.cnn12layer1 import CNN12Steps_Model2
from models.cnn12layer1 import CNN12Steps_Model3
from models.cnn12layer1 import CNN12Steps_Model4
from models.cnn12layer1 import CNN12Steps_Model5
from models.cnn12layer1 import CNN12Steps_Model6
from models.cnn14layer1 import CNN14Steps_Model1
from models.cnn14layer1 import CNN14Steps_Model2
from models.cnn14layer1 import CNN14Steps_Model3
from models.cnn14layer1 import CNN14Steps_Model4
from models.cnn14layer1 import CNN14Steps_Model5
from models.cnn14layer1 import CNN14Steps_Model6
from models.cnn16layer1 import CNN16Steps_Model1
from models.cnn16layer1 import CNN16Steps_Model2
from models.cnn16layer1 import CNN16Steps_Model3
from models.cnn16layer1 import CNN16Steps_Model4
from models.cnn16layer1 import CNN16Steps_Model5
from models.cnn16layer1 import CNN16Steps_Model6
from models.cnn16layer1 import CNN16Steps_Model7
from models.cnn16layer1 import CNN16Steps_Model8

# Register all your available models in a dictionary
MODEL_REGISTRY = {
    "model_12_1": CNN12Steps_Model1,
    "model_12_2": CNN12Steps_Model2,
    "model_12_3": CNN12Steps_Model3,
    "model_12_4": CNN12Steps_Model4,
    "model_12_5": CNN12Steps_Model5,
    "model_12_6": CNN12Steps_Model6,
    "model_14_1": CNN14Steps_Model1,
    "model_14_2": CNN14Steps_Model2,
    "model_14_3": CNN14Steps_Model3,
    "model_14_4": CNN14Steps_Model4,
    "model_14_5": CNN14Steps_Model5,
    "model_14_6": CNN14Steps_Model6,
    "model_16_1": CNN16Steps_Model1,
    "model_16_2": CNN16Steps_Model2,
    "model_16_3": CNN16Steps_Model3,
    "model_16_4": CNN16Steps_Model4,
    "model_16_5": CNN16Steps_Model5,     
    "model_16_6": CNN16Steps_Model6,
    "model_16_7": CNN16Steps_Model7,
    "model_16_8": CNN16Steps_Model8,    
    # Add your other models here as you write them!
}

def train_one_model(model_name, level=1, epochs=25, lr=3e-4):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry! Available: {list(MODEL_REGISTRY.keys())}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n==========================================")
    print(f"Training: {model_name} on Level {level} using {device}")
    print(f"==========================================")

    # 1. Load Data
    X_train, Y_train, X_val, Y_val = load_level_data(level=level, val_fold=5)
    
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32)),
        batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32)),
        batch_size=32, shuffle=False
    )

    # 2. Initialize Model
    model = MODEL_REGISTRY[model_name]().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)


    train_history = []
    val_history = []

    best_val_loss = float('inf')  # Starts at infinity
    train_history = []
    val_history = []

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        # --- A. TRAINING STEP ---
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #maybe bad
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
        
        scheduler.step()
        epoch_train_loss = train_loss / len(train_loader.dataset)
        train_history.append(epoch_train_loss)

        # --- B. VALIDATION TRACKER (Inside the loop!) ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(model(bx), by).item() * bx.size(0)
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_history.append(epoch_val_loss)

        #___________________
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), f"checkpoints/{model_name}_level1.pth")
        #________________
        
        # --- C. PRINT BOTH SIDE-BY-SIDE ---
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs} - Train Loss: {epoch_train_loss:.5f} | Val Loss: {epoch_val_loss:.5f}")

    # 4. Final Result
    #final_val_mse = val_history[-1]
    print(f"\n-> Best Validation MSE for {model_name}: {best_val_loss:.6f}")


    # 5. Save Model Weights
    os.makedirs("checkpoints", exist_ok=True)
    weights_path = f"checkpoints/{model_name}_level{level}.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"-> Saved weights to: {weights_path}")

    # 6. Save/Update Score in the JSON Leaderboard file
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Name of model to train (e.g. model_1)")
    parser.add_argument("--level", type=int, default=1, help="Level to train on (1, 2, or 3)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    args = parser.parse_args()

    train_one_model(args.model, level=args.level, epochs=args.epochs)