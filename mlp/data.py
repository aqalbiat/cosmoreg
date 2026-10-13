# Load log_power_spectra.npy, labels.npy, and simulation_ids.npy.
# Combine training folds while keeping the validation fold and test group separate.
# Fit feature and target scaling on training data only; reuse it for validation/test.

from pathlib import Path
import numpy as np



def load_fold(spectra_root, level, fold):
    leaf = Path(spectra_root) / f"level{level}" / f"fold{fold}"

    X = np.load(leaf / "log_power_spectra.npy", allow_pickle=False)

    # Load the correct answers.
    y = np.load(leaf/"labels.npy", allow_pickle=False)

    # Load the simulation IDs.
    simulation_ids = np.load(leaf/"simulation_ids.npy", allow_pickle=False)

    assert X.shape[0] == y.shape[0] == simulation_ids.shape[0], \
        "Features, targets, and simulation IDs must have matching row counts."

    assert np.isfinite(X).all(), \
        "Features contain Nan or infinity."

    assert np.isfinite(y).all(), \
        "Targets contain NaN or infinity."

    # Return all three arrays.
    return X, y, simulation_ids


def load_train_validation(spectra_root, level, validation_fold):
    train_folds = []

    X_parts = []
    y_parts = []
    id_parts = []

    for fold in range (1, 6):
        # Append this fold only if it is NOT the validation fold.
        # Your code here.
        if fold != validation_fold :
            train_folds.append(fold)


    for fold in train_folds:
        X, y, simulation_ids = load_fold(spectra_root, level, fold)

        X_parts.append(X)
        y_parts.append(y)
        id_parts.append(simulation_ids)

    X_train = np.concatenate(X_parts, axis=0)
    y_train = np.concatenate(y_parts, axis=0)
    id_train = np.concatenate(id_parts, axis=0)

    X_val, y_val, id_val = load_fold(spectra_root, level, validation_fold)

    return X_train, y_train, id_train, X_val, y_val, id_val

def load_development(spectra_root, level):
    """Join all five development folds without loading the test group."""
    if level not in (1, 2, 3):
        raise ValueError("level must be 1, 2, or 3.")
    parts = []
    seen_ids = set()
    for fold in range(1, 6):
        X, y, ids = load_fold(spectra_root, level, fold)
        current_ids = set(np.unique(ids).tolist())
        if seen_ids.intersection(current_ids):
            raise ValueError("Simulation IDs overlap across development folds.")
        seen_ids.update(current_ids)
        parts.append((X, y, ids))
    return tuple(np.concatenate([part[i] for part in parts], axis=0) for i in range(3))


def fit_feature_scaler(X_train):
    feature_mean = X_train.mean(axis=0)

    # Your part: calculate the standard deviation of each column.
    feature_std = X_train.std(axis=0)

    # Avoid division by zero if a feature is constant.
    feature_std = np.where(feature_std == 0, 1.0, feature_std)

    return feature_mean, feature_std

def transform_features(X, feature_mean, feature_std):
    X_scaled = (X - feature_mean) / feature_std
    return X_scaled.astype(np.float32)

def fit_target_scaler(y_train):
    # Calculate the mean of each target column.
    target_mean = y_train.mean(axis=0)

    # Calculate the standard deviation of each target column.
    target_std = y_train.std(axis=0)

    target_std = np.where(target_std == 0, 1.0, target_std)

    return target_mean, target_std

def transform_targets(y, target_mean, target_std):
    y_scaled = (y-target_mean) / target_std
    return y_scaled.astype(np.float32)


def inverse_transform_targets(y_scaled, target_mean, target_std):
    return y_scaled * target_std + target_mean
