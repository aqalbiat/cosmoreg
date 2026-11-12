
from pathlib import Path

import numpy as np


_ROOT = Path(__file__).resolve().parent


def _level_name(level):
    text = str(level)
    return text if text.startswith("level") else f"level{text}"


def _fold_name(fold):
    text = str(fold)
    return text if text.startswith("fold") else f"fold{text}"


def _load_leaf(root, level, group, data_file):
    leaf = _ROOT / root / _level_name(level) / group
    dataset_path = leaf / data_file
    labels_path = leaf / "labels.npy"
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)
    dataset = np.load(dataset_path, mmap_mode="r", allow_pickle=False)
    labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    return dataset, labels


def Get2dMapTrainFold(level, fold, root="maps2d"):
    return _load_leaf(root, level, _fold_name(fold), "maps.npy")


def Get2dMapTestData(level, root="maps2d"):
    return _load_leaf(root, level, "test", "maps.npy")


def GetSpectrumTrainFold(level, fold):
    return _load_leaf("powerspectrum", level, _fold_name(fold), "power_spectra.npy")


def GetSpectrumTestData(level):
    return _load_leaf("powerspectrum", level, "test", "power_spectra.npy")
