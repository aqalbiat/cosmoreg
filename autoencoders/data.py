"""Read existing splits without resampling or mixing simulations."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

TARGETS = ("Omega_m", "sigma_8")
FOLDS = tuple(f"fold{i}" for i in range(1, 6))


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass
class Group:
    name: str
    maps: np.ndarray
    labels: np.ndarray
    simulation_ids: np.ndarray
    map_ids: np.ndarray
    fingerprint: dict


def inspect_level(root, level):
    """Audit identities, including test IDs; do not open test maps or labels."""
    seen_simulations, seen_maps = set(), set()
    signature = {}
    for name in (*FOLDS, "test"):
        leaf = Path(root) / level / name
        sims = np.load(leaf / "simulation_ids.npy", allow_pickle=False)
        maps = np.load(leaf / "map_ids.npy", allow_pickle=False)
        if sims.ndim != 1 or maps.ndim != 1 or len(sims) != len(maps) or not len(sims):
            raise ValueError(f"Invalid identities: {leaf}")
        if not np.issubdtype(sims.dtype, np.integer) or not np.issubdtype(maps.dtype, np.integer):
            raise ValueError(f"IDs must be integers: {leaf}")
        current_sims, current_maps = set(sims.tolist()), set(maps.tolist())
        if len(current_maps) != len(maps) or seen_maps & current_maps or seen_simulations & current_sims:
            raise ValueError(f"Map or simulation leakage: {leaf}")
        seen_simulations.update(current_sims)
        seen_maps.update(current_maps)
        # Hash every input, including map contents, so resume cannot use stale codes.
        signature[name] = {file: digest(leaf / file) for file in
                           ("maps.npy", "labels.npy", "simulation_ids.npy", "map_ids.npy")}
    return signature


def load_group(root, level, name, cache, fingerprint):
    leaf = Path(root) / level / name
    labels = np.load(leaf / "labels.npy", allow_pickle=False)
    sims = np.load(leaf / "simulation_ids.npy", allow_pickle=False)
    ids = np.load(leaf / "map_ids.npy", allow_pickle=False)
    if labels.shape != (len(ids), 2) or not np.isfinite(labels).all():
        raise ValueError(f"Invalid labels: {leaf}")
    for sim in np.unique(sims):
        if not np.all(labels[sims == sim] == labels[sims == sim][0]):
            raise ValueError(f"Inconsistent labels for simulation {sim}")
    cache = Path(cache) / level / name
    cache.mkdir(parents=True, exist_ok=True)
    marker = cache / "source.json"
    destination = cache / "density_contrast.npy"
    if not (marker.exists() and destination.exists() and
            json.loads(marker.read_text()) == fingerprint):
        raw = np.load(leaf / "maps.npy", mmap_mode="r", allow_pickle=False)
        if raw.shape != (len(ids), 256, 256):
            raise ValueError(f"Expected [N,256,256] maps: {leaf}")
        temporary = destination.with_suffix(".partial.npy")
        delta = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=raw.shape)
        for start in range(0, len(raw), 32):
            batch = np.asarray(raw[start:start + 32], dtype=np.float64)
            means = batch.mean(axis=(1, 2), keepdims=True)
            if not np.isfinite(batch).all() or np.any(means <= 0):
                raise ValueError(f"Nonfinite maps or nonpositive map mean: {leaf}")
            delta[start:start + len(batch)] = batch / means - 1.0
        delta.flush()
        del delta
        temporary.replace(destination)
        write_json(marker, fingerprint)
    maps = np.load(destination, mmap_mode="r", allow_pickle=False)
    return Group(name, maps, labels.astype(np.float64), sims, ids, fingerprint)


def fit_map_scaler(groups):
    total, squared, count = 0.0, 0.0, 0
    for group in groups:
        for start in range(0, len(group.maps), 32):
            batch = np.asarray(group.maps[start:start + 32], dtype=np.float64)
            total += batch.sum()
            squared += np.square(batch).sum()
            count += batch.size
    mean = total / count
    scale = float(np.sqrt(max(squared / count - mean * mean, 0)))
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("Training maps have no finite variance")
    return {"mean": mean, "scale": scale}


class Maps(Dataset):
    def __init__(self, groups, scaler):
        self.groups = groups
        self.ends = np.cumsum([len(group.maps) for group in groups])
        self.scaler = scaler

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        group_index = int(np.searchsorted(self.ends, index, side="right"))
        offset = 0 if group_index == 0 else self.ends[group_index - 1]
        sample = np.array(self.groups[group_index].maps[index - offset], copy=True)
        sample -= self.scaler["mean"]
        sample /= self.scaler["scale"]
        return torch.from_numpy(sample[None])


def metadata(groups):
    return {name: np.concatenate([getattr(group, name) for group in groups])
            for name in ("labels", "simulation_ids", "map_ids")}
