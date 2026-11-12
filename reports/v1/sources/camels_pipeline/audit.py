from __future__ import annotations

from pathlib import Path

import numpy as np


def load_array(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def audit_outputs(maps_root: Path, spectra_root: Path) -> dict[str, dict[str, int]]:
    results: dict[str, dict[str, int]] = {}
    for maps_leaf in sorted(path for path in maps_root.glob("*/**") if path.is_dir()):
        rel = maps_leaf.relative_to(maps_root)
        spectra_leaf = spectra_root / rel
        if not (maps_leaf / "maps.npy").exists():
            continue
        required = ["labels.npy", "simulation_ids.npy", "map_ids.npy"]
        for name in required:
            if not (maps_leaf / name).exists() or not (spectra_leaf / name).exists():
                raise FileNotFoundError(f"Missing paired {name} in {rel}.")
            left = load_array(maps_leaf / name)
            right = load_array(spectra_leaf / name)
            if not np.array_equal(left, right):
                raise ValueError(f"Paired row mismatch for {name} in {rel}.")
        maps = np.load(maps_leaf / "maps.npy", mmap_mode="r", allow_pickle=False)
        power = load_array(spectra_leaf / "power_spectra.npy")
        if maps.shape[0] != power.shape[0]:
            raise ValueError(f"Map/spectrum row count mismatch in {rel}.")
        if not np.all(np.isfinite(power)):
            raise ValueError(f"Non-finite power spectrum in {rel}.")
        if np.any(load_array(spectra_leaf / "mode_counts.npy") <= 0):
            raise ValueError(f"Empty Fourier bins in {rel}.")
        results[str(rel)] = {"rows": int(maps.shape[0]), "simulations": int(len(np.unique(load_array(maps_leaf / "simulation_ids.npy"))))}
    if not results:
        raise ValueError(f"No output leaves found under {maps_root}.")
    all_locations: dict[int, str] = {}
    for rel in results:
        sim_ids = np.unique(load_array(maps_root / rel / "simulation_ids.npy"))
        for sim_id in sim_ids.tolist():
            if sim_id in all_locations:
                raise ValueError(f"Simulation {sim_id} appears in both {all_locations[sim_id]} and {rel}.")
            all_locations[int(sim_id)] = rel
    return results
