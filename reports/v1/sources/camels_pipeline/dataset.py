from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import PipelineConfig


@dataclass(frozen=True)
class DatasetView:
    maps: np.memmap
    params: np.ndarray
    labels: np.ndarray
    simulation_ids: np.ndarray
    map_ids: np.ndarray

    @property
    def n_simulations(self) -> int:
        return int(self.params.shape[0])


def load_source_maps(path: Path) -> np.memmap:
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except ValueError as exc:
        raise ValueError(f"Could not load {path} as a non-pickle .npy array.") from exc


def load_params(path: Path) -> np.ndarray:
    params = np.loadtxt(path, dtype=np.float64)
    if params.ndim == 1:
        params = params.reshape(1, -1)
    return params


def validate_dataset(maps: np.ndarray, params: np.ndarray, config: PipelineConfig) -> DatasetView:
    if maps.ndim != 3:
        raise ValueError(f"Maps must have shape [n_maps, nx, ny], got {maps.shape}.")
    if tuple(maps.shape[1:]) != tuple(config.expected_map_shape):
        raise ValueError(f"Expected map shape {config.expected_map_shape}, got {maps.shape[1:]}.")
    if str(maps.dtype) != config.expected_dtype:
        raise ValueError(f"Expected map dtype {config.expected_dtype}, got {maps.dtype}.")
    if params.ndim != 2:
        raise ValueError("Parameter file must be a 2D array.")
    if params.shape[1] < 2:
        raise ValueError("Parameter file must contain at least Omega_m and sigma_8.")
    if config.expected_parameter_count and params.shape[1] != config.expected_parameter_count:
        raise ValueError(f"Expected {config.expected_parameter_count} parameter columns, got {params.shape[1]}.")
    if maps.shape[0] != params.shape[0] * config.maps_per_simulation:
        raise ValueError(
            f"Expected n_maps == n_params * {config.maps_per_simulation}; "
            f"got {maps.shape[0]} maps and {params.shape[0]} parameter rows."
        )
    n_maps = int(maps.shape[0])
    map_ids = np.arange(n_maps, dtype=np.int64)
    simulation_ids = map_ids // config.maps_per_simulation
    labels = params[simulation_ids, :2].astype(np.float32, copy=False)
    return DatasetView(maps=maps, params=params, labels=labels, simulation_ids=simulation_ids, map_ids=map_ids)


def load_and_validate_sources(maps_path: Path, params_path: Path, config: PipelineConfig) -> DatasetView:
    return validate_dataset(load_source_maps(maps_path), load_params(params_path), config)


def rows_for_simulations(simulation_ids: np.ndarray, maps_per_simulation: int) -> np.ndarray:
    sim_ids = np.asarray(simulation_ids, dtype=np.int64)
    offsets = np.arange(maps_per_simulation, dtype=np.int64)
    return (sim_ids[:, None] * maps_per_simulation + offsets[None, :]).reshape(-1)
