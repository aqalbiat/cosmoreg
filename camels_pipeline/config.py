from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only without PyYAML
    yaml = None


OFFICIAL_BASE_URL = (
    "https://users.flatironinstitute.org/~fvillaescusa/priv/"
    "DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/IllustrisTNG"
)
DEFAULT_MAPS_URL = f"{OFFICIAL_BASE_URL}/Maps_Mtot_IllustrisTNG_LH_z=0.00.npy"
DEFAULT_PARAMS_URL = f"{OFFICIAL_BASE_URL}/params_LH_IllustrisTNG.txt"


@dataclass(frozen=True)
class PipelineConfig:
    suite: str = "IllustrisTNG"
    simulation_set: str = "LH"
    redshift: str = "0.00"
    field: str = "Mtot"
    maps_url: str = DEFAULT_MAPS_URL
    params_url: str = DEFAULT_PARAMS_URL
    local_maps_path: str | None = None
    local_params_path: str | None = None
    raw_data_dir: str = "raw"
    maps_output_dir: str = "maps2d"
    spectra_output_dir: str = "powerspectrum"
    metadata_dir: str = "metadata"
    box_size_mpc_h: float = 25.0
    seed: int = 42
    maps_per_simulation: int = 15
    level_simulation_counts: tuple[int, int, int] = (67, 333, 600)
    test_fraction: float = 0.2
    n_folds: int = 5
    n_power_bins: int = 50
    batch_size: int = 64
    log_epsilon: float = 1.0e-30
    overwrite: bool = False
    expected_map_shape: tuple[int, int] = (256, 256)
    expected_dtype: str = "float32"
    expected_parameter_count: int = 6
    parameter_columns: tuple[str, ...] = (
        "Omega_m",
        "sigma_8",
        "A_SN1",
        "A_AGN1",
        "A_SN2",
        "A_AGN2",
    )

    def validate(self) -> None:
        if self.field != "Mtot":
            raise ValueError("This project scope supports only the Mtot field.")
        if self.suite != "IllustrisTNG" or self.simulation_set != "LH":
            raise ValueError("Default pipeline must use one IllustrisTNG LH source unless extended explicitly.")
        if self.maps_per_simulation <= 0:
            raise ValueError("maps_per_simulation must be positive.")
        if len(self.level_simulation_counts) != 3 or min(self.level_simulation_counts) <= 0:
            raise ValueError("level_simulation_counts must contain three positive counts.")
        if not (0.0 < self.test_fraction < 1.0):
            raise ValueError("test_fraction must be between 0 and 1.")
        if self.n_folds <= 1:
            raise ValueError("n_folds must be at least 2.")
        if self.n_power_bins <= 0:
            raise ValueError("n_power_bins must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.box_size_mpc_h <= 0:
            raise ValueError("box_size_mpc_h must be positive.")
        if self.log_epsilon <= 0:
            raise ValueError("log_epsilon must be positive.")
        if len(self.expected_map_shape) != 2 or min(self.expected_map_shape) <= 1:
            raise ValueError("expected_map_shape must contain two dimensions > 1.")
        if self.expected_parameter_count < 2:
            raise ValueError("At least Omega_m and sigma_8 must be present.")

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("level_simulation_counts", "expected_map_shape", "parameter_columns"):
            data[key] = list(data[key])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        normalized = dict(data)
        aliases = {"set": "simulation_set"}
        for old, new in aliases.items():
            if old in normalized and new not in normalized:
                normalized[new] = normalized.pop(old)
        tuple_fields = ("level_simulation_counts", "expected_map_shape", "parameter_columns")
        for name in tuple_fields:
            if name in normalized and normalized[name] is not None:
                normalized[name] = tuple(normalized[name])
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(normalized) - known)
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")
        config = cls(**normalized)
        config.validate()
        return config


def load_config(path: str | Path = "config.yaml") -> PipelineConfig:
    path = Path(path)
    if not path.exists():
        config = PipelineConfig()
        config.validate()
        return config
    if yaml is None:
        raise RuntimeError("PyYAML is required to read config.yaml. Install requirements.txt.")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a mapping.")
    return PipelineConfig.from_dict(data)
