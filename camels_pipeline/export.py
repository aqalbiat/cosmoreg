from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig
from .dataset import DatasetView, rows_for_simulations
from .provenance import software_versions, utc_now_iso, write_json
from .spectrum import make_geometry, spectra_for_batch


def _prepare_leaf_dir(path: Path, overwrite: bool) -> Path:
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    if path.exists():
        has_contents = any(path.iterdir())
        if has_contents and not overwrite:
            raise FileExistsError(f"{path} exists. Set overwrite=true to replace generated outputs.")
        shutil.rmtree(path)
    partial.mkdir(parents=True)
    return partial


def _publish(partial: Path, final: Path) -> None:
    os.replace(partial, final)


def export_leaf(
    dataset: DatasetView,
    row_ids: np.ndarray,
    maps_leaf: Path,
    spectra_leaf: Path,
    config: PipelineConfig,
    source_identity: dict[str, Any],
) -> dict[str, Any]:
    row_ids = np.asarray(row_ids, dtype=np.int64)
    if len(row_ids) == 0:
        raise ValueError("Cannot export an empty leaf.")
    map_shape = tuple(dataset.maps.shape[1:])
    n = map_shape[0]
    geometry = make_geometry(n=n, box_size=config.box_size_mpc_h, n_bins=config.n_power_bins)
    maps_partial = _prepare_leaf_dir(maps_leaf, config.overwrite)
    spectra_partial = _prepare_leaf_dir(spectra_leaf, config.overwrite)
    try:
        maps_out = np.lib.format.open_memmap(
            maps_partial / "maps.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(row_ids), *map_shape),
        )
        power_out = np.lib.format.open_memmap(
            spectra_partial / "power_spectra.npy",
            mode="w+",
            dtype=np.float64,
            shape=(len(row_ids), config.n_power_bins),
        )
        log_power_out = np.lib.format.open_memmap(
            spectra_partial / "log_power_spectra.npy",
            mode="w+",
            dtype=np.float64,
            shape=(len(row_ids), config.n_power_bins),
        )
        clipped_total = 0
        for start in range(0, len(row_ids), config.batch_size):
            stop = min(start + config.batch_size, len(row_ids))
            ids = row_ids[start:stop]
            raw = np.asarray(dataset.maps[ids], dtype=np.float32)
            maps_out[start:stop] = raw
            power = spectra_for_batch(raw, geometry, config.box_size_mpc_h)
            power_out[start:stop] = power
            clipped_total += int(np.count_nonzero(power <= config.log_epsilon))
            log_power_out[start:stop] = np.log10(np.maximum(power, config.log_epsilon))
        maps_out.flush()
        power_out.flush()
        log_power_out.flush()
        del maps_out
        del power_out
        del log_power_out

        labels = dataset.labels[row_ids].astype(np.float32, copy=False)
        sim_ids = dataset.simulation_ids[row_ids].astype(np.int64, copy=False)
        map_ids = dataset.map_ids[row_ids].astype(np.int64, copy=False)
        for leaf in (maps_partial, spectra_partial):
            np.save(leaf / "labels.npy", labels, allow_pickle=False)
            np.save(leaf / "simulation_ids.npy", sim_ids, allow_pickle=False)
            np.save(leaf / "map_ids.npy", map_ids, allow_pickle=False)
        np.save(spectra_partial / "k.npy", geometry.k, allow_pickle=False)
        np.save(spectra_partial / "bin_edges.npy", geometry.bin_edges, allow_pickle=False)
        np.save(spectra_partial / "mode_counts.npy", geometry.mode_counts, allow_pickle=False)

        common = {
            "created_utc": utc_now_iso(),
            "source_identity": source_identity,
            "config": config.as_dict(),
            "row_count": int(len(row_ids)),
            "simulation_count": int(len(np.unique(sim_ids))),
            "map_shape": list(map_shape),
            "labels": ["Omega_m", "sigma_8"],
            "parameter_columns": list(config.parameter_columns),
            "software": software_versions(),
        }
        maps_manifest = {**common, "product": "maps2d"}
        spectra_manifest = {
            **common,
            "product": "powerspectrum",
            "power_convention": "delta=rho/mean(rho)-1; P2D=(L^2/N^4)|fft2(delta)|^2; radial mean over full FFT modes excluding DC through axis Nyquist",
            "box_size_mpc_h": config.box_size_mpc_h,
            "k_units": "h/Mpc",
            "power_units": "(Mpc/h)^2 for contrast spectrum",
            "log_epsilon": config.log_epsilon,
            "clipped_log_bins": clipped_total,
        }
        write_json(maps_partial / "manifest.json", maps_manifest)
        write_json(spectra_partial / "manifest.json", spectra_manifest)
        _publish(maps_partial, maps_leaf)
        _publish(spectra_partial, spectra_leaf)
        return spectra_manifest
    except Exception:
        if maps_partial.exists():
            shutil.rmtree(maps_partial)
        if spectra_partial.exists():
            shutil.rmtree(spectra_partial)
        raise


def export_splits(
    dataset: DatasetView,
    splits: dict[str, dict[str, np.ndarray]],
    config: PipelineConfig,
    source_identity: dict[str, Any],
    root: Path = Path("."),
) -> dict[str, Any]:
    leaves: dict[str, Any] = {}
    for level, groups in splits.items():
        for group, sim_ids in groups.items():
            row_ids = rows_for_simulations(sim_ids, config.maps_per_simulation)
            maps_leaf = root / config.maps_output_dir / level / group
            spectra_leaf = root / config.spectra_output_dir / level / group
            manifest = export_leaf(dataset, row_ids, maps_leaf, spectra_leaf, config, source_identity)
            leaves[f"{level}/{group}"] = {
                "row_count": manifest["row_count"],
                "simulation_count": manifest["simulation_count"],
            }
    root_manifest = {
        "created_utc": utc_now_iso(),
        "source_identity": source_identity,
        "config": config.as_dict(),
        "leaves": leaves,
        "software": software_versions(),
        "note": "Synthetic smoke outputs are not scientific CAMELS results when source_identity.synthetic is true.",
    }
    write_json(root / config.metadata_dir / "root_manifest.json", root_manifest)
    return root_manifest
