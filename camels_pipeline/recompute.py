"""Recompute spectral bin counts from existing maps without changing their splits.

Run from the repository root:
    python -m camels_pipeline.recompute --dimensions 10 30 70 90
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np

from .audit import audit_outputs
from .config import PipelineConfig
from .provenance import file_identity, read_json, software_versions, utc_now_iso, write_json
from .spectrum import make_geometry, spectra_for_batch


def recompute_spectra(
    maps_root: Path,
    reference_root: Path,
    output_root: Path,
    dimensions: tuple[int, ...] = (10, 30, 70, 90),
) -> dict[str, dict]:
    """Export new spectra, copying reference labels/IDs in their existing order.

    Geometry, log clipping, and batching come from each reference leaf's config.
    No splits are regenerated. Existing output files are never overwritten.
    """
    maps_root, reference_root, output_root = map(Path, (maps_root, reference_root, output_root))
    if not dimensions or any(not isinstance(d, int) or d < 1 for d in dimensions):
        raise ValueError("dimensions must contain positive integers.")
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("dimensions must be unique.")

    summary = audit_outputs(maps_root, reference_root)
    references = {}
    for relative, counts in summary.items():
        source = reference_root / relative
        manifest = read_json(source / "manifest.json")
        config = PipelineConfig.from_dict(manifest["config"])
        maps = np.load(maps_root / relative / "maps.npy", mmap_mode="r", allow_pickle=False)
        if maps.shape != (counts["rows"], *config.expected_map_shape):
            raise ValueError(f"Unexpected map shape in {relative}: {maps.shape}")
        if maps.dtype != np.dtype(config.expected_dtype):
            raise ValueError(f"Unexpected map dtype in {relative}: {maps.dtype}")
        if manifest["row_count"] != counts["rows"] or manifest["simulation_count"] != counts["simulations"]:
            raise ValueError(f"Reference manifest counts disagree with arrays in {relative}.")
        if manifest["labels"] != ["Omega_m", "sigma_8"]:
            raise ValueError(f"Unexpected target order in {relative}.")
        # Check all requested geometries before starting any output writes.
        for dimension in dimensions:
            make_geometry(maps.shape[1], config.box_size_mpc_h, dimension)
        references[relative] = (manifest, config, file_identity(source / "manifest.json"))
        del maps

    for dimension in dimensions:
        destination = output_root / f"D={dimension}"
        if destination.exists() and any(path.is_file() for path in destination.rglob("*")):
            raise FileExistsError(f"Refusing to overwrite existing outputs under {destination}.")

    completed = {}
    for dimension in dimensions:
        destination = output_root / f"D={dimension}"
        for index, (relative, (reference, config, reference_identity)) in enumerate(references.items(), 1):
            source = reference_root / relative
            leaf = destination / relative
            partial = leaf.with_name(leaf.name + ".partial")
            partial.mkdir(parents=True, exist_ok=False)
            maps = np.load(maps_root / relative / "maps.npy", mmap_mode="r", allow_pickle=False)
            geometry = make_geometry(maps.shape[1], config.box_size_mpc_h, dimension)
            shape = (len(maps), dimension)
            power_out = np.lib.format.open_memmap(
                partial / "power_spectra.npy", mode="w+", dtype=np.float64, shape=shape,
            )
            log_out = np.lib.format.open_memmap(
                partial / "log_power_spectra.npy", mode="w+", dtype=np.float64, shape=shape,
            )
            clipped = 0
            for start in range(0, len(maps), config.batch_size):
                stop = min(start + config.batch_size, len(maps))
                raw = np.asarray(maps[start:stop], dtype=np.float32)
                power = spectra_for_batch(raw, geometry, config.box_size_mpc_h)
                power_out[start:stop] = power
                log_out[start:stop] = np.log10(np.maximum(power, config.log_epsilon))
                clipped += int(np.count_nonzero(power <= config.log_epsilon))
            power_out.flush()
            log_out.flush()
            del power_out, log_out, maps

            for filename in ("labels.npy", "simulation_ids.npy", "map_ids.npy"):
                shutil.copyfile(source / filename, partial / filename)
            for filename, values in (
                ("k.npy", geometry.k), ("bin_edges.npy", geometry.bin_edges),
                ("mode_counts.npy", geometry.mode_counts),
            ):
                np.save(partial / filename, values, allow_pickle=False)

            manifest = deepcopy(reference)
            manifest.update(
                created_utc=utc_now_iso(), software=software_versions(), clipped_log_bins=clipped,
                reference_spectrum_manifest=reference_identity,
                source_maps_leaf=str(maps_root / relative),
            )
            manifest["config"].update(
                n_power_bins=dimension, spectra_output_dir=str(destination),
                maps_output_dir=str(maps_root), metadata_dir=str(destination), overwrite=False,
            )
            write_json(partial / "manifest.json", manifest)
            # Empty placeholder directories are allowed, but never replace a populated leaf.
            if leaf.exists():
                leaf.rmdir()
            partial.rename(leaf)
            print(f"D={dimension}: {index}/{len(references)} {relative} ({shape[0]} maps)", flush=True)

        audited = audit_outputs(maps_root, destination)
        root_manifest = {
            "created_utc": utc_now_iso(), "n_power_bins": dimension,
            "maps_root": str(maps_root), "reference_spectra_root": str(reference_root),
            "leaves": audited, "software": software_versions(),
            "note": "Recomputed with camels_pipeline.spectrum; reference splits and row order preserved. "
                    "Each leaf retains its original source identity and synthetic-data flag.",
        }
        write_json(destination / "root_manifest.json", root_manifest)
        completed[f"D={dimension}"] = root_manifest
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps-root", type=Path, default=Path("maps2d"))
    parser.add_argument("--reference-root", type=Path, default=Path("powerspectrum"))
    parser.add_argument("--output-root", type=Path, default=Path("powerspectrum2"))
    parser.add_argument("--dimensions", type=int, nargs="+", default=[10, 30, 70, 90])
    args = parser.parse_args()
    recompute_spectra(args.maps_root, args.reference_root, args.output_root, tuple(args.dimensions))


if __name__ == "__main__":
    main()
