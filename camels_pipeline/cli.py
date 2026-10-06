from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .audit import audit_outputs
from .config import PipelineConfig, load_config
from .dataset import load_and_validate_sources, validate_dataset
from .download import download_file
from .export import export_splits
from .provenance import file_identity
from .splits import make_splits, make_subset_splits, split_summary


def source_paths(config: PipelineConfig) -> tuple[Path, Path]:
    maps_path = Path(config.local_maps_path) if config.local_maps_path else Path(config.raw_data_dir) / Path(config.maps_url).name
    params_path = Path(config.local_params_path) if config.local_params_path else Path(config.raw_data_dir) / Path(config.params_url).name
    return maps_path, params_path


def cmd_download(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    maps_path, params_path = source_paths(config)
    print(f"Downloading/verifying maps: {maps_path}")
    maps_identity = download_file(config.maps_url, maps_path, overwrite=args.overwrite or config.overwrite)
    print(f"Downloading/verifying params: {params_path}")
    params_identity = download_file(config.params_url, params_path, overwrite=args.overwrite or config.overwrite)
    print({"maps": maps_identity, "params": params_identity})


def cmd_prepare(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    maps_path = Path(args.maps) if args.maps else source_paths(config)[0]
    params_path = Path(args.params) if args.params else source_paths(config)[1]
    dataset = load_and_validate_sources(maps_path, params_path, config)
    if args.subset_250:
        splits = make_subset_splits(dataset.n_simulations, 250, config)
    else:
        splits = make_splits(dataset.n_simulations, config)
    source_identity = {
        "maps": file_identity(maps_path, config.maps_url),
        "params": file_identity(params_path, config.params_url),
        "synthetic": False,
    }
    manifest = export_splits(dataset, splits, config, source_identity, Path("."))
    print(split_summary(splits, config.maps_per_simulation))
    print(f"Wrote {len(manifest['leaves'])} paired leaves.")


def cmd_audit(args: argparse.Namespace) -> None:
    results = audit_outputs(Path(args.maps_root), Path(args.spectra_root))
    print(results)


def make_synthetic_source(root: Path) -> tuple[Path, Path, PipelineConfig]:
    root.mkdir(parents=True, exist_ok=True)
    n_sim, maps_per_sim, n = 30, 2, 16
    maps = np.empty((n_sim * maps_per_sim, n, n), dtype=np.float32)
    x = np.arange(n)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    for sim in range(n_sim):
        for view in range(maps_per_sim):
            idx = sim * maps_per_sim + view
            maps[idx] = 2.0 + 0.1 * sim + 0.01 * view + 0.05 * np.cos(2 * np.pi * (xx + view) / n)
    params = np.column_stack(
        [
            np.linspace(0.1, 0.5, n_sim),
            np.linspace(0.6, 1.0, n_sim),
            np.ones((n_sim, 4)),
        ]
    )
    maps_path = root / "synthetic_Maps_Mtot_IllustrisTNG_LH_z=0.00.npy"
    params_path = root / "synthetic_params_LH_IllustrisTNG.txt"
    np.save(maps_path, maps, allow_pickle=False)
    np.savetxt(params_path, params)
    config = PipelineConfig(
        local_maps_path=str(maps_path),
        local_params_path=str(params_path),
        maps_output_dir=str(root / "maps2d"),
        spectra_output_dir=str(root / "powerspectrum"),
        metadata_dir=str(root / "metadata"),
        maps_per_simulation=maps_per_sim,
        level_simulation_counts=(10, 10, 10),
        expected_map_shape=(n, n),
        n_power_bins=5,
        batch_size=4,
        overwrite=True,
    )
    return maps_path, params_path, config


def cmd_smoke_test(args: argparse.Namespace) -> None:
    root = Path(args.output)
    maps_path, params_path, config = make_synthetic_source(root / "input")
    dataset = validate_dataset(np.load(maps_path, mmap_mode="r", allow_pickle=False), np.loadtxt(params_path), config)
    splits = make_splits(dataset.n_simulations, config)
    source_identity = {
        "maps": file_identity(maps_path, None),
        "params": file_identity(params_path, None),
        "synthetic": True,
        "warning": "Synthetic fixture for software validation only; not scientific CAMELS results.",
    }
    export_splits(dataset, splits, config, source_identity, Path("."))
    print(audit_outputs(Path(config.maps_output_dir), Path(config.spectra_output_dir)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CAMELS Mtot data preparation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "prepare"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="config.yaml")
    sub.choices["download"].add_argument("--overwrite", action="store_true")
    sub.choices["download"].set_defaults(func=cmd_download)
    sub.choices["prepare"].add_argument("--maps")
    sub.choices["prepare"].add_argument("--params")
    sub.choices["prepare"].add_argument("--subset-250", action="store_true")
    sub.choices["prepare"].set_defaults(func=cmd_prepare)
    audit = sub.add_parser("audit")
    audit.add_argument("--maps-root", default="maps2d")
    audit.add_argument("--spectra-root", default="powerspectrum")
    audit.set_defaults(func=cmd_audit)
    smoke = sub.add_parser("smoke-test")
    smoke.add_argument("--output", default="synthetic_outputs")
    smoke.set_defaults(func=cmd_smoke_test)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
