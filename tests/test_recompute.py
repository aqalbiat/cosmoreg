from pathlib import Path

import numpy as np
import pytest

from camels_pipeline.cli import make_synthetic_source
from camels_pipeline.dataset import load_and_validate_sources
from camels_pipeline.export import export_splits
from camels_pipeline.provenance import read_json, sha256_file
from camels_pipeline.recompute import recompute_spectra
from camels_pipeline.splits import make_splits


def test_recompute_matches_original_export_and_preserves_splits(tmp_path: Path):
    maps_path, params_path, config = make_synthetic_source(tmp_path / "input")
    dataset = load_and_validate_sources(maps_path, params_path, config)
    splits = make_splits(dataset.n_simulations, config)
    export_splits(dataset, splits, config, {"synthetic": True})
    maps_root, reference_root = Path(config.maps_output_dir), Path(config.spectra_output_dir)
    original_hashes = {p: sha256_file(p) for root in (maps_root, reference_root) for p in root.rglob("*") if p.is_file()}
    output_root = tmp_path / "recomputed"
    result = recompute_spectra(maps_root, reference_root, output_root, (3, 5))
    assert set(result) == {"D=3", "D=5"}
    for dimension in (3, 5):
        assert len(result[f"D={dimension}"]["leaves"]) == 18
        for reference in reference_root.glob("*/*/power_spectra.npy"):
            relative = reference.parent.relative_to(reference_root)
            leaf = output_root / f"D={dimension}" / relative
            power = np.load(leaf / "power_spectra.npy", allow_pickle=False)
            assert power.shape == (len(np.load(reference)), dimension)
            assert power.dtype == np.float64
            np.testing.assert_array_equal(
                np.load(leaf / "log_power_spectra.npy"), np.log10(np.maximum(power, config.log_epsilon)),
            )
            if dimension == config.n_power_bins:
                # Same-bin reruns must reproduce the established exporter exactly.
                np.testing.assert_array_equal(power, np.load(reference))
                for name in ("k.npy", "bin_edges.npy", "mode_counts.npy"):
                    np.testing.assert_array_equal(np.load(leaf / name), np.load(reference.parent / name))
            for name in ("labels.npy", "simulation_ids.npy", "map_ids.npy"):
                assert (leaf / name).read_bytes() == (reference.parent / name).read_bytes()
            manifest = read_json(leaf / "manifest.json")
            assert manifest["config"]["n_power_bins"] == dimension
            assert manifest["source_identity"]["synthetic"] is True
            assert not (leaf / "maps.npy").exists()
    assert all(sha256_file(path) == digest for path, digest in original_hashes.items())
    with pytest.raises(FileExistsError):
        recompute_spectra(maps_root, reference_root, output_root, (3, 5))


@pytest.mark.parametrize("dimensions", [(), (0,), (-1,), (3, 3)])
def test_recompute_rejects_invalid_dimensions_before_writing(tmp_path, dimensions):
    with pytest.raises(ValueError):
        recompute_spectra(tmp_path / "maps", tmp_path / "reference", tmp_path / "output", dimensions)
    assert not (tmp_path / "output").exists()
