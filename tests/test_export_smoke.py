from pathlib import Path

import numpy as np

from camels_pipeline.audit import audit_outputs
from camels_pipeline.cli import make_synthetic_source
from camels_pipeline.dataset import load_and_validate_sources
from camels_pipeline.export import export_splits
from camels_pipeline.provenance import file_identity
from camels_pipeline.splits import make_splits


def test_synthetic_export_alignment_and_group_isolation(tmp_path: Path):
    maps_path, params_path, config = make_synthetic_source(tmp_path / "input")
    dataset = load_and_validate_sources(maps_path, params_path, config)
    splits = make_splits(dataset.n_simulations, config)
    export_splits(
        dataset,
        splits,
        config,
        {
            "maps": file_identity(maps_path),
            "params": file_identity(params_path),
            "synthetic": True,
        },
    )

    results = audit_outputs(Path(config.maps_output_dir), Path(config.spectra_output_dir))
    assert sum(item["rows"] for item in results.values()) == 60

    seen = {}
    for rel in results:
        maps_leaf = Path(config.maps_output_dir) / rel
        spectra_leaf = Path(config.spectra_output_dir) / rel
        labels = np.load(maps_leaf / "labels.npy", allow_pickle=False)
        ps_labels = np.load(spectra_leaf / "labels.npy", allow_pickle=False)
        sim_ids = np.load(maps_leaf / "simulation_ids.npy", allow_pickle=False)
        map_ids = np.load(maps_leaf / "map_ids.npy", allow_pickle=False)
        assert np.array_equal(labels, ps_labels)
        assert np.array_equal(sim_ids, map_ids // config.maps_per_simulation)
        for sim_id in np.unique(sim_ids).tolist():
            assert sim_id not in seen
            seen[sim_id] = rel
            expected_label = dataset.params[sim_id, :2].astype(np.float32)
            assert np.all(labels[sim_ids == sim_id] == expected_label)
    assert set(seen) == set(range(30))
