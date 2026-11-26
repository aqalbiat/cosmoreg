"""Leakage, selection, architecture, and a complete synthetic experiment."""

import json
from pathlib import Path
import uuid

import numpy as np
import pytest
import torch
from torch import nn

from autoencoders.data import FOLDS, TARGETS, fit_map_scaler, inspect_level, load_group
from autoencoders.experiment import execute, parser, ridge_candidates
from autoencoders.model import Autoencoder


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "maps"
    rng = np.random.default_rng(73)
    for i, group in enumerate((*FOLDS, "test")):
        leaf = root / "level1" / group
        leaf.mkdir(parents=True)
        # Two complete simulations, two maps per simulation; no rows are subsampled.
        simulation_ids = np.repeat([2 * i, 2 * i + 1], 2)
        labels = np.column_stack((0.2 + simulation_ids * 0.01, 0.7 + simulation_ids * 0.02))
        maps = rng.lognormal(0, 0.15, (4, 256, 256)).astype(np.float32)
        for name, array in {"maps": maps, "labels": labels, "simulation_ids": simulation_ids,
                            "map_ids": np.arange(4 * i, 4 * i + 4)}.items():
            np.save(leaf / f"{name}.npy", array, allow_pickle=False)
    return root


def test_architecture_and_linear_bottleneck():
    torch.set_num_threads(2)
    reference_convolutions = None
    for dimension in (10, 30, 50, 70, 90):
        model = Autoencoder(dimension)
        convolution_shapes = [tuple(layer.weight.shape) for layer in model.modules() if isinstance(layer, nn.Conv2d)]
        if reference_convolutions is not None:
            assert convolution_shapes == reference_convolutions
        reference_convolutions = convolution_shapes
        assert isinstance(model.to_code, nn.Linear) and isinstance(model.from_code, nn.Linear)
        assert isinstance(model.decoder[-1], nn.Conv2d)
        x = torch.randn(2, 1, 256, 256)
        with torch.no_grad():
            assert model.convolutions(x).shape == (2, 32, 16, 16)
            assert model.encode(x).shape == (2, dimension)
            assert model(x).shape == x.shape
            model.to_code.weight.zero_()
            model.to_code.bias.fill_(-2)
            assert torch.all(model.encode(x) == -2)


def test_simulation_leakage_rejected(source):
    leaf = source / "level1" / "test"
    np.save(leaf / "simulation_ids.npy", [0, 0, 11, 11])
    with pytest.raises(ValueError, match="leakage"):
        inspect_level(source, "level1")


def test_density_contrast_and_training_scaler(source, tmp_path):
    signature = inspect_level(source, "level1")
    group = load_group(source, "level1", "fold1", tmp_path / "cache", signature["fold1"])
    raw = np.load(source / "level1" / "fold1" / "maps.npy").astype(np.float64)
    expected = raw / raw.mean(axis=(1, 2), keepdims=True) - 1
    np.testing.assert_allclose(group.maps, expected, rtol=1e-6, atol=1e-7)
    scaler = fit_map_scaler([group])
    np.testing.assert_allclose(scaler["scale"], expected.std(), rtol=1e-6)
    # Changing a held-out map cannot alter a scaler fitted on the training group.
    held_out = source / "level1" / "fold2" / "maps.npy"
    np.save(held_out, np.ones((4, 256, 256), dtype=np.float32) * 100)
    assert fit_map_scaler([group]) == scaler


def test_selection_formula():
    rng = np.random.default_rng(1)
    training = {"codes": rng.normal(size=(20, 5)), "labels": rng.normal(size=(20, 2)) * [1, 20]}
    validation = {"codes": rng.normal(size=(8, 5)), "labels": rng.normal(size=(8, 2)) * [1, 20]}
    records, target_std = ridge_candidates(training, validation, [0.1, 1, 10])
    for record in records:
        expected = np.mean([record["prediction"][name]["mae"] / target_std[i] for i, name in enumerate(TARGETS)])
        assert record["normalized_mae"] == pytest.approx(expected)


def test_complete_experiment_and_resume(source, tmp_path):
    package = Path(__file__).resolve().parent
    # Keep artifacts inside the required roots even with pytest's default tmpdir.
    tmp_path = package / "runs" / "software_tests" / uuid.uuid4().hex
    run = tmp_path / "synthetic"
    hidden = package.parent / "autoencoders_hidden" / "software_tests" / tmp_path.name
    args = parser().parse_args(["--maps-root", str(source), "--run-root", str(run),
        "--cache-root", str(tmp_path / "cache"), "--hidden-root", str(hidden),
        "--levels", "1", "--dimensions", "30", "10", "--max-epochs", "1",
        "--patience", "1", "--batch-size", "4", "--device", "cpu", "--threads", "2", "--no-amp"])
    execute(args)
    assert json.loads((run / "completed.json").read_text())["status"] == "complete"
    lock = json.loads((run / "settings_lock.json").read_text())
    assert lock["winning_dimension"]["level1"] in (10, 30)
    lock_time = (run / "settings_lock.json").stat().st_mtime_ns
    for dimension in (10, 30):
        cv = json.loads((run / "level1" / f"D={dimension}" / "cross_validation.json").read_text())
        assert len(cv["folds"]) == 5
        assert cv["refit_epochs"] == 1
        selected = min(cv["alpha_scores"], key=lambda r: (r["normalized_mae"], r["alpha"]))
        assert cv["selected"] == selected
        for index, alpha in enumerate(args.alphas):
            assert cv["alpha_scores"][index]["normalized_mae"] == pytest.approx(
                np.mean([fold["candidates"][index]["normalized_mae"] for fold in cv["folds"]]))
        code_root = hidden / run.name / f"D={dimension}" / "level1"
        for fold in FOLDS:
            train_dir, val_dir = code_root / fold / "train", code_root / fold / "validation"
            train_manifest = json.loads((train_dir / "manifest.json").read_text())
            val_manifest = json.loads((val_dir / "manifest.json").read_text())
            assert train_manifest["encoder_sha256"] == val_manifest["encoder_sha256"]
            assert fold not in train_manifest["source_training_folds"]
            assert not set(np.load(train_dir / "simulation_ids.npy")) & set(np.load(val_dir / "simulation_ids.npy"))
        dev = code_root / "final" / "development"
        test = code_root / "final" / "test"
        dev_manifest = json.loads((dev / "manifest.json").read_text())
        test_manifest = json.loads((test / "manifest.json").read_text())
        assert dev_manifest["encoder_sha256"] == test_manifest["encoder_sha256"]
        assert np.load(dev / "codes.npy").shape == (20, dimension)
        assert np.load(test / "codes.npy").shape == (4, dimension)
        final = run / "level1" / f"D={dimension}" / "final"
        assert (final / "test_results.json").stat().st_mtime_ns >= lock_time
        prediction = np.load(final / "test_predictions.npy")
        labels = np.load(test / "labels.npy")
        reported = json.loads((final / "test_results.json").read_text())["prediction"]
        for i, name in enumerate(TARGETS):
            assert reported[name]["mae"] == pytest.approx(np.abs(prediction[:, i] - labels[:, i]).mean())
    result_before = (run / "results.json").read_bytes()
    checkpoint = run / "level1" / "D=30" / "fold1" / "encoder_decoder.pt"
    checkpoint_time = checkpoint.stat().st_mtime_ns
    execute(args)
    assert (run / "results.json").read_bytes() == result_before
    assert checkpoint.stat().st_mtime_ns == checkpoint_time
    assert (run / "settings_lock.json").stat().st_mtime_ns == lock_time
