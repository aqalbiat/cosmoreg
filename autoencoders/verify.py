"""Independently audit an experiment's saved metrics, codes and split identities."""

import argparse
import json
from pathlib import Path
import statistics

import numpy as np

from .data import FOLDS, TARGETS, digest, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_export(leaf, checkpoint, maps_root, level, groups):
    manifest = read(leaf / "manifest.json")
    assert manifest["encoder_sha256"] == digest(checkpoint), f"Wrong encoder: {leaf}"
    for name, expected_hash in manifest["arrays_sha256"].items():
        assert digest(leaf / f"{name}.npy") == expected_hash, f"Changed export: {leaf}/{name}"
    arrays = {name: np.load(leaf / f"{name}.npy", allow_pickle=False) for name in
              ("codes", "labels", "simulation_ids", "map_ids", "source_folds")}
    assert arrays["codes"].shape == (len(arrays["labels"]), manifest["dimension"])
    assert np.isfinite(arrays["codes"]).all()
    for name in ("labels", "simulation_ids", "map_ids"):
        expected = np.concatenate([np.load(maps_root / level / group / f"{name}.npy", allow_pickle=False)
                                   for group in groups])
        np.testing.assert_array_equal(arrays[name], expected)
    expected_folds = np.concatenate([np.full(len(np.load(maps_root / level / group / "map_ids.npy")), group)
                                     for group in groups])
    np.testing.assert_array_equal(arrays["source_folds"], expected_folds)
    return arrays


def audit(run):
    run = Path(run).resolve()
    identity = read(run / "experiment.json")
    settings = identity["settings"]
    maps_root, hidden = Path(settings["maps_root"]), Path(settings["hidden_root"])
    assert (run / "completed.json").exists(), "Experiment is not complete"
    lock_path = run / "settings_lock.json"
    lock = read(lock_path)
    assert lock["settings"] == settings
    results = read(run / "results.json")
    exported_leaves, fold_count = 0, 0
    for level in settings["levels"]:
        selections = []
        for dimension in settings["dimensions"]:
            root = run / level / f"D={dimension}"
            code_root = hidden / f"D={dimension}" / level
            cv = read(root / "cross_validation.json")
            selections.append(cv)
            assert cv == lock["selection"][level][str(dimension)]
            assert cv["refit_epochs"] == int(statistics.median([fold["best_epoch"] for fold in cv["folds"]]))
            for fold in cv["folds"]:
                name = fold["fold"]
                checkpoint = root / name / "encoder_decoder.pt"
                history = read(root / name / "history.json")
                assert fold["best_epoch"] == min(history, key=lambda row: row["validation_mse_normalized"])["epoch"]
                training_groups = [group for group in FOLDS if group != name]
                training = verify_export(code_root / name / "train", checkpoint, maps_root, level, training_groups)
                validation = verify_export(code_root / name / "validation", checkpoint, maps_root, level, [name])
                assert not set(training["simulation_ids"]) & set(validation["simulation_ids"])
                std = training["labels"].std(axis=0)
                np.testing.assert_allclose(std, fold["training_target_std"], rtol=1e-12)
                for candidate in fold["candidates"]:
                    expected = np.mean([candidate["prediction"][target]["mae"] / std[i]
                                        for i, target in enumerate(TARGETS)])
                    np.testing.assert_allclose(expected, candidate["normalized_mae"], rtol=1e-12)
                exported_leaves += 2
                fold_count += 1
            for index, candidate in enumerate(cv["alpha_scores"]):
                expected = np.mean([fold["candidates"][index]["normalized_mae"] for fold in cv["folds"]])
                np.testing.assert_allclose(candidate["normalized_mae"], expected, rtol=1e-12)
            assert cv["selected"] == min(cv["alpha_scores"], key=lambda row: (row["normalized_mae"], row["alpha"]))
            final = root / "final"
            checkpoint = final / "encoder_decoder.pt"
            refit = read(final / "refit.json")
            assert refit["epoch"] == cv["refit_epochs"]
            assert refit["alpha"] == cv["selected"]["alpha"]
            development = verify_export(code_root / "final" / "development", checkpoint, maps_root, level, FOLDS)
            test = verify_export(code_root / "final" / "test", checkpoint, maps_root, level, ["test"])
            assert not set(development["simulation_ids"]) & set(test["simulation_ids"])
            with np.load(final / "ridge.npz", allow_pickle=False) as ridge:
                np.testing.assert_allclose(ridge["feature_mean"], development["codes"].astype(np.float64).mean(axis=0), rtol=1e-10, atol=1e-12)
                expected_scale = development["codes"].astype(np.float64).std(axis=0)
                expected_scale[expected_scale == 0] = 1
                np.testing.assert_allclose(ridge["feature_scale"], expected_scale, rtol=1e-10, atol=1e-12)
                predicted = ((test["codes"] - ridge["feature_mean"]) / ridge["feature_scale"]) @ ridge["coefficient"].T + ridge["intercept"]
            np.testing.assert_allclose(predicted, np.load(final / "test_predictions.npy"), rtol=1e-12)
            test_result = read(final / "test_results.json")
            assert test_result == results["test"][level][str(dimension)]
            assert (final / "test_results.json").stat().st_mtime_ns >= lock_path.stat().st_mtime_ns
            residual = predicted - test["labels"]
            for i, target in enumerate(TARGETS):
                np.testing.assert_allclose(test_result["prediction"][target]["mae"], np.abs(residual[:, i]).mean(), rtol=1e-12)
                np.testing.assert_allclose(test_result["prediction"][target]["rmse"], np.sqrt(np.square(residual[:, i]).mean()), rtol=1e-12)
            exported_leaves += 2
        winner = min(selections, key=lambda row: (row["selected"]["normalized_mae"], row["dimension"]))["dimension"]
        assert winner == lock["winning_dimension"][level] == results["winning_dimension"][level]
    summary = {"status": "passed", "cross_validation_encoders": fold_count,
               "final_encoders": len(settings["levels"]) * len(settings["dimensions"]),
               "verified_export_leaves": exported_leaves,
               "checks": ["encoder and export hashes", "exact source identities and labels", "split separation",
                          "early stopping epochs", "normalized selection scores", "median refit epochs",
                          "development-only latent scaler", "test predictions and original-unit errors",
                          "CV-only dimension selection", "test evaluation after settings lock"]}
    write_json(run / "audit.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="autoencoders/runs/sweep30")
    args = parser.parse_args()
    print(json.dumps(audit(args.run_root), indent=2))


if __name__ == "__main__":
    main()
