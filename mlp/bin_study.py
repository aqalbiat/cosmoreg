"""Run the power spectrum bin comparison from the repository root.

Usage: python -m mlp.bin_study [--output DIRECTORY]
"""

import argparse
from contextlib import redirect_stdout
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import torch

from mlp.evaluate import evaluate_test_model
from mlp.experiment import run_experiment
from mlp.final_train import load_final_model, save_final_model, train_final_model


BIN_COUNTS = (10, 30, 70, 90)
LEVELS = (1, 2, 3)
FOLDS = (1, 2, 3, 4, 5)


def write_json(path, report):
    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot encode {type(value).__name__}")

    pending = path.with_suffix(".partial")
    pending.write_text(json.dumps(report, default=encode, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def check_development_alignment(root, pairs):
    """Verify that different bin counts use the same development rows."""
    for level, bins in pairs:
        for fold in FOLDS:
            relative = Path(f"level{level}") / f"fold{fold}"
            reference = root / "powerspectrum" / relative
            candidate = root / "powerspectrum2" / f"D={bins}" / relative
            for filename in ("simulation_ids.npy", "map_ids.npy", "labels.npy"):
                actual = np.load(candidate / filename, allow_pickle=False)
                expected = np.load(reference / filename, allow_pickle=False)
                if not np.array_equal(actual, expected):
                    raise ValueError(f"{candidate / filename} differs from the reference rows.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("mlp/bin_study_results"))
    output = parser.parse_args().output.resolve()
    root = Path(__file__).resolve().parents[1]
    pairs = [(level, bins) for level in LEVELS for bins in BIN_COUNTS]
    check_development_alignment(root, pairs)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)

    configurations = {}
    for level in LEVELS:
        comparison = root / "mlp" / f"level{level}" / f"level{level}_config_comparison.json"
        comparison_bytes = comparison.read_bytes()
        previous = json.loads(comparison_bytes)
        name = previous["ranking"][0]
        configurations[level] = {
            "name": name,
            "settings": previous["configs"][name],
            "source": str(comparison.relative_to(root)),
            "source_sha256": sha256(comparison_bytes).hexdigest(),
        }

    selections = []
    with open(os.devnull, "w") as quiet:
        for level, bins in pairs:
            directory = output / f"D={bins}" / f"level{level}"
            directory.mkdir(parents=True)
            spectra = root / "powerspectrum2" / f"D={bins}"
            configuration = configurations[level]
            settings = configuration["settings"]
            folds = []
            for fold in FOLDS:
                with redirect_stdout(quiet):
                    result = run_experiment(spectra, level, fold, **settings)
                if result["settings"]["input_dim"] != bins:
                    raise ValueError(f"Feature width does not match D={bins}.")
                folds.append({key: result[key] for key in (
                    "validation_fold", "validation_simulations", "settings",
                    "best_epoch", "best_val_loss", "baseline_loss", "map_metrics",
                    "simulation_metrics", "train_history", "val_history",
                )})
                print(f"CV level {level}, D={bins}, fold {fold}: epoch {result['best_epoch']}", flush=True)

            counts = [fold["validation_simulations"] for fold in folds]
            pooled = {}
            for metric in ("rmse", "mae", "baseline_rmse", "baseline_mae"):
                values = np.stack([fold["simulation_metrics"][metric] for fold in folds])
                if metric.endswith("rmse"):
                    pooled[metric] = np.sqrt(np.average(values ** 2, axis=0, weights=counts))
                else:
                    pooled[metric] = np.average(values, axis=0, weights=counts)
            best_epochs = [fold["best_epoch"] for fold in folds]
            selection = {
                "level": level,
                "bins": bins,
                "target_names": result["target_names"],
                "configuration": configuration,
                "final_settings": {**settings, "epochs": sorted(best_epochs)[len(FOLDS) // 2]},
                "development_best_epochs": best_epochs,
                "epoch_selection": "median best epoch across five development folds",
                "validation_simulations": sum(counts),
                "pooled_validation_simulation_metrics": pooled,
            }
            write_json(directory / "validation.json", {**selection, "folds": folds})
            selections.append(selection)

        write_json(output / "final_selections.json", selections)
        frozen = []
        for selection in selections:
            level, bins = selection["level"], selection["bins"]
            directory = output / f"D={bins}" / f"level{level}"
            spectra = root / "powerspectrum2" / f"D={bins}"
            with redirect_stdout(quiet):
                fitted = train_final_model(spectra, level, **selection["final_settings"])
            if fitted["settings"]["input_dim"] != bins:
                raise ValueError(f"Final model input width does not match D={bins}.")
            fitted["settings"].update({
                "configuration_name": selection["configuration"]["name"],
                "development_best_epochs": selection["development_best_epochs"],
                "epoch_selection": selection["epoch_selection"],
            })
            checkpoint = save_final_model(fitted, directory / "checkpoints" / "final.pt")
            restored = load_final_model(checkpoint)
            with torch.no_grad():
                example = torch.zeros((1, bins))
                torch.testing.assert_close(
                    fitted["model"](example), restored["model"](example), rtol=0, atol=0,
                )
            frozen.append({
                **selection,
                "checkpoint": str(checkpoint.relative_to(output)),
                "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
                "training_maps": fitted["training_maps"],
                "training_simulations": len(fitted["training_simulation_ids"]),
                "training_history": fitted["train_history"],
            })
            print(f"Saved level {level}, D={bins}: {len(fitted['train_history'])} epochs", flush=True)

    write_json(output / "frozen_models.json", frozen)
    # All selections and checkpoints are on disk before the first test evaluation.
    print("All 12 final checkpoints are frozen. Starting held-out evaluation.", flush=True)
    summary = []
    for entry in frozen:
        level, bins = entry["level"], entry["bins"]
        directory = output / f"D={bins}" / f"level{level}"
        checkpoint = output / entry["checkpoint"]
        if sha256(checkpoint.read_bytes()).hexdigest() != entry["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed after it was frozen: {checkpoint}")
        report = evaluate_test_model(checkpoint, root / "powerspectrum2" / f"D={bins}")
        write_json(directory / "test_results.json", {"bins": bins, "checkpoint": entry["checkpoint"], **report})
        summary.append({**entry, "test_report": str((directory / "test_results.json").relative_to(output)),
                        "test_simulations": report["test_simulations"],
                        "test_maps": report["test_maps"],
                        "test_map_metrics": report["map_metrics"],
                        "test_simulation_metrics": report["simulation_metrics"]})
        print(f"Test level {level}, D={bins}: RMSE {report['simulation_metrics']['rmse']}", flush=True)
    write_json(output / "summary.json", summary)


if __name__ == "__main__":
    main()
