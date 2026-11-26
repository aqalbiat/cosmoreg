"""Summarize an audited run and reproduce its development-only mean baseline."""

import argparse
import json
from pathlib import Path

import numpy as np

from .data import FOLDS, TARGETS, digest, write_json


def read(path):
    return Path(path).read_text(encoding="utf-8")


def summarize(run_root):
    run = Path(run_root).resolve()
    package = Path(__file__).resolve().parent
    if not run.is_relative_to(package):
        raise ValueError("run-root must be under autoencoders/")
    completed = json.loads(read(run / "completed.json"))
    audit = json.loads(read(run / "audit.json"))
    if completed["status"] != "complete" or audit["status"] != "passed":
        raise ValueError("A complete experiment and passing audit are required")
    identity = json.loads(read(run / "experiment.json"))
    settings = identity["settings"]
    results = json.loads(read(run / "results.json"))
    lock = json.loads(read(run / "settings_lock.json"))
    assert results["winning_dimension"] == lock["winning_dimension"]
    maps = Path(settings["maps_root"])
    baseline, comparisons, winners, diagnostics = {}, {}, {}, {}
    for level in settings["levels"]:
        labels = {}
        for fold in FOLDS:
            path = maps / level / fold / "labels.npy"
            assert digest(path) == identity["inputs"][level][fold]["labels.npy"]
            labels[fold] = np.load(path, allow_pickle=False).astype(np.float64)
        folds = []
        for fold in FOLDS:
            training = np.concatenate([labels[name] for name in FOLDS if name != fold])
            residual = training.mean(axis=0) - labels[fold]
            mae = np.abs(residual).mean(axis=0)
            rmse = np.sqrt(np.square(residual).mean(axis=0))
            folds.append({
                "fold": fold,
                "normalized_mae": float(np.mean(mae / training.std(axis=0))),
                "prediction": {target: {"mae": float(mae[i]), "rmse": float(rmse[i])}
                               for i, target in enumerate(TARGETS)},
            })
        baseline[level] = {
            "normalized_mae": float(np.mean([fold["normalized_mae"] for fold in folds])),
            "prediction_mean_across_folds": {
                target: {metric: float(np.mean([fold["prediction"][target][metric] for fold in folds]))
                         for metric in ("mae", "rmse")} for target in TARGETS},
            "folds": folds,
        }
        # Retain and independently check the baseline recorded during training.
        old_baseline_path = run / "baseline_cv.json"
        if old_baseline_path.exists():
            old = json.loads(read(old_baseline_path))[level]
            np.testing.assert_allclose(baseline[level]["normalized_mae"], old["normalized_mae"], rtol=1e-12)
            for new_fold, old_fold in zip(folds, old["folds"]):
                assert new_fold["fold"] == f"fold{old_fold['fold']}"
                np.testing.assert_allclose(new_fold["normalized_mae"], old_fold["normalized_mae"], rtol=1e-12)
                for metric in ("mae", "rmse"):
                    np.testing.assert_allclose([new_fold["prediction"][t][metric] for t in TARGETS],
                                               old_fold[metric], rtol=1e-12)
        comparisons[level] = {}
        diagnostics[level] = {}
        for dimension in settings["dimensions"]:
            cv = results["selection"][level][str(dimension)]
            score = cv["selected"]["normalized_mae"]
            reference = baseline[level]["normalized_mae"]
            comparisons[level][str(dimension)] = {
                "cv_normalized_mae": score,
                "baseline_normalized_mae": reference,
                "improvement_percent": 100 * (reference - score) / reference,
                "alpha": cv["selected"]["alpha"],
                "refit_epochs": cv["refit_epochs"],
                "cv_reconstruction_rmse": cv["validation_reconstruction_mean_across_folds"]["rmse"],
            }
            final = run / level / f"D={dimension}" / "final"
            refit = json.loads(read(final / "refit.json"))
            scaler = json.loads(read(final / "map_scaler.json"))
            history = json.loads(read(final / "history.json"))
            codes_path = Path(settings["hidden_root"]) / f"D={dimension}" / level / "final" / "development" / "codes.npy"
            codes = np.load(codes_path, allow_pickle=False)
            diagnostics[level][str(dimension)] = {
                "first_epoch_training_mse_normalized": history[0]["training_mse_normalized"],
                "last_epoch_training_mse_normalized": history[-1]["training_mse_normalized"],
                "development_reconstruction_mse_normalized": refit["development_reconstruction"]["mse"] / scaler["scale"] ** 2,
                "varying_latent_coordinates": int(np.count_nonzero(np.ptp(codes, axis=0))),
                "latent_coordinates": dimension,
            }
        winner = lock["winning_dimension"][level]
        assert winner == min(settings["dimensions"], key=lambda d: (
            comparisons[level][str(d)]["cv_normalized_mae"], d))
        winners[level] = {
            "dimension": winner,
            **comparisons[level][str(winner)],
            "test_prediction": results["test"][level][str(winner)]["prediction"],
            "test_reconstruction_rmse": results["test"][level][str(winner)]["reconstruction"]["rmse"],
        }
    summary = {"baseline_recipe": "For each held-out development fold, predict the other four folds' "
               "training-label mean. Average MAE / training-label population std equally over folds and targets.",
               "baseline": baseline, "comparisons": comparisons, "cv_selected_winners": winners,
               "final_fit_diagnostics": diagnostics,
               "evidence_sha256": {name: digest(run / name) for name in
                                   ("experiment.json", "settings_lock.json", "results.json", "audit.json")}}
    write_json(run / "summary.json", summary)
    dimensions_text = ",".join(str(d) for d in settings["dimensions"])
    lines = ["# Autoencoder experiment interpretation", "",
             f"The sweep is complete: {audit['cross_validation_encoders']} independent cross-validation "
             f"encoders, {audit['final_encoders']} final development refits, and "
             f"{audit['verified_export_leaves']} code export leaves across "
             f"{len(settings['levels'])} levels and D={dimensions_text}. "
             "The independent [artifact audit](audit.json) passed. "
             "[Full CV and test results](REPORT.md), [machine-readable results](results.json), "
             "and [recomputed baseline evidence](summary.json) are saved beside this document.", "",
             "## Models selected using development folds", "",
             "D and Ridge alpha were selected using the equally weighted, training-standard-deviation-normalized "
             "validation MAE over five folds and both targets. Final epoch counts are median best CV epochs. "
             "The [settings lock](settings_lock.json) precedes every test evaluation. "
             "The following test MAE and RMSE values are per map and in original target units.", "",
             "| Level | Selected D | Alpha | Refit epochs | Test MAE Omega_m | Test RMSE Omega_m | Test MAE sigma_8 | Test RMSE sigma_8 | Test reconstruction RMSE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for level, winner in winners.items():
        errors = [f"{winner['test_prediction'][target][metric]:.8f}"
                  for target in TARGETS for metric in ("mae", "rmse")]
        values = [level, str(winner["dimension"]), f"{winner['alpha']:g}", str(winner["refit_epochs"]),
                  *errors, f"{winner['test_reconstruction_rmse']:.6f}"]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "Reconstruction RMSE is in density-contrast units. "
              f"All {audit['final_encoders']} D/level test combinations, including the unselected "
              "dimensions, are in the full report.", "",
              "## Comparison with a training-mean baseline", "",
              summary["baseline_recipe"], "This baseline uses development labels only. "
              "It reproduces the baseline recorded during training. Positive improvement means lower CV "
              "normalized MAE than the baseline; negative means higher error. "
              "No test scores enter this comparison.", "",
              "| Level | D | Baseline CV normalized MAE | Autoencoder + Ridge CV normalized MAE | Improvement (%) | CV reconstruction RMSE |",
              "|---|---:|---:|---:|---:|---:|"]
    for level, dimensions in comparisons.items():
        for dimension, row in dimensions.items():
            lines.append(f"| {level} | {dimension} | {row['baseline_normalized_mae']:.9f} | "
                         f"{row['cv_normalized_mae']:.9f} | {row['improvement_percent']:+.4f} | "
                         f"{row['cv_reconstruction_rmse']:.6f} |")
    maximum_difference = max(abs(row["improvement_percent"])
                             for dimensions in comparisons.values() for row in dimensions.values())
    lines += ["", f"Across this sweep, all CV prediction scores lie within {maximum_difference:.3f}% "
              "of the training-mean baseline. Better reconstruction at larger D therefore does not establish "
              "a useful improvement in cosmological prediction. The small CV differences do not establish "
              "statistical significance; these runs provide no repeated-seed analysis or confidence intervals."]
    if all(row["alpha"] == max(settings["alphas"])
           for dimensions in comparisons.values() for row in dimensions.values()):
        lines += ["", f"Every selected Ridge alpha is {max(settings['alphas']):g}, the largest value in "
                  "the predefined grid. This is a boundary selection and is consistent with strong shrinkage "
                  "being preferred among the tested candidates. The optimum beyond that grid was not tested; "
                  "the grid was kept fixed throughout this experiment."]
    weak_refits = [(level, dimension, row) for level, dimensions in diagnostics.items()
                   for dimension, row in dimensions.items()
                   if row["development_reconstruction_mse_normalized"] >= 0.99]
    if weak_refits:
        lines += ["", "## Final refits with little reconstruction learning", "",
                  "Predicting the training pixel mean gives normalized development reconstruction MSE 1. "
                  "The following final refits have exported development MSE at least 0.99, so they provide "
                  "little reconstruction improvement over that reference. This is a reporting diagnostic; "
                  "it did not change selection, seeds, or the locked training budget.", "",
                  "| Level | D | Last training MSE (normalized) | Exported development MSE (normalized) | Varying latent coordinates |",
                  "|---|---:|---:|---:|---:|"]
        for level, dimension, row in weak_refits:
            lines.append(f"| {level} | {dimension} | {row['last_epoch_training_mse_normalized']:.9f} | "
                         f"{row['development_reconstruction_mse_normalized']:.9f} | "
                         f"{row['varying_latent_coordinates']}/{row['latent_coordinates']} |")
        lines += ["", "These checkpoints and codes are retained and their test errors are reported. "
                  "Variation in latent codes alone does not demonstrate that useful features were learned. "
                  "Full diagnostics for every refit are in summary.json."]
    lines += ["", f"The experiment used one seeded fit per fold, at most {settings['max_epochs']} training "
              f"epochs, and patience {settings['patience']}. "
              "The same held-out fold supplies reconstruction early stopping and Ridge validation. "
              "CV is consequently a tuning estimate. All maps from each simulation stay together, "
              "but the reported per-map errors are correlated within simulations. "
              "The result supports reconstruction with the requested bottlenecks under this training budget; "
              "it does not demonstrate that unsupervised reconstruction learns predictive cosmological features.", "",
              "## Reproduce the final checks and summary", "", "```powershell",
              f"python -m autoencoders.verify --run-root {run.relative_to(package.parent).as_posix()}",
              f"python -m autoencoders.summarize --run-root {run.relative_to(package.parent).as_posix()}", "```", "",
              "The summarizer reads saved results and development labels; it does not train models or "
              "reevaluate test maps. Running the experiment driver again regenerates REPORT.md; rerun "
              "this summarizer afterward to restore its interpretation link.", ""]
    (run / "INTERPRETATION.md").write_text("\n".join(lines), encoding="utf-8")
    report_path = run / "REPORT.md"
    report = read(report_path)
    link = "See [baseline comparison and interpretation](INTERPRETATION.md) for the development-only baseline, "
    link += "CV-selected model test summary, and prediction limitations."
    if link not in report:
        report_path.write_text(report.rstrip() + "\n\n" + link + "\n", encoding="utf-8")
    return {"status": "complete", "summary": str(run / "INTERPRETATION.md"), "winners": winners}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="autoencoders/runs/sweep30")
    args = parser.parse_args()
    print(json.dumps(summarize(args.run_root), indent=2))


if __name__ == "__main__":
    main()
