"""Export publication figures from a completed, audited autoencoder experiment.

Run from the repository root: python -m autoencoders.plot_results
No fitting, selection, or scientific result files are changed.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import AsinhNorm, Normalize
import numpy as np
import torch

from .data import FOLDS, TARGETS, digest
from .model import Autoencoder


LABELS = (r"$\Omega_m$", r"$\sigma_8$")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#7570B3")
METRICS = ("mae", "rmse")


def errors(truth, prediction):
    residual = prediction - truth
    return {"mae": np.abs(residual).mean(axis=0),
            "rmse": np.sqrt(np.square(residual).mean(axis=0))}


def by_simulation(truth, prediction, ids):
    simulations = np.unique(ids)
    return (np.stack([truth[ids == i][0] for i in simulations]),
            np.stack([prediction[ids == i].mean(axis=0) for i in simulations]))


class Export:
    def __init__(self, run, output, dpi):
        self.run, self.output, self.dpi = run.resolve(), output.resolve(), dpi
        self.repo = Path(__file__).resolve().parent.parent
        self.inputs, self.figures, self.rows, self.prediction_rows = {}, [], [], []
        self.histories, self.alpha_rows, self.examples = [], [], []
        self.data = {}
        if self.read(self.run / "completed.json")["status"] != "complete":
            raise ValueError("The experiment must be complete")
        if self.read(self.run / "audit.json")["status"] != "passed":
            raise ValueError("The saved artifact audit must have passed")
        self.identity = self.read(self.run / "experiment.json")
        self.settings = self.identity["settings"]
        self.training_source_checks = {}
        for name, expected in self.settings["source_sha256"].items():
            path = self.repo / "autoencoders" / name
            current = self.record(path)
            normalized = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            if expected not in (current, normalized):
                raise ValueError(f"Training source changed beyond checkout line endings: {name}")
            self.training_source_checks[name] = {"recorded_sha256": expected,
                "checkout_sha256": current, "lf_normalized_sha256": normalized,
                "matches_after_lf_normalization": normalized == expected}
        self.results = self.read(self.run / "results.json")
        self.summary = self.read(self.run / "summary.json")
        self.lock = self.read(self.run / "settings_lock.json")
        for name, expected in self.summary["evidence_sha256"].items():
            if self.record(self.run / name) != expected:
                raise ValueError(f"Summary evidence changed: {name}")
        if self.results["winning_dimension"] != self.lock["winning_dimension"]:
            raise ValueError("Results disagree with the locked CV winners")
        self.levels = self.settings["levels"]
        self.dimensions = sorted(self.settings["dimensions"])
        self.winners = self.lock["winning_dimension"]
        self.maps = Path(self.settings["maps_root"])
        self.hidden = Path(self.settings["hidden_root"])
        self.output.mkdir(parents=True, exist_ok=True)
        old_manifest = self.output / "manifest.json"
        if old_manifest.exists():
            old = json.loads(old_manifest.read_text(encoding="utf-8"))
            if old["source_run"] != self.relative(self.run):
                raise ValueError("Use a separate output directory for a different run")

    def relative(self, path):
        path = Path(path).resolve()
        return path.relative_to(self.repo).as_posix() if path.is_relative_to(self.repo) else str(path)

    def record(self, path):
        key = self.relative(path)
        if key not in self.inputs:
            self.inputs[key] = digest(path)
        return self.inputs[key]

    def read(self, path):
        self.record(path)
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def array(self, path, expected=None, mmap=False):
        actual = self.record(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Input hash mismatch: {path}")
        return np.load(path, allow_pickle=False, mmap_mode="r" if mmap else None)

    def load(self):
        for level in self.levels:
            labels = [self.array(self.maps / level / fold / "labels.npy",
                                 self.identity["inputs"][level][fold]["labels.npy"])
                      for fold in FOLDS]
            training_mean = np.concatenate(labels).astype(np.float64).mean(axis=0)
            for dimension in self.dimensions:
                key = str(dimension)
                cv = self.results["selection"][level][key]
                if cv != self.lock["selection"][level][key]:
                    raise ValueError("CV results disagree with the settings lock")
                final = self.run / level / f"D={dimension}" / "final"
                leaf = self.hidden / f"D={dimension}" / level / "final" / "test"
                manifest = self.read(leaf / "manifest.json")
                arrays = {name: self.array(leaf / f"{name}.npy", manifest["arrays_sha256"][name])
                          for name in ("labels", "simulation_ids", "map_ids")}
                truth, ids = arrays["labels"], arrays["simulation_ids"]
                source = self.maps / level / "test"
                for name, array in arrays.items():
                    expected = self.identity["inputs"][level]["test"][f"{name}.npy"]
                    np.testing.assert_array_equal(array, self.array(source / f"{name}.npy", expected))
                prediction = self.array(final / "test_predictions.npy")
                if prediction.shape != truth.shape or not np.isfinite(prediction).all():
                    raise ValueError(f"Invalid saved predictions: {final}")
                score = errors(truth, prediction)
                test = self.results["test"][level][key]
                for metric in METRICS:
                    np.testing.assert_allclose(score[metric],
                        [test["prediction"][t][metric] for t in TARGETS], rtol=1e-10, atol=1e-12)
                baseline = errors(truth, np.broadcast_to(training_mean, truth.shape))
                sim_truth, sim_prediction = by_simulation(truth, prediction, ids)
                sim_score = errors(sim_truth, sim_prediction)
                sim_baseline = errors(sim_truth, np.broadcast_to(training_mean, sim_truth.shape))
                history = self.read(final / "history.json")
                refit = self.read(final / "refit.json")
                scaler = self.read(final / "map_scaler.json")
                diagnostic = self.summary["final_fit_diagnostics"][level][key]
                normalized_mse = refit["development_reconstruction"]["mse"] / scaler["scale"] ** 2
                np.testing.assert_allclose(normalized_mse,
                    diagnostic["development_reconstruction_mse_normalized"], rtol=1e-12)
                for candidate in cv["alpha_scores"]:
                    self.alpha_rows.append({"level": level, "dimension": dimension, **{
                        k: candidate[k] for k in ("alpha", "normalized_mae")}})
                for fold in (*FOLDS, "final"):
                    records = history if fold == "final" else self.read(
                        self.run / level / f"D={dimension}" / fold / "history.json")
                    for record in records:
                        self.histories.append({"level": level, "dimension": dimension,
                                               "fold": fold, **record})
                row = {"level": level, "dimension": dimension,
                       "selected_from_cv": dimension == self.winners[level],
                       "alpha": cv["selected"]["alpha"], "refit_epochs": cv["refit_epochs"],
                       "cv_normalized_mae": cv["selected"]["normalized_mae"],
                       "baseline_cv_normalized_mae": self.summary["baseline"][level]["normalized_mae"],
                       "cv_improvement_percent": self.summary["comparisons"][level][key]["improvement_percent"],
                       "cv_reconstruction_rmse": cv["validation_reconstruction_mean_across_folds"]["rmse"],
                       "test_reconstruction_rmse": test["reconstruction"]["rmse"],
                       "development_reconstruction_mse_normalized": normalized_mse,
                       "weak_refit": normalized_mse >= 0.99,
                       "test_maps": len(truth), "test_simulations": len(sim_truth)}
                for i, target in enumerate(TARGETS):
                    for metric in METRICS:
                        row[f"cv_{target}_{metric}"] = cv["selected"]["prediction_mean_across_folds"][target][metric]
                        for prefix, values in (("test", score), ("baseline_test", baseline),
                                               ("simulation_test", sim_score), ("baseline_simulation_test", sim_baseline)):
                            row[f"{prefix}_{target}_{metric}"] = float(values[metric][i])
                self.rows.append(row)
                self.data[level, dimension] = dict(row=row, cv=cv, truth=truth, prediction=prediction,
                    ids=ids, map_ids=arrays["map_ids"], baseline_mean=training_mean,
                    sim_truth=sim_truth, sim_prediction=sim_prediction, history=history,
                    test_manifest=manifest, scaler=scaler)
                for i in range(len(truth)):
                    record = {"level": level, "dimension": dimension,
                              "selected_from_cv": row["selected_from_cv"],
                              "simulation_id": int(ids[i]), "map_id": int(arrays["map_ids"][i])}
                    for j, target in enumerate(TARGETS):
                        record.update({f"true_{target}": float(truth[i, j]),
                                       f"predicted_{target}": float(prediction[i, j]),
                                       f"baseline_{target}": float(training_mean[j])})
                    self.prediction_rows.append(record)

    def axes(self, title, rows=1, cols=3, size=None, note=None, **kwargs):
        fig, axes = plt.subplots(rows, cols, figsize=size or (4.2 * cols, 3.5 * rows),
                                 squeeze=False, layout="constrained", **kwargs)
        fig.suptitle(title + ("\n" + note if note else ""), fontsize=13, fontweight="semibold")
        for ax in axes.flat:
            ax.grid(alpha=0.2)
            ax.set_axisbelow(True)
        return fig, axes

    def save(self, fig, name, caption):
        for extension in ("png", "pdf"):
            fig.savefig(self.output / f"{name}.{extension}", dpi=self.dpi,
                        bbox_inches="tight", facecolor="white")
        self.pdf.savefig(fig, bbox_inches="tight", facecolor="white")
        self.figures.append({"name": name, "caption": caption})
        plt.close(fig)
        print(f"Exported {name}", flush=True)

    def star(self, ax, x, y, color):
        ax.scatter([x], [y], s=150, marker="*", color=color, edgecolors="black", linewidths=0.6, zorder=5)

    def values(self, level, field):
        return [self.data[level, d]["row"][field] for d in self.dimensions]

    def dimension_axes(self, axes):
        for ax, level in zip(axes.flat, self.levels):
            ax.set(title=level.replace("level", "Level "), xlabel="Latent dimension D", xticks=self.dimensions)

    def selection_figures(self):
        for number, field, title, ylabel in (
            (1, "cv_normalized_mae", "Predictive model selection on development folds", "CV normalized MAE (lower is better)"),
            (2, "cv_improvement_percent", "Prediction gain relative to the training-mean baseline", "CV MAE reduction versus baseline (%)")):
            fig, axes = self.axes(title, note="Stars mark locked CV-selected dimensions; five-fold means, equally weighted targets.")
            self.dimension_axes(axes)
            for ax, level, color in zip(axes.flat, self.levels, COLORS):
                ax.plot(self.dimensions, self.values(level, field), "o-", color=color, label="Autoencoder + Ridge")
                reference = self.summary["baseline"][level]["normalized_mae"] if number == 1 else 0
                ax.axhline(reference, color="0.4", linestyle="--", label="Training-mean baseline")
                winner = self.winners[level]
                self.star(ax, winner, self.data[level, winner]["row"][field], color)
                ax.set_ylabel(ylabel)
                ax.legend()
            self.save(fig, f"{number:02d}_{field}_vs_dimension",
                      "Development-only CV comparison. Small differences do not establish statistical significance; no confidence intervals or repeated seeds are available.")
        fig, axes = self.axes("Ridge regularization on frozen autoencoder codes",
                              note="All selected alphas reach 1000, the upper boundary of the predefined grid.")
        for ax, level in zip(axes.flat, self.levels):
            for d, color in zip(self.dimensions, COLORS):
                candidates = self.data[level, d]["cv"]["alpha_scores"]
                ax.plot([c["alpha"] for c in candidates], [c["normalized_mae"] for c in candidates],
                        "o-", color=color, label=f"D={d}", markersize=3)
            ax.axhline(self.summary["baseline"][level]["normalized_mae"], color="0.4", linestyle="--")
            ax.set(xscale="log", xlabel="Ridge alpha", ylabel="CV normalized MAE", title=level.replace("level", "Level "))
            ax.legend(ncol=2)
        self.save(fig, "03_ridge_alpha_sensitivity", "Alpha is tuned separately for every D and level using five-fold CV. Dashed lines show the development-only mean baseline.")

    def error_figures(self):
        for number, split in ((4, "cv"), (5, "test")):
            fig, axes = self.axes(f"{'Cross-validation' if split == 'cv' else 'Held-out test'} cosmological prediction errors",
                rows=2, cols=2, size=(11, 7.5), note="Original target units, per map. Stars identify CV-selected models, including on test plots.")
            for j, target in enumerate(TARGETS):
                for i, metric in enumerate(METRICS):
                    ax = axes[i, j]
                    for level, color in zip(self.levels, COLORS):
                        values = self.values(level, f"{split}_{target}_{metric}")
                        ax.plot(self.dimensions, values, "o-", color=color, label=level.replace("level", "Level "))
                        winner = self.winners[level]
                        self.star(ax, winner, self.data[level, winner]["row"][f"{split}_{target}_{metric}"], color)
                    ax.set(xlabel="Latent dimension D", ylabel=f"{LABELS[j]} {metric.upper()}", xticks=self.dimensions)
                    ax.legend()
            self.save(fig, f"{number:02d}_{split}_prediction_errors",
                "All D/level combinations are retained. CV values average fold metrics. The unselected level-3 D=70 final refit learned almost no reconstruction; its lower test error is not grounds to change the locked D=10 selection.")
        fig, axes = self.axes("Test errors of the models selected using CV", rows=2, cols=2, size=(11, 7.5),
                              note="Autoencoder + Ridge versus a constant prediction fitted on development labels only.")
        x = np.arange(len(self.levels))
        for j, target in enumerate(TARGETS):
            for i, metric in enumerate(METRICS):
                ax = axes[i, j]
                for offset, prefix, color, label in ((-0.18, "test", COLORS[0], "Autoencoder + Ridge"),
                        (0.18, "baseline_test", "0.65", "Development-mean baseline")):
                    values = [self.data[l, self.winners[l]]["row"][f"{prefix}_{target}_{metric}"] for l in self.levels]
                    bars = ax.bar(x + offset, values, width=0.35, color=color, label=label)
                    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
                ax.set(xticks=x, xticklabels=[f"Level {l[-1]}\nD={self.winners[l]}" for l in self.levels],
                       ylabel=f"{LABELS[j]} {metric.upper()} (per map)")
                ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
                ax.legend(loc="upper right")
        self.save(fig, "06_selected_test_errors_vs_baseline", "Per-map errors in original units. The baseline is estimated from all five development folds, independently of test labels.")

    def prediction_figures(self):
        for number, simulation in ((7, False), (8, True)):
            unit = "simulation" if simulation else "map"
            fig, axes = self.axes(f"CV-selected models: predicted versus true parameters per {unit}",
                rows=len(self.levels), cols=2, size=(10, 10),
                note="Dashed diagonal: perfect prediction. Dotted horizontal: development-mean baseline.")
            for i, level in enumerate(self.levels):
                record = self.data[level, self.winners[level]]
                truth = record["sim_truth"] if simulation else record["truth"]
                prediction = record["sim_prediction"] if simulation else record["prediction"]
                for j, target in enumerate(TARGETS):
                    ax = axes[i, j]
                    lo, hi = truth[:, j].min(), truth[:, j].max()
                    for other in self.levels:
                        data = self.data[other, self.winners[other]]
                        lo = min(lo, data["truth"][:, j].min(), data["prediction"][:, j].min())
                        hi = max(hi, data["truth"][:, j].max(), data["prediction"][:, j].max())
                    pad = (hi - lo) * 0.06
                    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="0.4", linewidth=1)
                    ax.axhline(record["baseline_mean"][j], color="0.35", linestyle=":")
                    ax.scatter(truth[:, j], prediction[:, j], s=14 if simulation else 7,
                               alpha=0.75 if simulation else 0.3, color=COLORS[i], rasterized=True)
                    prefix = "simulation_test" if simulation else "test"
                    rmse = record["row"][f"{prefix}_{target}_rmse"]
                    ax.set(xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad),
                        xlabel=f"True {LABELS[j]}", ylabel=f"Predicted {LABELS[j]}",
                        title=f"Level {level[-1]} | D={self.winners[level]} | n={len(truth)} | RMSE={rmse:.4f}")
            self.save(fig, f"{number:02d}_selected_predictions_per_{unit}",
                "Predictions use the locked final models and saved scalers. " +
                ("Map predictions are averaged by simulation ID before computing these metrics; no model is refitted." if simulation else
                 "Maps in each simulation share labels and are correlated. Shared target axis limits make shrinkage toward the development mean visible."))
        fig, axes = self.axes("CV-selected models: held-out test residual distributions", rows=len(self.levels),
                              cols=2, size=(10, 9), note="Residual = prediction minus truth, in original units; distributions are per map.")
        for i, level in enumerate(self.levels):
            data = self.data[level, self.winners[level]]
            for j, target in enumerate(TARGETS):
                ax = axes[i, j]
                residual = data["prediction"][:, j] - data["truth"][:, j]
                baseline = data["baseline_mean"][j] - data["truth"][:, j]
                bins = np.linspace(min(residual.min(), baseline.min()), max(residual.max(), baseline.max()), 27)
                ax.hist(residual, bins=bins, density=True, alpha=0.5, color=COLORS[i], label="Autoencoder + Ridge")
                ax.hist(baseline, bins=bins, density=True, histtype="step", color="0.25", linestyle="--", label="Mean baseline")
                ax.axvline(0, color="0.35", linewidth=1)
                ax.set(xlabel=f"{LABELS[j]} residual", ylabel="Density", title=f"Level {level[-1]} | D={self.winners[level]}")
                ax.legend()
        self.save(fig, "09_selected_test_residuals", "The same saved predictions as the scatter figures; baseline and model distributions use identical bins within each panel.")

    def reconstruction_figures(self):
        fig, axes = self.axes("Map reconstruction quality versus latent dimension",
            note="Density-contrast RMSE. CV is the mean of five fold RMSE values; test uses each final refit.")
        self.dimension_axes(axes)
        for ax, level in zip(axes.flat, self.levels):
            for field, color, label in (("cv_reconstruction_rmse", COLORS[0], "CV validation"),
                                        ("test_reconstruction_rmse", COLORS[1], "Held-out test")):
                ax.plot(self.dimensions, self.values(level, field), "o-", color=color, label=label)
                winner = self.winners[level]
                self.star(ax, winner, self.data[level, winner]["row"][field], color)
            ax.set_ylabel("Reconstruction RMSE")
            ax.legend()
        self.save(fig, "10_reconstruction_rmse_vs_dimension", "Stars mark CV-selected predictive models, which need not have the lowest reconstruction error. The weak D=70 level-3 final refit is retained.")
        fig, axes = self.axes("Better reconstruction does not guarantee better prediction",
            note="Development CV only: each point is one latent dimension, with five-fold mean metrics.")
        for ax, level, color in zip(axes.flat, self.levels, COLORS):
            ax.plot(self.values(level, "cv_reconstruction_rmse"), self.values(level, "cv_normalized_mae"), "o-", color=color)
            for dimension in self.dimensions:
                row = self.data[level, dimension]["row"]
                ax.annotate(f"D={dimension}", (row["cv_reconstruction_rmse"], row["cv_normalized_mae"]),
                            xytext=(4, 6), textcoords="offset points", fontsize=8)
            ax.axhline(self.summary["baseline"][level]["normalized_mae"], color="0.4", linestyle="--")
            ax.margins(x=0.2, y=0.25)
            ax.set(title=f"Level {level[-1]}", xlabel="Reconstruction RMSE (density contrast)", ylabel="Prediction CV normalized MAE")
        self.save(fig, "11_reconstruction_prediction_tradeoff", "Lower values are better on both axes. Dashed lines show the training-mean prediction baseline; no test data enter this plot.")
        fig, axes = self.axes("Autoencoder reconstruction learning curves for every CV fit",
            rows=len(self.levels), cols=len(self.dimensions), size=(19, 10),
            note="Dashed orange: training. Solid blue: validation. Black dots: best validation epoch in each fold.")
        for i, level in enumerate(self.levels):
            for j, dimension in enumerate(self.dimensions):
                ax = axes[i, j]
                for fold in FOLDS:
                    history = [r for r in self.histories if r["level"] == level and r["dimension"] == dimension and r["fold"] == fold]
                    epochs = [r["epoch"] for r in history]
                    ax.plot(epochs, [r["training_mse_normalized"] for r in history], "--", color=COLORS[1], alpha=0.55, linewidth=1)
                    ax.plot(epochs, [r["validation_mse_normalized"] for r in history], color=COLORS[0], alpha=0.55, linewidth=1)
                    best = min(history, key=lambda r: r["validation_mse_normalized"])
                    ax.scatter(best["epoch"], best["validation_mse_normalized"], s=11, color="0.15", zorder=3)
                ax.set(title=f"Level {level[-1]} | D={dimension}", xlabel="Epoch", ylabel="Normalized pixel MSE")
        self.save(fig, "12_all_cv_learning_curves", "All 75 CV histories are shown individually. Each encoder has its own training-fitted normalization. Early stopping uses reconstruction validation MSE.")
        for number, final_loss in ((13, False), (14, True)):
            title = "Final development refit training curves" if not final_loss else "Final refit reconstruction diagnostic"
            fig, axes = self.axes(title, note="The training pixel-mean reference has normalized development MSE 1.")
            for ax, level in zip(axes.flat, self.levels):
                if final_loss:
                    ax.plot(self.dimensions, self.values(level, "development_reconstruction_mse_normalized"), "o-", color=COLORS[0])
                    ax.set(xlabel="Latent dimension D", xticks=self.dimensions)
                else:
                    for d, color in zip(self.dimensions, COLORS):
                        history = self.data[level, d]["history"]
                        ax.plot([r["epoch"] for r in history], [r["training_mse_normalized"] for r in history], color=color, label=f"D={d}")
                    ax.set_xlabel("Epoch")
                    ax.legend(ncol=2)
                ax.axhline(1, color="0.4", linestyle="--")
                ax.set(title=f"Level {level[-1]}", ylabel="Normalized pixel MSE")
            self.save(fig, "14_final_refit_reconstruction_diagnostic" if final_loss else "13_final_refit_learning_curves",
                "Level 3, D=70 stays near the mean-map reference. Final fits retain their last epoch after the CV-prescribed duration; no test-based checkpoint selection occurs. " +
                ("Values are exported development reconstruction MSE divided by the training pixel variance." if final_loss else
                 "Training curves show online epoch loss while parameters are updated, so they need not equal frozen-checkpoint export metrics."))
        fig, axes = self.axes("Training duration selected from development folds",
            note="Dots: best reconstruction epoch in each CV fold. Line: median epoch count used for final refitting.")
        self.dimension_axes(axes)
        for ax, level in zip(axes.flat, self.levels):
            for d in self.dimensions:
                folds = self.data[level, d]["cv"]["folds"]
                ax.scatter(d + np.linspace(-3, 3, len(folds)), [r["best_epoch"] for r in folds], color="0.5", s=24)
            ax.plot(self.dimensions, self.values(level, "refit_epochs"), "o-", color=COLORS[0])
            ax.axhline(self.settings["max_epochs"], color="0.4", linestyle="--", label="Epoch cap")
            ax.set(ylabel="Epoch", ylim=(0, self.settings["max_epochs"] + 2))
            ax.legend()
        self.save(fig, "15_cv_epochs_and_final_refit_budget", "Five folds determine each median duration. Horizontal offsets distinguish overlapping fold markers and do not represent different D values.")

    def map_examples(self, level, dimension, number, weak=False):
        data = self.data[level, dimension]
        ids = data["ids"]
        unique = np.unique(ids)
        selected_ids = unique[np.linspace(0, len(unique) - 1, min(3, len(unique)), dtype=int)]
        indices = [int(np.flatnonzero(ids == sim)[0]) for sim in selected_ids]
        source = self.maps / level / "test" / "maps.npy"
        raw = self.array(source, self.identity["inputs"][level]["test"]["maps.npy"], mmap=True)
        sample = np.asarray(raw[indices], dtype=np.float64)
        contrast = (sample / sample.mean(axis=(1, 2), keepdims=True) - 1).astype(np.float32)
        checkpoint = self.run / level / f"D={dimension}" / "final" / "encoder_decoder.pt"
        if self.record(checkpoint) != data["test_manifest"]["encoder_sha256"]:
            raise ValueError("Reconstruction checkpoint does not match the saved test export")
        # Locally trained, hash-verified checkpoints contain NumPy scaler scalars,
        # matching the experiment's checkpoint loader.
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["map_scaler"] != data["scaler"] or state["dimension"] != dimension:
            raise ValueError("Checkpoint settings do not match the final refit")
        model = Autoencoder(dimension)
        model.load_state_dict(state["model"])
        model.eval()
        normalized = contrast[:, None].copy()
        normalized -= state["map_scaler"]["mean"]
        normalized /= state["map_scaler"]["scale"]
        with torch.inference_mode():
            latent = model.encode(torch.from_numpy(normalized))
            reconstruction = model.decode(latent).numpy()[:, 0]
        leaf = self.hidden / f"D={dimension}" / level / "final" / "test"
        codes = self.array(leaf / "codes.npy", data["test_manifest"]["arrays_sha256"]["codes"])
        np.testing.assert_allclose(latent.numpy(), codes[indices], rtol=3e-4, atol=3e-5)
        reconstruction = reconstruction * state["map_scaler"]["scale"] + state["map_scaler"]["mean"]
        residual = reconstruction - contrast
        upper = max(float(np.percentile(contrast, 99.5)), float(np.percentile(reconstruction, 99.5)), 1.0)
        residual_limit = max(float(np.percentile(np.abs(residual), 99.5)), 1e-8)
        norm = AsinhNorm(linear_width=1, vmin=-1, vmax=upper)
        residual_norm = Normalize(vmin=-residual_limit, vmax=residual_limit)
        fig, axes = self.axes(f"{'Weak final refit' if weak else 'CV-selected final model'} | Level {level[-1]}, D={dimension}",
            rows=len(indices), cols=3, size=(12, 10),
            note="Three predetermined simulations; first map in each. Color limits clip outer tails for display only.")
        for i, index in enumerate(indices):
            rmse = float(np.sqrt(np.mean(residual[i].astype(np.float64) ** 2)))
            self.examples.append({"level": level, "dimension": dimension,
                "simulation_id": int(ids[index]), "map_id": int(data["map_ids"][index]), "rmse": rmse,
                "display_contrast_min": -1, "display_contrast_max": upper, "display_residual_limit": residual_limit})
            for j, image in enumerate((contrast[i], reconstruction[i], residual[i])):
                ax = axes[i, j]
                rendered = ax.imshow(image, origin="lower", cmap="RdBu_r" if j == 2 else "magma",
                    norm=residual_norm if j == 2 else norm, interpolation="nearest", extent=(0, 25, 0, 25))
                if j == 0:
                    signal_image = rendered
                ax.grid(False)
                title = ("Original density contrast", "Reconstruction", f"Residual | RMSE={rmse:.3f}")[j]
                ax.set(title=title, xlabel="x [Mpc/h]", ylabel=f"Simulation {ids[index]}, map {data['map_ids'][index]}\ny [Mpc/h]")
        fig.colorbar(signal_image, ax=axes[:, :2], shrink=0.65, label="Density contrast (asinh color scale)", extend="both")
        fig.colorbar(rendered, ax=axes[:, 2], shrink=0.65, label="Reconstructed minus original density contrast", extend="both")
        self.save(fig, f"{number:02d}_{'weak_refit' if weak else 'selected'}_reconstructions_{level}_D{dimension}",
            "Examples use the first map from the first, middle and last sorted test simulation IDs, independent of reconstruction quality. The checkpoint is frozen; CPU inference only. Original/reconstruction colors share [-1, joint 99.5th percentile], and residual colors use symmetric 99.5th-percentile absolute residual limits. Full-map RMSE includes all pixels. Box side is 25 Mpc/h for this saved experiment.")

    def table(self, name, rows):
        with (self.output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def finish(self):
        self.table("metrics.csv", self.rows)
        self.table("test_predictions.csv", self.prediction_rows)
        self.table("learning_curves.csv", self.histories)
        self.table("ridge_alpha_scores.csv", self.alpha_rows)
        self.table("reconstruction_examples.csv", self.examples)
        lines = ["# Autoencoder result figures", "",
            "Figures from the completed, audited `sweep30` experiment: three levels, D=10,30,50,70,90, "
            "75 CV encoders and 15 final refits. No retraining or model selection was performed during export.", "",
            "Each figure is available as a PNG and a PDF. [All figures in one PDF](all_figures.pdf).", "",
            "## Interpretation", "",
            "The locked CV-selected dimensions are level 1: D=90, level 2: D=90, and level 3: D=10. "
            "Larger latent dimensions improve CV reconstruction, but prediction CV scores remain within 0.4% "
            "of the training-mean baseline. These small differences do not establish useful predictive gains "
            "or statistical significance. All selected Ridge alphas are 1000, the upper grid boundary.", "",
            "The level-3 D=70 final refit has normalized development reconstruction MSE approximately 1. "
            "Its test scores and reconstruction examples are retained as a disclosed diagnostic; test errors "
            "were not used to replace the locked level-3 D=10 winner.", "",
            "Unless a figure explicitly says per simulation, prediction errors are per map in original target "
            "units. Maps from the same simulation are correlated. CV figures average fold metrics; the validation "
            "fold also supplies reconstruction early stopping, so CV is a tuning estimate. Fold curves are "
            "individual fits, not confidence intervals. Reconstruction errors use density contrast, without "
            "logarithms. The mean-prediction baseline is fitted only on development labels.", "",
            "## Figures", "", "| Figure | Description |", "|---|---|"]
        for item in self.figures:
            name = item["name"]
            lines.append(f"| [{name}]({name}.png) ([PDF]({name}.pdf)) | {item['caption']} |")
        lines += ["", "## Data and provenance", "",
            "- [metrics.csv](metrics.csv): all 15 CV/test combinations, baseline errors, CV selection and refit diagnostics.",
            "- [test_predictions.csv](test_predictions.csv): predictions, labels, map/simulation IDs and baseline for all dimensions.",
            "- [learning_curves.csv](learning_curves.csv): all 75 CV and 15 final training histories.",
            "- [ridge_alpha_scores.csv](ridge_alpha_scores.csv): all predefined alpha candidates.",
            "- [reconstruction_examples.csv](reconstruction_examples.csv): fixed example identities, full-image RMSE and display limits.",
            "- [manifest.json](manifest.json): source and output SHA-256 hashes, software versions, locked winners and export settings.", "",
            "Scatter figures use saved test predictions. Simulation figures average those predictions by simulation ID. "
            "Reconstruction example figures run the frozen checkpoints on three fixed test maps per level; "
            "no fitted state or experimental results are overwritten. Their image color limits clip extreme "
            "values for legibility, while RMSE includes every pixel. These examples are illustrative, not aggregate estimates.", "",
            "## Reproduce", "", "```powershell", "python -m autoencoders.plot_results", "```", "",
            "Requires the original ignored run artifacts, hidden codes/labels and map exports in their saved "
            "locations. The exported figures and CSV files can be shared independently. Dependencies: "
            "`python -m pip install -r autoencoders/requirements.txt`.", "",
            "The exporter verifies completion, the saved passing audit and summary evidence hashes, source "
            "label/ID hashes, locked selections, saved prediction errors for every D/level, refit diagnostics, "
            "and reconstruction checkpoint identities. It does not repeat the full scientific artifact audit.", ""]
        lines += ["Training sources match the recorded training source hashes after normalizing checkout "
                  "line endings to LF. Both recorded and current source hashes are retained in the manifest. "
                  "No training source file was edited during this export.", ""]
        (self.output / "README.md").write_text("\n".join(lines), encoding="utf-8")
        outputs = [p for item in self.figures for p in (self.output / f"{item['name']}.png", self.output / f"{item['name']}.pdf")]
        outputs += [self.output / name for name in ("all_figures.pdf", "README.md", "metrics.csv", "test_predictions.csv",
                    "learning_curves.csv", "ridge_alpha_scores.csv", "reconstruction_examples.csv")]
        manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_run": self.relative(self.run),
            "dpi": self.dpi, "versions": {"matplotlib": matplotlib.__version__, "numpy": np.__version__, "torch": str(torch.__version__)},
            "locked_winning_dimensions": self.winners, "figures": self.figures, "source_sha256": self.inputs,
            "training_source_checks": self.training_source_checks,
            "reconstruction_example_rule": "First map from first, middle and last sorted test simulation IDs",
            "checks": {"saved_audit_passed": True, "summary_hashes_match": True, "locked_selection_matches": True,
                "all_15_saved_prediction_metrics_recomputed": True, "test_identity_and_label_alignment": True,
                "refit_diagnostics_match": True, "example_checkpoint_hashes_match": True,
                "example_codes_match_saved_exports": True},
            "output_sha256": {p.name: digest(p) for p in outputs}}
        manifest["exporter_sha256"] = digest(Path(__file__))
        (self.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def execute(self):
        self.load()
        torch.set_num_threads(2)
        plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.spines.top": False,
            "axes.spines.right": False, "legend.fontsize": 8, "pdf.fonttype": 42, "savefig.facecolor": "white"})
        with PdfPages(self.output / "all_figures.pdf", metadata={"Title": "Autoencoder sweep30 results"}) as self.pdf:
            self.selection_figures()
            self.error_figures()
            self.prediction_figures()
            self.reconstruction_figures()
            for i, level in enumerate(self.levels, 16):
                self.map_examples(level, self.winners[level], i)
            number = 16 + len(self.levels)
            for level in self.levels:
                for d in self.dimensions:
                    if self.data[level, d]["row"]["weak_refit"] and d != self.winners[level]:
                        self.map_examples(level, d, number, weak=True)
                        number += 1
        self.finish()
        print(f"Saved {len(self.figures)} figures as PNG + PDF, a combined PDF and supporting data to {self.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("autoencoders/runs/sweep30"))
    parser.add_argument("--output-root", type=Path, default=Path("autoencoders/results"))
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("dpi must be at least 72")
    Export(args.run_root, args.output_root, args.dpi).execute()


if __name__ == "__main__":
    main()
