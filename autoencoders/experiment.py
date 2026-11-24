"""Resumable CV, locked selection, development refits, and held-out evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import time

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

from .data import (FOLDS, TARGETS, Maps, digest, fit_map_scaler, inspect_level,
                   load_group, metadata, write_json)
from .model import Autoencoder

DIMENSIONS = (10, 30, 50, 70, 90)
SELECTION_RULE = "mean across five folds and two targets of MAE / training-target population std"


def log(message):
    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def loader(groups, scaler, settings, shuffle=False, seed=0):
    return DataLoader(Maps(groups, scaler), batch_size=settings["batch_size"],
                      shuffle=shuffle, num_workers=0,
                      pin_memory=settings["device"].startswith("cuda"),
                      generator=torch.Generator().manual_seed(seed))


def train(groups, validation, dimension, directory, settings, epochs, seed):
    directory.mkdir(parents=True, exist_ok=True)
    scaler_file = directory / "map_scaler.json"
    if scaler_file.exists():
        scaler = json.loads(scaler_file.read_text())
    else:
        scaler = fit_map_scaler(groups)
        write_json(scaler_file, scaler)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = Autoencoder(dimension).to(settings["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    amp = settings["amp"] and settings["device"].startswith("cuda")
    grad_scaler = torch.amp.GradScaler("cuda", enabled=amp)
    device_type = torch.device(settings["device"]).type
    best_loss, best_epoch, history, start = float("inf"), 0, [], 1
    progress_path, best_path = directory / "progress.pt", directory / "encoder_decoder.pt"
    if progress_path.exists():
        state = torch.load(progress_path, map_location=settings["device"], weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        grad_scaler.load_state_dict(state["grad_scaler"])
        best_loss, best_epoch = state["best_loss"], state["best_epoch"]
        history, start = state["history"], state["epoch"] + 1
        torch.set_rng_state(state["rng"].cpu())
        if "cuda_rng" in state:
            torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda_rng"]])
    validation_loader = loader(validation, scaler, settings) if validation else None
    # A completed early-stopped checkpoint must not acquire extra epochs on resume.
    stopped = validation is not None and start - 1 - best_epoch >= settings["patience"]
    for epoch in range(start, epochs + 1) if not stopped else ():
        started = time.perf_counter()
        model.train()
        loss_sum, count = 0.0, 0
        train_loader = loader(groups, scaler, settings, shuffle=True, seed=seed + epoch)
        for maps in train_loader:
            maps = maps.to(settings["device"], non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device_type, enabled=amp):
                reconstructed = model(maps)
                loss = (reconstructed.float() - maps).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite reconstruction loss")
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            loss_sum += float(loss.detach()) * len(maps)
            count += len(maps)
        train_loss = loss_sum / count
        val_loss = None
        if validation_loader is not None:
            model.eval()
            total, count = 0.0, 0
            with torch.inference_mode():
                for maps in validation_loader:
                    maps = maps.to(settings["device"], non_blocking=True)
                    with torch.autocast(device_type=device_type, enabled=amp):
                        reconstructed = model(maps)
                    total += float((reconstructed.float() - maps).square().mean()) * len(maps)
                    count += len(maps)
            val_loss = total / count
        score = train_loss if val_loss is None else val_loss
        # Final refits retain the LAST epoch, never pick an epoch using training loss.
        if validation is None or score < best_loss:
            best_loss, best_epoch = score, epoch
            save_checkpoint(best_path, {"model": model.state_dict(), "dimension": dimension,
                                       "map_scaler": scaler, "epoch": epoch})
        history.append({"epoch": epoch, "training_mse_normalized": train_loss,
                        "validation_mse_normalized": val_loss,
                        "seconds": time.perf_counter() - started})
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "grad_scaler": grad_scaler.state_dict(), "best_loss": best_loss,
                 "best_epoch": best_epoch, "history": history, "epoch": epoch,
                 "rng": torch.get_rng_state()}
        if torch.cuda.is_available():
            state["cuda_rng"] = torch.cuda.get_rng_state_all()
        save_checkpoint(progress_path, state)
        write_json(directory / "history.json", history)
        log(f"{directory.parent.name}/{directory.name} D={dimension} epoch {epoch}/{epochs} "
            f"train={train_loss:.6g} val={val_loss if val_loss is not None else '-'} "
            f"({history[-1]['seconds']:.1f}s)")
        if validation is not None and epoch - best_epoch >= settings["patience"]:
            break
    model.load_state_dict(torch.load(best_path, map_location=settings["device"], weights_only=False)["model"])
    model.eval().requires_grad_(False)
    return model, scaler, best_epoch, history


def encode(model, groups, scaler, settings, destination, encoder_id, source_folds):
    """Codes and reconstruction metrics from exactly one frozen encoder."""
    destination.mkdir(parents=True, exist_ok=True)
    codes, squared, absolute, count = [], 0.0, 0.0, 0
    # Export and evaluation use float32 even if training uses autocast.
    with torch.inference_mode():
        for maps in loader(groups, scaler, settings):
            maps = maps.to(settings["device"], non_blocking=True)
            latent = model.encode(maps)
            reconstruction = model.decode(latent)
            difference = (reconstruction - maps) * scaler["scale"]
            squared += float(difference.double().square().sum())
            absolute += float(difference.double().abs().sum())
            count += difference.numel()
            codes.append(latent.cpu().numpy())
    codes = np.concatenate(codes)
    if not np.isfinite(codes).all():
        raise RuntimeError("Nonfinite latent codes")
    arrays = {"codes": codes, **metadata(groups)}
    for name, value in arrays.items():
        np.save(destination / f"{name}.npy", value, allow_pickle=False)
    # Preserve input fold membership when training/development combines folds.
    np.save(destination / "source_folds.npy", np.concatenate([
        np.full(len(group.maps), group.name, dtype="U5") for group in groups]), allow_pickle=False)
    metrics = {"mse": squared / count, "rmse": float(np.sqrt(squared / count)),
               "mae": absolute / count, "units": "density contrast"}
    write_json(destination / "manifest.json", {
        "encoder_sha256": encoder_id, "source_training_folds": list(source_folds),
        "dimension": int(codes.shape[1]), "rows": len(codes), "targets": TARGETS,
        "map_preprocessing": {"formula": "rho / mean(rho) - 1", **scaler},
        "reconstruction": metrics,
        "inputs": {group.name: group.fingerprint for group in groups},
        "arrays_sha256": {name: digest(destination / f"{name}.npy") for name in (*arrays, "source_folds")},
    })
    return arrays, metrics


def prediction_metrics(truth, predicted):
    residual = np.asarray(predicted, dtype=np.float64) - truth
    return {name: {"mae": float(np.abs(residual[:, i]).mean()),
                   "rmse": float(np.sqrt(np.square(residual[:, i]).mean()))}
            for i, name in enumerate(TARGETS)}


def ridge_candidates(training, validation, alphas):
    scaler = StandardScaler().fit(training["codes"])
    train_codes, validation_codes = scaler.transform(training["codes"]), scaler.transform(validation["codes"])
    target_std = training["labels"].std(axis=0)
    if np.any(target_std <= 0):
        raise ValueError("Each training target must have nonzero variance")
    records = []
    for alpha in alphas:
        regressor = Ridge(alpha=alpha, solver="svd").fit(train_codes, training["labels"])
        metrics = prediction_metrics(validation["labels"], regressor.predict(validation_codes))
        score = float(np.mean([metrics[name]["mae"] / target_std[i] for i, name in enumerate(TARGETS)]))
        records.append({"alpha": alpha, "normalized_mae": score, "prediction": metrics})
    return records, target_std.tolist()


def cross_validate(groups, dimension, level, run, hidden, settings):
    dimension_dir = run / level / f"D={dimension}"
    summary_path = dimension_dir / "cross_validation.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    results = []
    for fold in FOLDS:
        directory = dimension_dir / fold
        result_path = directory / "result.json"
        if result_path.exists():
            results.append(json.loads(result_path.read_text()))
            continue
        training = [groups[name] for name in FOLDS if name != fold]
        validation = [groups[fold]]
        seed = settings["seed"] + 10000 * int(level[-1]) + 100 * dimension + int(fold[-1])
        model, scaler, best_epoch, history = train(training, validation, dimension, directory,
                                                   settings, settings["max_epochs"], seed)
        encoder_id = digest(directory / "encoder_decoder.pt")
        code_dir = hidden / f"D={dimension}" / level / fold
        train_arrays, train_reconstruction = encode(model, training, scaler, settings,
            code_dir / "train", encoder_id, [g.name for g in training])
        val_arrays, val_reconstruction = encode(model, validation, scaler, settings,
            code_dir / "validation", encoder_id, [g.name for g in training])
        candidates, target_std = ridge_candidates(train_arrays, val_arrays, settings["alphas"])
        result = {"fold": fold, "best_epoch": best_epoch, "epochs_run": len(history),
                  "encoder_sha256": encoder_id, "training_target_std": target_std,
                  "reconstruction": {"train": train_reconstruction, "validation": val_reconstruction},
                  "candidates": candidates}
        write_json(result_path, result)
        results.append(result)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    scores = []
    for index, alpha in enumerate(settings["alphas"]):
        candidates = [fold["candidates"][index] for fold in results]
        scores.append({"alpha": alpha,
                       "normalized_mae": float(np.mean([c["normalized_mae"] for c in candidates])),
                       "prediction_mean_across_folds": {
                           name: {metric: float(np.mean([c["prediction"][name][metric] for c in candidates]))
                                  for metric in ("mae", "rmse")} for name in TARGETS}})
    selected = min(scores, key=lambda record: (record["normalized_mae"], record["alpha"]))
    summary = {"dimension": dimension, "selection_rule": SELECTION_RULE, "selected": selected,
               "refit_epochs": int(statistics.median([fold["best_epoch"] for fold in results])),
               "validation_reconstruction_mean_across_folds": {
                   metric: float(np.mean([fold["reconstruction"]["validation"][metric] for fold in results]))
                   for metric in ("mse", "mae", "rmse")},
               "alpha_scores": scores, "folds": results}
    write_json(summary_path, summary)
    return summary


def refit(groups, dimension, level, selection, run, hidden, settings):
    directory = run / level / f"D={dimension}" / "final"
    completion = directory / "refit.json"
    if completion.exists():
        return json.loads(completion.read_text())
    development = [groups[name] for name in FOLDS]
    seed = settings["seed"] + 10000 * int(level[-1]) + 100 * dimension + 6
    model, scaler, epoch, _ = train(development, None, dimension, directory, settings,
                                  selection["refit_epochs"], seed)
    encoder_id = digest(directory / "encoder_decoder.pt")
    arrays, reconstruction = encode(model, development, scaler, settings,
        hidden / f"D={dimension}" / level / "final" / "development", encoder_id, FOLDS)
    latent_scaler = StandardScaler().fit(arrays["codes"])
    ridge = Ridge(alpha=selection["selected"]["alpha"], solver="svd")
    ridge.fit(latent_scaler.transform(arrays["codes"]), arrays["labels"])
    np.savez(directory / "ridge.npz", feature_mean=latent_scaler.mean_, feature_scale=latent_scaler.scale_,
             coefficient=ridge.coef_, intercept=ridge.intercept_, alpha=ridge.alpha)
    result = {"dimension": dimension, "epoch": epoch, "alpha": ridge.alpha,
              "encoder_sha256": encoder_id, "development_reconstruction": reconstruction}
    write_json(completion, result)
    return result


def evaluate_test(test, dimension, level, run, hidden, settings):
    if not (run / "settings_lock.json").exists():
        raise RuntimeError("Test evaluation requires a settings lock")
    directory = run / level / f"D={dimension}" / "final"
    output = directory / "test_results.json"
    if output.exists():
        return json.loads(output.read_text())
    state = torch.load(directory / "encoder_decoder.pt", map_location=settings["device"], weights_only=False)
    model = Autoencoder(dimension).to(settings["device"])
    model.load_state_dict(state["model"])
    model.eval().requires_grad_(False)
    encoder_id = digest(directory / "encoder_decoder.pt")
    arrays, reconstruction = encode(model, [test], state["map_scaler"], settings,
        hidden / f"D={dimension}" / level / "final" / "test", encoder_id, FOLDS)
    with np.load(directory / "ridge.npz", allow_pickle=False) as ridge:
        prediction = ((arrays["codes"] - ridge["feature_mean"]) / ridge["feature_scale"]) @ ridge["coefficient"].T + ridge["intercept"]
    np.save(directory / "test_predictions.npy", prediction, allow_pickle=False)
    result = {"dimension": dimension, "prediction": prediction_metrics(arrays["labels"], prediction),
              "reconstruction": reconstruction, "encoder_sha256": encoder_id}
    write_json(output, result)
    return result


def render_report(run, settings, selection, test_results):
    lines = ["# Convolutional autoencoder + Ridge results", "",
             f"Selection: {SELECTION_RULE}. Equal weight per fold and target; ties use smaller alpha, then smaller D.",
             f"Training cap: {settings['max_epochs']} epochs; patience: {settings['patience']}. "
             "Refits use the median best CV epoch. All errors use original units.", "",
             "RMSE/MAE of reconstruction use density contrast. CV prediction columns are arithmetic means of fold metrics.", "",
             "| Level | D | alpha | Refit epochs | CV normalized MAE | CV MAE Ωm | CV RMSE Ωm | CV MAE σ8 | CV RMSE σ8 | CV reconstruction RMSE | Test MAE Ωm | Test RMSE Ωm | Test MAE σ8 | Test RMSE σ8 | Test reconstruction RMSE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for level in settings["levels"]:
        for dimension in settings["dimensions"]:
            cv = selection[level][str(dimension)]
            test = test_results.get(level, {}).get(str(dimension))
            prediction = cv["selected"]["prediction_mean_across_folds"]
            values = [level, str(dimension), f"{cv['selected']['alpha']:g}", str(cv["refit_epochs"]),
                      f"{cv['selected']['normalized_mae']:.6g}"]
            values += [f"{prediction[name][metric]:.6g}" for name in TARGETS for metric in ("mae", "rmse")]
            values += [f"{cv['validation_reconstruction_mean_across_folds']['rmse']:.6g}"]
            values += ([f"{test['prediction'][name][metric]:.6g}" for name in TARGETS for metric in ("mae", "rmse")]
                       + [f"{test['reconstruction']['rmse']:.6g}"] if test else ["pending"] * 5)
            lines.append("| " + " | ".join(values) + " |")
    lines += ["", "## Winners chosen from CV", ""]
    for level in settings["levels"]:
        winner = min(selection[level].values(), key=lambda result: (result["selected"]["normalized_mae"], result["dimension"]))
        lines.append(f"- {level}: D={winner['dimension']} (CV normalized MAE {winner['selected']['normalized_mae']:.6g}).")
    lines += ["", "A single seeded fit per fold was used. Reported errors are per map; maps from a simulation are correlated. "
              "The held-out fold also supplies reconstruction early stopping, so CV is a tuning estimate. "
              "Test results are held-out estimates, and were not used to select D or alpha.", ""]
    (run / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def execute(args):
    started = time.time()
    run = Path(args.run_root).resolve()
    hidden = Path(args.hidden_root).resolve() / run.name
    # Keep source, checkpoints, metrics, cache and exports in the requested roots.
    package = Path(__file__).resolve().parent
    if not run.is_relative_to(package) or not Path(args.cache_root).resolve().is_relative_to(package):
        raise ValueError("run-root and cache-root must be under autoencoders/")
    hidden_base = package.parent / "autoencoders_hidden"
    if not hidden.is_relative_to(hidden_base):
        raise ValueError("hidden-root must be under autoencoders_hidden/")
    run.mkdir(parents=True, exist_ok=True)
    settings = {"levels": [f"level{i}" for i in args.levels], "dimensions": args.dimensions,
                "max_epochs": args.max_epochs, "patience": args.patience, "batch_size": args.batch_size,
                "learning_rate": args.learning_rate, "alphas": args.alphas, "seed": args.seed,
                "device": args.device, "amp": args.amp,
                "maps_root": str(Path(args.maps_root).resolve()), "hidden_root": str(hidden),
                "cache_root": str(Path(args.cache_root).resolve()),
                "source_sha256": {name: digest(package / name) for name in
                                  ("model.py", "data.py", "experiment.py")}}
    # Basic split checks and input hashes happen before the first training job.
    log("Auditing input splits and hashing source arrays")
    signatures = {level: inspect_level(args.maps_root, level) for level in settings["levels"]}
    identity = {"settings": settings, "inputs": signatures}
    identity_path = run / "experiment.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Run settings, source, or data changed; use a new run-root")
    write_json(identity_path, identity)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    groups = {}
    for level in settings["levels"]:
        groups[level] = {fold: load_group(args.maps_root, level, fold, args.cache_root, signatures[level][fold])
                         for fold in FOLDS}
    order = ([30] if 30 in args.dimensions else []) + [d for d in args.dimensions if d != 30]
    selections = {level: {} for level in settings["levels"]}
    for dimension in order:
        for level in settings["levels"]:
            log(f"Cross-validation {level} D={dimension}")
            selections[level][str(dimension)] = cross_validate(groups[level], dimension, level, run, hidden, settings)
        if dimension == 30:
            write_json(run / "D30_verification.json", {
                "status": "passed", "levels": settings["levels"], "checks": [
                    "five independent encoders per level", "finite training/validation codes",
                    "train-only preprocessing and latent scaler", "all Ridge alphas scored",
                    "test not evaluated"],
                "cross_validation_scores": {level: selections[level]["30"]["selected"] for level in settings["levels"]}})
            log("D=30 cross-validation verified; proceeding to remaining dimensions")
    lock = {"locked_utc": datetime.now(timezone.utc).isoformat(), "selection_rule": SELECTION_RULE,
            "settings": settings, "selection": selections,
            "winning_dimension": {level: min(selections[level].values(), key=lambda result:
                (result["selected"]["normalized_mae"], result["dimension"]))["dimension"] for level in selections}}
    if not (run / "settings_lock.json").exists():
        write_json(run / "settings_lock.json", lock)
    render_report(run, settings, selections, {})
    for level in settings["levels"]:
        for dimension in args.dimensions:
            refit(groups[level], dimension, level, selections[level][str(dimension)], run, hidden, settings)
    # Test maps and labels enter the model pipeline only AFTER settings are locked.
    results = {level: {} for level in settings["levels"]}
    for level in settings["levels"]:
        test = load_group(args.maps_root, level, "test", args.cache_root, signatures[level]["test"])
        for dimension in args.dimensions:
            results[level][str(dimension)] = evaluate_test(test, dimension, level, run, hidden, settings)
            render_report(run, settings, selections, results)
    write_json(run / "results.json", {"selection": selections, "test": results,
                                      "winning_dimension": lock["winning_dimension"]})
    write_json(run / "completed.json", {"status": "complete", "seconds_this_invocation": time.time() - started})
    log(f"Experiment complete: {run / 'REPORT.md'}")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--maps-root", default="maps2d")
    result.add_argument("--run-root", default="autoencoders/runs/sweep30")
    result.add_argument("--hidden-root", default="autoencoders_hidden")
    result.add_argument("--cache-root", default="autoencoders/cache")
    result.add_argument("--levels", nargs="+", type=int, choices=(1, 2, 3), default=[1, 2, 3])
    result.add_argument("--dimensions", nargs="+", type=int, choices=DIMENSIONS, default=list(DIMENSIONS))
    result.add_argument("--max-epochs", type=int, default=30)
    result.add_argument("--patience", type=int, default=10)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--learning-rate", type=float, default=0.001)
    result.add_argument("--alphas", nargs="+", type=float, default=[0.001, 0.01, 0.1, 1., 10., 100., 1000.])
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return result


def main():
    args = parser().parse_args()
    if min(args.max_epochs, args.patience, args.batch_size, args.threads) < 1 or args.learning_rate <= 0:
        raise SystemExit("Epochs, patience, batch size, threads, and learning rate must be positive")
    if len(set(args.dimensions)) != len(args.dimensions) or len(set(args.levels)) != len(args.levels):
        raise SystemExit("Duplicate dimensions or levels")
    if any(alpha < 0 or not np.isfinite(alpha) for alpha in args.alphas) or len(set(args.alphas)) != len(args.alphas):
        raise SystemExit("Alphas must be unique finite nonnegative numbers")
    execute(args)


if __name__ == "__main__":
    main()
