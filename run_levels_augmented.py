"""
Train a group of models on one level, then rank them.

Narrowing the models down level by level (for example 20 -> 10 -> 5):

    python run_levels.py --level 1              # every model of depth 4, 6, 8, 10
    python run_levels.py --level 2 --top 10     # the 10 best from level 1
    python run_levels.py --level 3 --top 5      # the 5 best from level 2

Scores are written to results_level{N}_{tag}.json (default tag: "danial"),
not to the shared results_level{N}.json, so two people training in parallel
never edit the same results file.

A run can be stopped with Ctrl+C and started again with the same command:
models that already have a score for that level are skipped.

Other useful options:

    --time                  run one epoch per model, save nothing, and print
                            an estimate of the full training time
    --models A B C          train exactly these models instead of a ranking
    --force                 retrain models that already have a score
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time
import traceback

from loaddata import load_level_data
from train_single import MODEL_REGISTRY, train_one_model

DEFAULT_DEPTHS = [4, 6, 8, 10]
DEFAULT_TAG = "danial"


def results_path(level, tag):
    return f"results_level{level}_{tag}.json"


def load_scores(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def depth_of(model_name):
    return int(model_name.split("_")[1])   # "model_10_3" -> 10


def print_ranking(title, scores, names):
    ranked = sorted((n for n in names if n in scores), key=scores.get)
    print(f"\n{title}")
    for place, name in enumerate(ranked, 1):
        print(f"  {place:2d}. {name:<12} {scores[name]:.6f}")
    return ranked


def choose_models(args, parser):
    """Return the list of models to train on this level."""
    pool = [n for n in MODEL_REGISTRY if depth_of(n) in args.depths]

    if args.models:
        unknown = [n for n in args.models if n not in MODEL_REGISTRY]
        if unknown:
            parser.error(f"unknown model(s): {unknown}")
        return args.models

    if args.level == 1:
        if args.top is not None:
            parser.error("--top ranks the previous level's scores, and level 1 has no previous level")
        return pool

    if args.top is None:
        parser.error(f"level {args.level} needs --top N (how many models to keep from "
                     f"level {args.level - 1}), or an explicit list with --models")

    previous_file = results_path(args.level - 1, args.tag)
    previous = load_scores(previous_file)
    candidates = [n for n in pool if n in previous]
    if not candidates:
        parser.error(f"no scores for these models in {previous_file}; run level {args.level - 1} first")

    ranked = print_ranking(f"Level {args.level - 1} ranking ({len(candidates)} models, from {previous_file}):",
                           previous, candidates)
    if len(ranked) < args.top:
        print(f"\nWARNING: only {len(ranked)} models have a level {args.level - 1} score; --top {args.top} keeps all of them.")
    return ranked[:args.top]


def time_models(models, level, epochs):
    """Run one epoch per model without saving anything and estimate the full run."""
    start = time.time()
    load_level_data(level=level, val_fold=5)
    load_seconds = time.time() - start
    print(f"\nLoading the level {level} data takes {load_seconds:.0f} s (repeated for every model).")
    print(f"\n{'model':<12} {'1 epoch':>10} {f'{epochs} epochs':>12}")

    total = 0.0
    for name in models:
        start = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            train_one_model(name, level=level, epochs=1, save=False)
        epoch_seconds = max(time.time() - start - load_seconds, 0.0)
        full_seconds = load_seconds + epoch_seconds * epochs
        total += full_seconds
        print(f"{name:<12} {epoch_seconds:>8.0f} s {full_seconds / 60:>8.1f} min", flush=True)

    print(f"\nEstimated total for {len(models)} models on level {level}: {total / 3600:.1f} hours")


def main():
    parser = argparse.ArgumentParser(description="Train a group of models on one level and rank them.")
    parser.add_argument("--level", type=int, required=True, help="Level to train on (1, 2, or 3)")
    parser.add_argument("--top", type=int, default=None,
                        help="Keep the N best models from the previous level (needed for level 2 and 3)")
    parser.add_argument("--models", nargs="+", default=None, help="Train exactly these models")
    parser.add_argument("--depths", nargs="+", type=int, default=DEFAULT_DEPTHS,
                        help=f"Model depths to include (default: {DEFAULT_DEPTHS})")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--tag", type=str, default=DEFAULT_TAG,
                        help=f"Name used in the results file (default: {DEFAULT_TAG})")
    parser.add_argument("--force", action="store_true", help="Retrain models that already have a score")
    parser.add_argument("--time", action="store_true",
                        help="Only time one epoch per model; nothing is saved")
    parser.add_argument("--data-root", type=str, default="maps2d", help="Data folder to use")
    parser.add_argument("--prefix", type=str, default="aug_", help="Prefix for saved .pth files")
    args = parser.parse_args()

    models = choose_models(args, parser)

    if args.time:
        time_models(models, args.level, args.epochs)
        return

    out_file = results_path(args.level, args.tag)
    done = load_scores(out_file)
    todo = [n for n in models if args.force or n not in done]
    skipped = [n for n in models if n not in todo]

    print(f"\nLevel {args.level}: {len(models)} models selected, {len(todo)} to train, scores -> {out_file}")
    if skipped:
        print(f"Skipping (already scored, use --force to retrain): {', '.join(skipped)}")

    failed = []
    for i, name in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {name}")
        start = time.time()
        try:
            train_one_model(name, level=args.level, epochs=args.epochs, results_file=out_file, data_root=args.data_root, weight_prefix=args.prefix)
        except Exception:
            traceback.print_exc()
            failed.append(name)
            continue
        print(f"[{i}/{len(todo)}] {name} finished in {(time.time() - start) / 60:.1f} min")

    print_ranking(f"Level {args.level} ranking (from {out_file}):", load_scores(out_file), models)

    if failed:
        print(f"\nFAILED ({len(failed)}): {', '.join(failed)}. Fix the error above, then run the command again without --force to train only these.")
        sys.exit(1)


if __name__ == "__main__":
    main()