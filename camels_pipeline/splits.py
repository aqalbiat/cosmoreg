from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PipelineConfig


@dataclass(frozen=True)
class SplitLeaf:
    level: str
    group: str
    simulation_ids: np.ndarray


def _even_split(ids: np.ndarray, n_parts: int) -> list[np.ndarray]:
    base = len(ids) // n_parts
    remainder = len(ids) % n_parts
    out = []
    start = 0
    for part in range(n_parts):
        size = base + (1 if part < remainder else 0)
        out.append(ids[start : start + size])
        start += size
    return out


def make_splits(n_simulations: int, config: PipelineConfig) -> dict[str, dict[str, np.ndarray]]:
    needed = sum(config.level_simulation_counts)
    if n_simulations < needed:
        raise ValueError(f"Need at least {needed} simulations for configured levels, got {n_simulations}.")
    rng = np.random.default_rng(config.seed)
    permuted = rng.permutation(np.arange(n_simulations, dtype=np.int64))
    splits: dict[str, dict[str, np.ndarray]] = {}
    start = 0
    for idx, count in enumerate(config.level_simulation_counts, start=1):
        level_ids = permuted[start : start + count]
        start += count
        level_ids = rng.permutation(level_ids)
        n_test = int(np.floor(config.test_fraction * count + 0.5))
        test_ids = np.sort(level_ids[:n_test])
        dev_ids = level_ids[n_test:]
        folds = _even_split(dev_ids, config.n_folds)
        level_name = f"level{idx}"
        splits[level_name] = {"test": test_ids}
        for fold_idx, fold_ids in enumerate(folds, start=1):
            splits[level_name][f"fold{fold_idx}"] = np.sort(fold_ids)
    audit_splits(splits)
    return splits


def audit_splits(splits: dict[str, dict[str, np.ndarray]]) -> None:
    seen: dict[int, str] = {}
    for level, groups in splits.items():
        level_seen: set[int] = set()
        for group, ids in groups.items():
            ids = np.asarray(ids, dtype=np.int64)
            if ids.ndim != 1:
                raise ValueError(f"{level}/{group} IDs must be one-dimensional.")
            if len(np.unique(ids)) != len(ids):
                raise ValueError(f"Duplicate simulation IDs inside {level}/{group}.")
            overlap = level_seen.intersection(ids.tolist())
            if overlap:
                raise ValueError(f"Simulation overlap inside {level}: {sorted(overlap)[:5]}")
            level_seen.update(ids.tolist())
            for sim_id in ids.tolist():
                previous = seen.get(int(sim_id))
                if previous is not None:
                    raise ValueError(f"Simulation {sim_id} appears in both {previous} and {level}/{group}.")
                seen[int(sim_id)] = f"{level}/{group}"


def split_summary(splits: dict[str, dict[str, np.ndarray]], maps_per_simulation: int) -> dict[str, dict[str, dict[str, int]]]:
    return {
        level: {
            group: {"simulations": int(len(ids)), "maps": int(len(ids) * maps_per_simulation)}
            for group, ids in groups.items()
        }
        for level, groups in splits.items()
    }


def make_subset_splits(
    n_simulations: int,
    subset_count: int,
    config: PipelineConfig,
    name: str = "subset_250",
) -> dict[str, dict[str, np.ndarray]]:
    if subset_count > n_simulations:
        raise ValueError(f"Cannot select {subset_count} simulations from {n_simulations}.")
    rng = np.random.default_rng(config.seed)
    selected = rng.choice(np.arange(n_simulations, dtype=np.int64), size=subset_count, replace=False)
    selected = rng.permutation(selected)
    n_test = int(np.floor(config.test_fraction * subset_count + 0.5))
    test_ids = np.sort(selected[:n_test])
    folds = _even_split(selected[n_test:], config.n_folds)
    splits = {name: {"test": test_ids}}
    for fold_idx, fold_ids in enumerate(folds, start=1):
        splits[name][f"fold{fold_idx}"] = np.sort(fold_ids)
    audit_splits(splits)
    return splits
