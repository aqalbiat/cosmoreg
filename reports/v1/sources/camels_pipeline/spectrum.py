from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FourierGeometry:
    k: np.ndarray
    bin_edges: np.ndarray
    mode_counts: np.ndarray
    bin_index: np.ndarray
    radial_mask: np.ndarray
    k_grid: np.ndarray


def make_geometry(n: int, box_size: float, n_bins: int) -> FourierGeometry:
    if n <= 1:
        raise ValueError("n must be > 1.")
    kfreq = 2.0 * np.pi * np.fft.fftfreq(n, d=box_size / n)
    kx, ky = np.meshgrid(kfreq, kfreq, indexing="ij")
    k_grid = np.sqrt(kx**2 + ky**2)
    k_fundamental = 2.0 * np.pi / box_size
    k_nyquist_axis = np.pi * n / box_size
    bin_edges = np.linspace(k_fundamental, k_nyquist_axis, n_bins + 1, dtype=np.float64)
    radial_mask = (k_grid >= bin_edges[0]) & (k_grid <= bin_edges[-1])
    radial_mask[0, 0] = False
    flat_k = k_grid.ravel()
    flat_mask = radial_mask.ravel()
    bin_index = np.full(flat_k.shape, -1, dtype=np.int64)
    selected_bins = np.searchsorted(bin_edges, flat_k[flat_mask], side="right") - 1
    selected_bins = np.minimum(selected_bins, n_bins - 1)
    bin_index[flat_mask] = selected_bins
    mode_counts = np.bincount(selected_bins, minlength=n_bins).astype(np.int64)
    if np.any(mode_counts == 0):
        empty = np.flatnonzero(mode_counts == 0).tolist()
        raise ValueError(f"Empty radial Fourier bins for n={n}, n_bins={n_bins}: {empty}")
    k = np.bincount(selected_bins, weights=flat_k[flat_mask], minlength=n_bins) / mode_counts
    return FourierGeometry(k=k, bin_edges=bin_edges, mode_counts=mode_counts, bin_index=bin_index, radial_mask=radial_mask, k_grid=k_grid)


def density_contrast(raw_map: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw_map, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Map contains non-finite values.")
    mean = float(arr.mean())
    if mean <= 0.0:
        raise ValueError("Map mean must be strictly positive.")
    return arr / mean - 1.0


def modal_power_from_delta(delta: np.ndarray, box_size: float) -> np.ndarray:
    n = int(delta.shape[0])
    if delta.ndim != 2 or delta.shape[0] != delta.shape[1]:
        raise ValueError("Power spectrum requires a square 2D map.")
    fft = np.fft.fft2(delta)
    return (box_size**2 / n**4) * np.abs(fft) ** 2


def parseval_variance(delta: np.ndarray, modal_power: np.ndarray, box_size: float) -> tuple[float, float]:
    return float(modal_power.sum() / box_size**2), float(np.mean(delta**2))


def power_spectrum(raw_map: np.ndarray, geometry: FourierGeometry, box_size: float) -> np.ndarray:
    delta = density_contrast(raw_map)
    modal_power = modal_power_from_delta(delta, box_size)
    flat_power = modal_power.ravel()
    selected = geometry.bin_index >= 0
    sums = np.bincount(geometry.bin_index[selected], weights=flat_power[selected], minlength=len(geometry.mode_counts))
    spectrum = sums / geometry.mode_counts
    if not np.all(np.isfinite(spectrum)):
        raise ValueError("Computed non-finite power spectrum.")
    return spectrum.astype(np.float64)


def spectra_for_batch(batch: np.ndarray, geometry: FourierGeometry, box_size: float) -> np.ndarray:
    out = np.empty((len(batch), len(geometry.mode_counts)), dtype=np.float64)
    for idx, raw_map in enumerate(batch):
        out[idx] = power_spectrum(raw_map, geometry, box_size)
    return out
