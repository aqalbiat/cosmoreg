import numpy as np

from camels_pipeline.spectrum import density_contrast, make_geometry, modal_power_from_delta, parseval_variance, power_spectrum


def test_parseval_for_full_fft_modal_power():
    n = 16
    box = 25.0
    raw = 2.0 + np.arange(n * n, dtype=np.float64).reshape(n, n) / 100.0
    delta = density_contrast(raw)
    modal_power = modal_power_from_delta(delta, box)
    left, right = parseval_variance(delta, modal_power, box)
    assert np.isclose(left, right, rtol=1e-12, atol=1e-12)


def test_constant_map_has_zero_power_and_finite_log_input():
    geom = make_geometry(16, 25.0, 5)
    spectrum = power_spectrum(np.ones((16, 16), dtype=np.float32), geom, 25.0)
    assert np.all(spectrum == 0.0)
    assert np.all(np.isfinite(np.log10(np.maximum(spectrum, 1e-30))))


def test_single_fourier_mode_lands_in_expected_bin():
    n = 16
    box = 25.0
    amplitude = 0.2
    mode = 2
    x = np.arange(n)
    xx, _ = np.meshgrid(x, x, indexing="ij")
    raw = 1.0 + amplitude * np.cos(2 * np.pi * mode * xx / n)
    geom = make_geometry(n, box, 5)
    spectrum = power_spectrum(raw, geom, box)
    delta = density_contrast(raw)
    modal = modal_power_from_delta(delta, box)
    target_k = 2 * np.pi * mode / box
    target_bin = np.searchsorted(geom.bin_edges, target_k, side="right") - 1
    expected = modal[mode, 0] * 2 / geom.mode_counts[target_bin]
    assert np.isclose(spectrum[target_bin], expected, rtol=1e-12, atol=1e-12)
    other = np.delete(spectrum, target_bin)
    assert np.allclose(other, 0.0, atol=1e-12)


def test_periodic_translation_and_90_degree_rotation_invariance():
    n = 16
    box = 25.0
    rng = np.random.default_rng(5)
    raw = 2.0 + rng.random((n, n), dtype=np.float64)
    geom = make_geometry(n, box, 5)
    base = power_spectrum(raw, geom, box)
    shifted = power_spectrum(np.roll(raw, shift=(3, 5), axis=(0, 1)), geom, box)
    rotated = power_spectrum(np.rot90(raw), geom, box)
    assert np.allclose(base, shifted, rtol=1e-12, atol=1e-12)
    assert np.allclose(base, rotated, rtol=1e-12, atol=1e-12)


def test_bins_are_finite_and_nonempty():
    geom = make_geometry(16, 25.0, 5)
    assert np.all(geom.mode_counts > 0)
    assert np.all(np.isfinite(geom.k))
