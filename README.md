# CAMELS Mtot Data Preparation

This repository prepares data for the CSCI 494 project "Inferring Cosmological Parameters from Simulated Matter Density Fields with Convolutional Neural Networks". The current scope is data extraction, validation, simulation-grouped splits, and matching 2D power spectra. It does not train Ridge, MLP, or CNN models.

## Source

Default source: CAMELS Multifield Dataset, `Maps_Mtot_IllustrisTNG_LH_z=0.00.npy` plus `params_LH_IllustrisTNG.txt`.

The official CMD documentation says LH contains 1,000 simulations, 2D map files contain 15,000 maps of shape 256 by 256, and map `i` uses parameter row `i // 15`. The maps cover a periodic 25 h^-1 Mpc square at redshift 0. The first two parameter columns are `Omega_m` and `sigma_8`; the remaining four columns are kept as auxiliary metadata.

The full Mtot map file is about 3.66 GiB / 3.93 decimal GB, so even a 250-simulation pilot may require obtaining the full source file unless you already have a local subset with the same layout.

## Install

```bash
python -m pip install -r requirements.txt
```

## Commands

Download the verified default files:

```bash
python -m camels_pipeline.cli download --config config.yaml
```

Prepare the full three-level export from downloaded or configured local files:

```bash
python -m camels_pipeline.cli prepare --config config.yaml
```

Use local files explicitly:

```bash
python -m camels_pipeline.cli prepare --config config.yaml --maps "D:/path/Maps_Mtot_IllustrisTNG_LH_z=0.00.npy" --params "D:/path/params_LH_IllustrisTNG.txt"
```

Prepare a separate random 250-simulation pilot export:

```bash
python -m camels_pipeline.cli prepare --config config.yaml --subset-250
```

Run a tiny synthetic smoke test. These outputs are software fixtures, not scientific results:

```bash
python -m camels_pipeline.cli smoke-test --output synthetic_outputs
```

Audit paired outputs:

```bash
python -m camels_pipeline.cli audit --maps-root maps2d --spectra-root powerspectrum
```

Run tests:

```bash
pytest
```

## Split Contract

The original README requested 1,000 / 5,000 / 9,000 maps. That cannot be done without splitting a simulation because each LH simulation contributes exactly 15 related maps. This implementation preserves complete simulations and uses:

| Level | Simulations | Maps | Test simulations | Fold simulations |
|---|---:|---:|---:|---|
| level1 | 67 | 1,005 | 13 | 11, 11, 11, 11, 10 |
| level2 | 333 | 4,995 | 67 | 54, 53, 53, 53, 53 |
| level3 | 600 | 9,000 | 120 | 96, 96, 96, 96, 96 |

Splits are by simulation, never by individual map. A simulation appears in exactly one level and one group. Folds are development partitions: train on four folds and validate on one. Test groups are held out from tuning.

## Output Layout

```text
maps2d/level{1,2,3}/{fold1,fold2,fold3,fold4,fold5,test}/
powerspectrum/level{1,2,3}/{fold1,fold2,fold3,fold4,fold5,test}/
```

Each `maps2d` leaf contains:

- `maps.npy`: `[M, 256, 256]` float32
- `labels.npy`: `[M, 2]` float32, columns `[Omega_m, sigma_8]`
- `simulation_ids.npy`
- `map_ids.npy`
- `manifest.json`

Each matching `powerspectrum` leaf contains:

- `power_spectra.npy`: `[M, 50]`
- `log_power_spectra.npy`: `[M, 50]`
- paired `labels.npy`, `simulation_ids.npy`, `map_ids.npy`
- `k.npy`, `bin_edges.npy`, `mode_counts.npy`
- `manifest.json`

Generated data arrays and manifests are ignored by git. Source code, tests, configuration, and documentation are versionable.

## Power Spectrum Convention

For each raw map `rho`, the code computes `delta = rho / mean(rho) - 1` before any log transform or learned normalization. It then uses NumPy's full `fft2` and

```text
P2D = (L^2 / N^4) * |FFT(delta)|^2
```

with `L = 25 Mpc/h` and `N = 256` for the real CAMELS maps. The DC mode is excluded. Fifty fixed linear radial bins span the fundamental mode `2*pi/L` through the axis Nyquist mode `pi*N/L`; modes beyond the axis Nyquist are excluded. Saved `k` values are the mean mode magnitude in each bin. `log_power_spectra.npy` uses `log10(max(P, 1e-30))`.

This is a 2D projected-field statistic, not CAMELS' supplied 3D matter power spectrum. Learned scaling for models must be fitted later on training folds only, never on validation or test data.

## Key Code Paths

- `camels_pipeline/config.py`: defaults and config validation
- `camels_pipeline/download.py`: safe URL download, temporary files, HTML rejection, computed SHA-256 provenance
- `camels_pipeline/dataset.py`: memory-mapped loading and source validation
- `camels_pipeline/splits.py`: deterministic simulation-level levels, folds, and pilot subset
- `camels_pipeline/spectrum.py`: density contrast and Fourier/radial power spectrum
- `camels_pipeline/export.py`: bounded-batch paired export and manifests
- `camels_pipeline/audit.py`: output row alignment and leakage checks
- `data.ipynb`: notebook frontend to the same modules

## AI Assistance

This preprocessing pipeline was implemented with AI assistance. Students remain responsible for understanding, checking, and defending the code, data loading assumptions, splits, preprocessing, and any later model evaluation. No experimental model results are claimed here.
