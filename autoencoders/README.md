# Convolutional autoencoder features

Run from the repository root:

```powershell
python -m pip install -r autoencoders/requirements.txt
New-Item -ItemType Directory -Force autoencoders/runs
python -m pytest autoencoders/test_pipeline.py --basetemp autoencoders/runs/test-fixtures -o cache_dir=autoencoders/.pytest_cache -q
python -u -m autoencoders.experiment
python -m autoencoders.verify
python -m autoencoders.summarize
```

The default experiment runs all three existing levels independently, first D=30
across all levels to verify the pipeline, then D=10,50,70,90. It reuses the exact
five development folds and separate test group, without subsampling or
resplitting. The user-selected cap is 30 epochs, with early-stopping patience 10.
Adam uses learning rate 0.001 and batch size 32. The seven predefined Ridge
alphas are 0.001, 0.01, 0.1, 1, 10, 100, 1000. Seed 42 determines fresh,
independent model initialization and epoch shuffles. CUDA mixed precision is
enabled when available; exports and final evaluation use float32.

Architecture: four 3×3 encoder convolutions (stride 2, padding 1), channels
1→8→16→32→32, each followed by ReLU. The 32×16×16 activation is flattened to
8192 and projected by Linear(8192,D). Linear(D,8192) reshapes directly to
32×16×16. Four nearest-neighbor 2× upsampling blocks use 3×3 convolutions
(stride 1, padding 1), channels 32→32→16→8→1. Hidden decoder convolutions use
ReLU; both dense layers, the latent code and final reconstruction are linear.
Only the two dense layers vary with D. No labels enter autoencoder training.

Each raw map becomes δ=ρ/mean(ρ)−1, matching the repository's density-contrast
convention. No logarithm or clipping is applied. Each fold encoder uses one
global pixel mean and standard deviation fitted only on its four training
folds. It minimizes pixel MSE against that normalized density-contrast map.
Validation reconstruction MSE chooses the best epoch (first minimum). After
freezing that encoder, training and validation maps are exported separately.
Reconstruction MAE, MSE and RMSE are converted back to density-contrast units.
Raw-density reconstruction, if needed, additionally requires the original map
mean; the encoder reconstructs density contrast as requested.

A latent StandardScaler is fitted on each fold's training codes. Each Ridge
alpha is fitted on those scaled codes and the original labels, then scored on
the corresponding validation codes. The selection score is the arithmetic
mean over all five folds and both targets of MAE divided by that fold's
training-label population standard deviation. Targets have equal weight.
Alpha is selected independently per D and level. Ties choose smaller alpha.
If choosing one D per level, use the lowest selected CV score, breaking ties
with smaller D. Codes from different fold encoders are never pooled for Ridge.

All CV completes before `settings_lock.json` records alpha, D winners, and
the median best CV epoch for each refit. Each fresh final encoder trains on
all five development folds for exactly that median epoch count. A new latent
scaler and Ridge are fitted on its development codes. Test maps and labels
enter the model pipeline only after the lock; each test set uses the same
final encoder and scaler as its development set. Input hashes, test identities
and split leakage are audited before training; test contents never influence
selection. MAE and RMSE for Omega_m and sigma_8 are reported separately in
original units. Metrics are per map; simulations remain intact in all splits.
Fold RMSE values are averaged as fold metrics, not pooled residuals.

Outputs:

```text
autoencoders/cache/levelN/{fold1..fold5,test}/density_contrast.npy
autoencoders/runs/sweep30/
  experiment.json              # settings, software-source and input hashes
  D30_verification.json         # written after the first D=30 CV pass
  settings_lock.json           # written before refits and test evaluation
  levelN/D=N/foldN/             # best checkpoint, epoch history, Ridge scores
  levelN/D=N/cross_validation.json
  levelN/D=N/final/             # final encoder, latent scaler/Ridge, test scores
  REPORT.md                    # CV comparison and both test-target errors
  INTERPRETATION.md            # CV-selected test summary and baseline comparison
  summary.json                 # independently recomputed development-only baseline
  results.json                 # full machine-readable results
  completed.json               # created only after all evaluations finish
autoencoders_hidden/sweep30/D=N/levelN/
  foldN/{train,validation}/
  final/{development,test}/
```

Each code leaf includes `codes.npy`, `labels.npy`, `map_ids.npy`,
`simulation_ids.npy`, `source_folds.npy` and a manifest linking it to its encoder
hash, preprocessing, source data and training-fold identities. All generated
artifacts are ignored by git. Existing empty `H=N` folders are left intact.

Re-run the identical command to resume: checkpoints save model, optimizer,
mixed-precision scaler, RNG state and epoch history after every epoch.
Completed folds and refits are reused. Settings, input hashes and source-code
hashes must match; changes require a different `--run-root` under
`autoencoders/`. The hidden export run name follows its run directory name.
Do not run simultaneous writers against the same run directory.

To launch the full run in the background on Windows without opening a window:

```powershell
$trainingProcess = Start-Process -FilePath (Get-Command python).Source -ArgumentList '-u','-m','autoencoders.experiment' -WorkingDirectory (Get-Location).Path -WindowStyle Hidden -RedirectStandardOutput 'autoencoders/runs/sweep30/stdout.log' -RedirectStandardError 'autoencoders/runs/sweep30/stderr.log' -PassThru
$trainingProcess.Id
```

Create the run directory before using those log redirections. Follow progress
in `stdout.log` and epoch histories. Scientific results are valid only once
the run has `completed.json`. Synthetic tests are software checks, not results.

`python -m autoencoders.verify` independently audits the completed artifacts and
writes `audit.json`: encoder hashes, array hashes, row alignment with the source,
split separation, early-stopping epochs, alpha scores, median refit epochs,
development-only latent scaling, and recomputed test metrics. Use
`--run-root autoencoders/runs/other-run` to audit a separate run. Run it after
training finishes. Numerical experiment settings are recorded in
`experiment.json`; the executed run also has `environment.json` for library and
GPU versions, plus `global_split_audit.json` for cross-level simulation checks.

After the audit passes, `python -m autoencoders.summarize` reproduces the
training-mean baseline using development labels only, checks the earlier saved
baseline when present, and writes `INTERPRETATION.md` and `summary.json`. It also
links that interpretation from `REPORT.md`. The summary identifies the winners
using the locked CV selection and reports their original-unit test errors;
all dimensions remain in the full report. It does not retrain or reevaluate
test maps. It also records final development reconstruction loss and latent
variation, flagging refits whose normalized reconstruction MSE stays at least
0.99 (the training pixel-mean reference is 1). These reporting diagnostics do
not change the locked settings or remove weak fits. Rerun it after a driver
resume, which regenerates `REPORT.md`.
