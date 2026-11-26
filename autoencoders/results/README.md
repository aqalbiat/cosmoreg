# Autoencoder result figures

Figures from the completed, audited `sweep30` experiment: three levels, D=10,30,50,70,90, 75 CV encoders and 15 final refits. No retraining or model selection was performed during export.

Each figure is available as a PNG and a PDF. [All figures in one PDF](all_figures.pdf).

## Interpretation

The locked CV-selected dimensions are level 1: D=90, level 2: D=90, and level 3: D=10. Larger latent dimensions improve CV reconstruction, but prediction CV scores remain within 0.4% of the training-mean baseline. These small differences do not establish useful predictive gains or statistical significance. All selected Ridge alphas are 1000, the upper grid boundary.

The level-3 D=70 final refit has normalized development reconstruction MSE approximately 1. Its test scores and reconstruction examples are retained as a disclosed diagnostic; test errors were not used to replace the locked level-3 D=10 winner.

Unless a figure explicitly says per simulation, prediction errors are per map in original target units. Maps from the same simulation are correlated. CV figures average fold metrics; the validation fold also supplies reconstruction early stopping, so CV is a tuning estimate. Fold curves are individual fits, not confidence intervals. Reconstruction errors use density contrast, without logarithms. The mean-prediction baseline is fitted only on development labels.

## Figures

| Figure | Description |
|---|---|
| [01_cv_normalized_mae_vs_dimension](01_cv_normalized_mae_vs_dimension.png) ([PDF](01_cv_normalized_mae_vs_dimension.pdf)) | Development-only CV comparison. Small differences do not establish statistical significance; no confidence intervals or repeated seeds are available. |
| [02_cv_improvement_percent_vs_dimension](02_cv_improvement_percent_vs_dimension.png) ([PDF](02_cv_improvement_percent_vs_dimension.pdf)) | Development-only CV comparison. Small differences do not establish statistical significance; no confidence intervals or repeated seeds are available. |
| [03_ridge_alpha_sensitivity](03_ridge_alpha_sensitivity.png) ([PDF](03_ridge_alpha_sensitivity.pdf)) | Alpha is tuned separately for every D and level using five-fold CV. Dashed lines show the development-only mean baseline. |
| [04_cv_prediction_errors](04_cv_prediction_errors.png) ([PDF](04_cv_prediction_errors.pdf)) | All D/level combinations are retained. CV values average fold metrics. The unselected level-3 D=70 final refit learned almost no reconstruction; its lower test error is not grounds to change the locked D=10 selection. |
| [05_test_prediction_errors](05_test_prediction_errors.png) ([PDF](05_test_prediction_errors.pdf)) | All D/level combinations are retained. CV values average fold metrics. The unselected level-3 D=70 final refit learned almost no reconstruction; its lower test error is not grounds to change the locked D=10 selection. |
| [06_selected_test_errors_vs_baseline](06_selected_test_errors_vs_baseline.png) ([PDF](06_selected_test_errors_vs_baseline.pdf)) | Per-map errors in original units. The baseline is estimated from all five development folds, independently of test labels. |
| [07_selected_predictions_per_map](07_selected_predictions_per_map.png) ([PDF](07_selected_predictions_per_map.pdf)) | Predictions use the locked final models and saved scalers. Maps in each simulation share labels and are correlated. Shared target axis limits make shrinkage toward the development mean visible. |
| [08_selected_predictions_per_simulation](08_selected_predictions_per_simulation.png) ([PDF](08_selected_predictions_per_simulation.pdf)) | Predictions use the locked final models and saved scalers. Map predictions are averaged by simulation ID before computing these metrics; no model is refitted. |
| [09_selected_test_residuals](09_selected_test_residuals.png) ([PDF](09_selected_test_residuals.pdf)) | The same saved predictions as the scatter figures; baseline and model distributions use identical bins within each panel. |
| [10_reconstruction_rmse_vs_dimension](10_reconstruction_rmse_vs_dimension.png) ([PDF](10_reconstruction_rmse_vs_dimension.pdf)) | Stars mark CV-selected predictive models, which need not have the lowest reconstruction error. The weak D=70 level-3 final refit is retained. |
| [11_reconstruction_prediction_tradeoff](11_reconstruction_prediction_tradeoff.png) ([PDF](11_reconstruction_prediction_tradeoff.pdf)) | Lower values are better on both axes. Dashed lines show the training-mean prediction baseline; no test data enter this plot. |
| [12_all_cv_learning_curves](12_all_cv_learning_curves.png) ([PDF](12_all_cv_learning_curves.pdf)) | All 75 CV histories are shown individually. Each encoder has its own training-fitted normalization. Early stopping uses reconstruction validation MSE. |
| [13_final_refit_learning_curves](13_final_refit_learning_curves.png) ([PDF](13_final_refit_learning_curves.pdf)) | Level 3, D=70 stays near the mean-map reference. Final fits retain their last epoch after the CV-prescribed duration; no test-based checkpoint selection occurs. Training curves show online epoch loss while parameters are updated, so they need not equal frozen-checkpoint export metrics. |
| [14_final_refit_reconstruction_diagnostic](14_final_refit_reconstruction_diagnostic.png) ([PDF](14_final_refit_reconstruction_diagnostic.pdf)) | Level 3, D=70 stays near the mean-map reference. Final fits retain their last epoch after the CV-prescribed duration; no test-based checkpoint selection occurs. Values are exported development reconstruction MSE divided by the training pixel variance. |
| [15_cv_epochs_and_final_refit_budget](15_cv_epochs_and_final_refit_budget.png) ([PDF](15_cv_epochs_and_final_refit_budget.pdf)) | Five folds determine each median duration. Horizontal offsets distinguish overlapping fold markers and do not represent different D values. |
| [16_selected_reconstructions_level1_D90](16_selected_reconstructions_level1_D90.png) ([PDF](16_selected_reconstructions_level1_D90.pdf)) | Examples use the first map from the first, middle and last sorted test simulation IDs, independent of reconstruction quality. The checkpoint is frozen; CPU inference only. Original/reconstruction colors share [-1, joint 99.5th percentile], and residual colors use symmetric 99.5th-percentile absolute residual limits. Full-map RMSE includes all pixels. Box side is 25 Mpc/h for this saved experiment. |
| [17_selected_reconstructions_level2_D90](17_selected_reconstructions_level2_D90.png) ([PDF](17_selected_reconstructions_level2_D90.pdf)) | Examples use the first map from the first, middle and last sorted test simulation IDs, independent of reconstruction quality. The checkpoint is frozen; CPU inference only. Original/reconstruction colors share [-1, joint 99.5th percentile], and residual colors use symmetric 99.5th-percentile absolute residual limits. Full-map RMSE includes all pixels. Box side is 25 Mpc/h for this saved experiment. |
| [18_selected_reconstructions_level3_D10](18_selected_reconstructions_level3_D10.png) ([PDF](18_selected_reconstructions_level3_D10.pdf)) | Examples use the first map from the first, middle and last sorted test simulation IDs, independent of reconstruction quality. The checkpoint is frozen; CPU inference only. Original/reconstruction colors share [-1, joint 99.5th percentile], and residual colors use symmetric 99.5th-percentile absolute residual limits. Full-map RMSE includes all pixels. Box side is 25 Mpc/h for this saved experiment. |
| [19_weak_refit_reconstructions_level3_D70](19_weak_refit_reconstructions_level3_D70.png) ([PDF](19_weak_refit_reconstructions_level3_D70.pdf)) | Examples use the first map from the first, middle and last sorted test simulation IDs, independent of reconstruction quality. The checkpoint is frozen; CPU inference only. Original/reconstruction colors share [-1, joint 99.5th percentile], and residual colors use symmetric 99.5th-percentile absolute residual limits. Full-map RMSE includes all pixels. Box side is 25 Mpc/h for this saved experiment. |

## Data and provenance

- [metrics.csv](metrics.csv): all 15 CV/test combinations, baseline errors, CV selection and refit diagnostics.
- [test_predictions.csv](test_predictions.csv): predictions, labels, map/simulation IDs and baseline for all dimensions.
- [learning_curves.csv](learning_curves.csv): all 75 CV and 15 final training histories.
- [ridge_alpha_scores.csv](ridge_alpha_scores.csv): all predefined alpha candidates.
- [reconstruction_examples.csv](reconstruction_examples.csv): fixed example identities, full-image RMSE and display limits.
- [manifest.json](manifest.json): source and output SHA-256 hashes, software versions, locked winners and export settings.

Scatter figures use saved test predictions. Simulation figures average those predictions by simulation ID. Reconstruction example figures run the frozen checkpoints on three fixed test maps per level; no fitted state or experimental results are overwritten. Their image color limits clip extreme values for legibility, while RMSE includes every pixel. These examples are illustrative, not aggregate estimates.

## Reproduce

```powershell
python -m autoencoders.plot_results
```

Requires the original ignored run artifacts, hidden codes/labels and map exports in their saved locations. The exported figures and CSV files can be shared independently. Dependencies: `python -m pip install -r autoencoders/requirements.txt`.

The exporter verifies completion, the saved passing audit and summary evidence hashes, source label/ID hashes, locked selections, saved prediction errors for every D/level, refit diagnostics, and reconstruction checkpoint identities. It does not repeat the full scientific artifact audit.

Training sources match the recorded training source hashes after normalizing checkout line endings to LF. Both recorded and current source hashes are retained in the manifest. No training source file was edited during this export.
