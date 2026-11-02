# MLP Deep Learning Branch — House-Price Regression

Copied from [`Focus-In-Self-Coding/P4AI-DS/dl_mlp_project`](https://github.com/Focus-In-Self-Coding/P4AI-DS/tree/d90149bcc93e8d9debfc6a03128d690f957c1d45/dl_mlp_project) at commit `d90149b`. Changes since that commit:

- The CV `KFold` and both 15% early-stopping splits (inside each CV fold, and for the final model) use `random_state=SPLIT_SEED` (42) instead of `SEED` (36), so they select the same rows as FT-Transformer and TabNet in `dl/`. MLP fold `k` (numbered from 1) is fold `k − 1` in `dl/splits/<dataset>/folds.csv`.
- The CSV paths in `check_datasets.py` and the notebook point to the repository root, where both CSVs live in this repo. The Windows `.bat` launchers, `START_HERE.txt` and `data/README.md` were not copied: this repo uses the `house-dl` conda environment and the root CSVs instead.

The code implements a feed-forward Multilayer Perceptron (MLP) on both datasets:

1. [Housing Prices Dataset (Kaggle)](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)
2. [King County House Sales Prediction (Kaggle)](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)

## Files

- `dl_mlp_experiment.py` — main training, CV tuning, evaluation and artifact-generation program.
- `check_datasets.py` — checks whether both CSV files exist, can be read, and contain a usable `price` target.
- `smoke_test.py` — short end-to-end test using synthetic data only; its metrics must never be reported as project results.
- `requirements.txt` — Python dependencies (the `house-dl` environment from the root README already satisfies them).
- `MLP_House_Prices_Experiments.ipynb` — optional Jupyter launcher for the same main experiment.

## Running in this repo

From `dl_mlp_project/`, with the `house-dl` environment activated:

```bash
python check_datasets.py      # reads ../Housing.csv and ../kc_house_data.csv
python smoke_test.py          # synthetic data only; never report these scores
python dl_mlp_experiment.py --housing_csv ../Housing.csv --king_county_csv ../kc_house_data.csv --output_dir results_mlp
```

Add `--cpu` to force CPU mode. The complete official experiment is compute-intensive: two datasets, five configurations per dataset, five CV folds per configuration, early stopping, final fitting and feature importance. `results_mlp/` is tracked in git, like `dl/results/`.

## Experimental protocol

- Outer train/test split: 80% / 20%; `random_state=42`.
- Shuffled 5-fold CV: `random_state=42`, giving the same folds as `dl/splits/<dataset>/folds.csv`.
- 15% early-stopping splits (in each CV fold and for the final model): `random_state=42`.
- Model seed: `36`.
- Hyperparameter tuning: five-fold cross-validation on the outer training partition only.
- Early stopping uses an inner validation subset from the fold's fitting data; the CV scoring fold is excluded from model fitting and epoch selection.
- Final model early stopping uses a validation subset taken only from the training partition. The held-out test set is not used to choose the configuration or epoch.
- Numeric preprocessing: median imputation and standardization.
- Categorical preprocessing: most-frequent imputation and one-hot encoding.
- `id` is dropped, King County `date` is transformed into `sale_year`, `sale_month`, and `sale_dayofweek`, and `zipcode` is treated as categorical.
- The target is transformed with `log1p(price)` and standardized using the current model-fit subset. Predictions are returned to the original price scale before calculating MAE, RMSE and R².
- The selected configuration is the one with the lowest mean five-fold CV RMSE in original price units.
- Final reporting metrics: MAE, RMSE, R², final model-fitting time and total CV-search wall time.
- Permutation feature importance is calculated on a validation subset of the outer training data, never on the held-out test set.

## MLP configurations

| Configuration | Hidden layers | Dropout | Learning rate | Weight decay | Batch size |
|---|---:|---:|---:|---:|---:|
| 1 | 64, 32 | 0.00 | 0.001 | 0.0001 | 32 |
| 2 | 128, 64 | 0.10 | 0.001 | 0.0001 | 64 |
| 3 | 128, 64, 32 | 0.20 | 0.0005 | 0.001 | 128 |
| 4 | 256, 128 | 0.10 | 0.0005 | 0.0001 | 128 |
| 5 | 64, 32 | 0.20 | 0.001 | 0.01 | 64 |

All configurations use ReLU hidden activations, a one-value regression output, AdamW, mean squared error, a ReduceLROnPlateau scheduler, gradient clipping and early stopping (up to 200 epochs, patience 22).

## Output files after the official run

The program creates `results_mlp/` and writes:

### Root files

- `combined_test_summary.csv` — one row per dataset with selected config, CV RMSE, test MAE/RMSE/R² and timing.
- `experiment_protocol.json` — split, seeds, hyperparameter configurations and software versions.

### Inside each dataset subfolder

- `cv_results.csv` — mean and standard deviation of metrics for each configuration across CV folds.
- `cv_fold_details.csv` — each configuration/fold score, epoch and fit time.
- `best_hyperparameters.json` — selected configuration and final best epoch.
- `best_config_and_metrics.json` — selected configuration, validation/test metrics and timing.
- `split_manifest.csv` — original row indices assigned to train or test.
- `test_predictions.csv` — test actual price, predicted price and residual.
- `final_training_history.csv` — training/validation loss by epoch.
- `validation_permutation_importance.csv` — validation-RMSE increase after feature shuffling.
- `best_mlp_model.pt` — saved PyTorch model weights and target-scaling information.
- `preprocessor.joblib` — fitted preprocessing pipeline.
- `figures/` — CV comparison, actual-vs-predicted, residual and permutation-importance plots.
